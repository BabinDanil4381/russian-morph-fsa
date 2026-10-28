"""
Обучение нейросети и честное сравнение трёх способов разбирать
НЕЗНАКОМЫЕ слова (их нет в обучающем словаре):

  1. Автомат окончаний       — самый частый класс для окончания (как pymorphy2)
  2. Нейросеть               — CharBiLSTM, выбирает из всех ~6700 классов
  3. Гибрид                  — нейросеть выбирает только из классов,
                               которые автомат окончаний считает возможными

    python train.py                 # полное обучение (~20-40 мин на CPU)
    python train.py --epochs 1 --limit 50000   # быстрый прогон для проверки
"""
import argparse
import json
import os
import random
import time

import torch

from data import apply_rule, load
from model import CharBiLSTM, Encoder, any_correct_loss
from suffix_automaton import SuffixAutomaton

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache")


def evaluate(pred_ids, test, classes):
    """pred_ids[i] — предсказанный класс для test[i] (или None)."""
    n = len(test)
    hits = {"pos": 0, "tag": 0, "lemma": 0, "full": 0}
    for cid, (w, _, parses) in zip(pred_ids, test):
        if cid is None:
            continue
        tag, cut, add = classes[cid]
        lemma = apply_rule(w, cut, add)
        gold_tags = {t for t, *_ in parses}
        hits["pos"] += tag.split(",")[0].split()[0] in {t.split(",")[0].split()[0] for t in gold_tags}
        hits["tag"] += tag in gold_tags
        hits["lemma"] += lemma in {l for *_, l in parses}
        hits["full"] += (tag, lemma) in {(t, l) for t, _, _, l in parses}
    return {k: round(100 * v / n, 2) for k, v in hits.items()}


@torch.no_grad()
def nn_logits(model, enc, words, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(words), bs):
        x, L = enc.batch(words[i:i + bs])
        out.append(model(x, L))
    return torch.cat(out)


def hybrid_predict(logits, cand_sets):
    preds = []
    for row, cands in zip(logits, cand_sets):
        if cands:
            ids = torch.tensor(cands)
            preds.append(int(ids[row[ids].argmax()]))
        else:
            preds.append(int(row.argmax()))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="взять только N обучающих слов")
    ap.add_argument("--eval-size", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--out", default=OUT, help="куда сохранить модель")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(os.cpu_count() or 2)

    ds = load()
    train, classes, chars = ds["train"], ds["classes"], ds["chars"]
    if args.limit:
        train = random.sample(train, args.limit)
    test = random.Random(1).sample(ds["test"], min(args.eval_size, len(ds["test"])))
    print(f"train: {len(train)} слов, test: {len(test)} незнакомых слов, классов: {len(classes)}")

    # ---------------- автомат окончаний
    t0 = time.time()
    sa = SuffixAutomaton(train, classes)
    cand = [sa.candidates(w)[0] for w, _, _ in test]
    cand_sets = [sorted(c) for c in cand]
    sa_preds = [c.most_common(1)[0][0] if c else None for c in cand]
    res_sa = evaluate(sa_preds, test, classes)
    avg_cands = sum(map(len, cand_sets)) / len(cand_sets)
    print(f"автомат окончаний: {sa.n_states()} состояний, построен за {time.time() - t0:.0f} c, "
          f"в среднем {avg_cands:.1f} возможных класса из {len(classes)}")
    print("  точность:", res_sa)

    # ---------------- нейросеть
    enc = Encoder(chars)
    model = CharBiLSTM(len(chars), len(classes))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"нейросеть: {n_params / 1e6:.2f} млн параметров")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    steps = args.epochs * ((len(train) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)

    history = []
    test_words = [w for w, _, _ in test]
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        t0, total, nb = time.time(), 0.0, 0
        for i in range(0, len(train), args.bs):
            chunk = train[i:i + args.bs]
            x, L = enc.batch([w for w, _, _ in chunk])
            gold = torch.zeros(len(chunk), len(classes), dtype=torch.bool)
            for r, (_, ids, _) in enumerate(chunk):
                gold[r, ids] = True
            loss = any_correct_loss(model(x, L), gold)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
            nb += 1
        logits = nn_logits(model, enc, test_words)
        r_nn = evaluate(logits.argmax(1).tolist(), test, classes)
        r_hy = evaluate(hybrid_predict(logits, cand_sets), test, classes)
        history.append({"epoch": epoch, "loss": round(total / nb, 4), "nn": r_nn, "hybrid": r_hy})
        print(f"эпоха {epoch}: loss={total / nb:.3f}  ({time.time() - t0:.0f} c)  "
              f"нейросеть full={r_nn['full']}  гибрид full={r_hy['full']}")

    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"state": model.state_dict(), "chars": chars, "classes": classes}, f"{OUT_DIR}/model.pt")
    with open(f"{OUT_DIR}/sa.pkl", "wb") as f:
        import pickle
        pickle.dump(sa, f)
    metrics = {
        "train_words": len(train), "test_words": len(test), "n_classes": len(classes),
        "params": n_params, "sa_states": sa.n_states(), "avg_candidates": round(avg_cands, 1),
        "suffix_automaton": res_sa, "neural": history[-1]["nn"], "hybrid": history[-1]["hybrid"],
        "history": history,
    }
    with open(f"{OUT_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: metrics[k] for k in ("suffix_automaton", "neural", "hybrid")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
