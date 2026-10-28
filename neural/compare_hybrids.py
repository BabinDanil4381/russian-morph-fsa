"""
Как лучше соединить автомат и нейросеть? Сравниваем варианты на тех же
20 000 незнакомых словах, что и train.py.

  A. Только автомат окончаний (самый частый класс для окончания)
  B. Только нейросеть
  C. Жёсткий фильтр: нейросеть выбирает ТОЛЬКО из классов, которые автомат
     видел с этим окончанием (может отсечь правильный ответ!)
  D. Фильтр "грамматически возможно": автомат окончаний классов проверяет,
     что отрезаемое окончание действительно стоит в конце слова.
     Правильный ответ так отсечь нельзя.
  E. Байесовское объединение: p(класс) ~ p_нейросети * p_автомата^w
     (статистика автомата — как "мнение второго эксперта")

    python compare_hybrids.py
"""
import json
import os
import pickle
import random

import torch

from automata_bridge import EndingAutomaton
from data import load
from model import CharBiLSTM, Encoder
from train import evaluate, nn_logits

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cache")
    ds = load()
    test = random.Random(1).sample(ds["test"], min(20000, len(ds["test"])))
    ck = torch.load(os.path.join(cache, "model.pt"), map_location="cpu")
    classes = ck["classes"]
    model = CharBiLSTM(len(ck["chars"]), len(classes))
    model.load_state_dict(ck["state"])
    with open(os.path.join(cache, "sa.pkl"), "rb") as f:
        sa = pickle.load(f)
    ea = EndingAutomaton(classes)

    words = [w for w, _, _ in test]
    logp = torch.log_softmax(nn_logits(model, Encoder(ck["chars"]), words), 1)
    cands = [sa.candidates(w)[0] for w in words]

    # насколько часто правильный ответ вообще есть среди кандидатов
    gold_sets = [set(ids) for _, ids, _ in test]
    recall_sa = sum(bool(g & set(c)) for g, c in zip(gold_sets, cands)) / len(test)
    possible = [ea.possible(w) for w in words]
    recall_ea = sum(bool(g & set(p)) for g, p in zip(gold_sets, possible)) / len(test)

    res = {}
    res["A. Автомат окончаний"] = evaluate(
        [c.most_common(1)[0][0] if c else None for c in cands], test, classes)
    res["B. Нейросеть"] = evaluate(logp.argmax(1).tolist(), test, classes)

    def masked(ids_list):
        out = []
        for row, ids in zip(logp, ids_list):
            if ids:
                t = torch.tensor(sorted(ids))
                out.append(int(t[row[t].argmax()]))
            else:
                out.append(int(row.argmax()))
        return out

    res["C. Жёсткий фильтр"] = evaluate(masked([list(c) for c in cands]), test, classes)
    res["D. Фильтр возможных окончаний"] = evaluate(masked(possible), test, classes)

    best = None
    for w8 in (0.25, 0.5, 1.0, 2.0):
        preds = []
        for row, c, p in zip(logp, cands, possible):
            t = torch.tensor(sorted(p)) if p else torch.arange(len(classes))
            score = row[t].clone()
            total = sum(c.values())
            prior = torch.tensor([(c.get(int(k), 0) + 0.1) / (total + 0.1 * len(t)) for k in t])
            score += w8 * torch.log(prior)
            preds.append(int(t[score.argmax()]))
        r = evaluate(preds, test, classes)
        print(f"  вес автомата {w8}: {r}")
        if best is None or r["full"] > best[1]["full"]:
            best = (w8, r)
    res[f"E. Нейросеть × статистика автомата (вес {best[0]})"] = best[1]

    for k, v in res.items():
        print(f"{k:<48} {v}")
    print(f"правильный ответ среди кандидатов: жёсткий фильтр {100 * recall_sa:.1f}%, "
          f"фильтр окончаний {100 * recall_ea:.1f}%")
    print(f"среднее число вариантов: жёсткий {sum(map(len, cands)) / len(cands):.1f}, "
          f"фильтр окончаний {sum(map(len, possible)) / len(possible):.0f} из {len(classes)}")

    m = json.load(open(os.path.join(cache, "metrics.json"), encoding="utf-8"))
    m["variants"] = res
    m["recall_strict"] = round(100 * recall_sa, 1)
    m["recall_endings"] = round(100 * recall_ea, 1)
    m["avg_possible"] = round(sum(map(len, possible)) / len(possible))
    m["best_weight"] = best[0]
    json.dump(m, open(os.path.join(cache, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
