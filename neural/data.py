"""
Обучающие данные для нейросети из словаря OpenCorpora (он поставляется
вместе с pymorphy3 и сам хранится в виде DAWG-автомата).

Каждый пример:   словоформа  ->  класс
класс = (грамматический тег, правило получения леммы)
правило леммы  = (сколько букв отрезать с конца, что дописать)

    кружками  ->  ('NOUN,inan,femn plur,ablt', 'ками' -> 'ка')
    отрезать 'ками', дописать 'ка'  =>  лемма 'кружка'

Разбиение train/test делается ПО ЛЕММАМ: все формы тестовых слов
нейросеть ни разу не видела. Так мы честно имитируем незнакомые слова.
"""
import os
import pickle
import random
import re
from collections import Counter, defaultdict

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "dataset.pkl")
CYR = re.compile(r"^[а-я]+$")
SKIP_GRAMMEMES = {"Abbr", "Init", "Erro", "Dist", "Infr", "Slng", "Arch"}


def lemma_rule(word, lemma):
    """Самый длинный общий префикс -> (что отрезать, что дописать)."""
    p = 0
    while p < min(len(word), len(lemma)) and word[p] == lemma[p]:
        p += 1
    return word[p:], lemma[p:]


def apply_rule(word, cut, add):
    if cut and not word.endswith(cut):
        return None
    return word[: len(word) - len(cut)] + add


def build(test_share=0.04, max_train_lemmas=45000, seed=42, min_class_freq=3):
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()

    # 1. Все пары (слово, класс) по леммам
    by_lemma = defaultdict(list)       # (лемма, парадигма) -> [(слово, тег, cut, add)]
    for word, tag, lemma, para_id, _ in morph.dictionary.iter_known_words():
        word, lemma = word.replace("ё", "е"), lemma.replace("ё", "е")
        if not CYR.match(word) or not CYR.match(lemma) or len(word) < 2:
            continue
        if tag.grammemes & SKIP_GRAMMEMES:
            continue
        cut, add = lemma_rule(word, lemma)
        by_lemma[(lemma, para_id)].append((word, str(tag), cut, add))

    # 2. Разбиение по леммам
    keys = sorted(by_lemma)
    random.Random(seed).shuffle(keys)
    n_test = int(len(keys) * test_share)
    test_keys = keys[:n_test]
    train_keys = keys[n_test:n_test + max_train_lemmas]

    def collect(ks):
        d = defaultdict(set)            # слово -> множество правильных (tag, cut, add, lemma)
        for k in ks:
            for w, tag, cut, add in by_lemma[k]:
                d[w].add((tag, cut, add, k[0]))
        return d

    train, test = collect(train_keys), collect(test_keys)
    # честный тест: только слова, которых нет среди обучающих словоформ
    test = {w: v for w, v in test.items() if w not in train}

    # 3. Словарь классов (редкие отбрасываем)
    freq = Counter((t, c, a) for v in train.values() for (t, c, a, _) in v)
    classes = sorted(k for k, n in freq.items() if n >= min_class_freq)
    cls_id = {c: i for i, c in enumerate(classes)}

    def to_examples(d, keep_unknown):
        out = []
        for w, parses in d.items():
            ids = sorted({cls_id[(t, c, a)] for (t, c, a, _) in parses if (t, c, a) in cls_id})
            if ids or keep_unknown:
                out.append((w, ids, sorted(parses)))
        return out

    chars = sorted({ch for w in train for ch in w})
    ds = {
        "train": to_examples(train, keep_unknown=False),
        "test": to_examples(test, keep_unknown=True),
        "classes": classes,
        "chars": chars,
        "stats": {
            "lemmas_total": len(keys),
            "train_lemmas": len(train_keys),
            "test_lemmas": len(test_keys),
        },
    }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(ds, f)
    return ds


def load():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return build()


if __name__ == "__main__":
    import time
    t = time.time()
    ds = build()
    print(f"готово за {time.time() - t:.0f} c")
    print(ds["stats"])
    print("train слов:", len(ds["train"]), " test слов:", len(ds["test"]),
          " классов:", len(ds["classes"]), " букв:", len(ds["chars"]))
    amb = sum(len(ids) > 1 for _, ids, _ in ds["train"])
    print("омонимичных слов в train:", amb)
    for w, ids, p in ds["train"][:3]:
        print(w, [ds["classes"][i] for i in ids])
