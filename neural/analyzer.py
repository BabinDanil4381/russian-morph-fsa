"""
Гибридный морфологический анализатор: словарь-автомат + автомат окончаний + нейросеть.

    слово ──► 1) DAWG словаря OpenCorpora (pymorphy3) ── нашли ──► точные разборы
                 │ не нашли
                 ▼
              2) автомат окончаний: какие классы ВОЗМОЖНЫ для такого конца слова
                 ▼
              3) нейросеть CharBiLSTM: какой из возможных классов ВЕРОЯТНЕЕ
                 ▼
              разбор + лемма + вероятность

    python analyzer.py зумеров кринжовый загуглила хайпанули
    python analyzer.py --explain кружками
"""
import os
import pickle
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import apply_rule  # noqa: E402
from model import CharBiLSTM, Encoder  # noqa: E402
from automata_bridge import EndingAutomaton  # noqa: E402

# Человекочитаемые названия граммем OpenCorpora
RU = {
    "NOUN": "сущ", "ADJF": "прил", "ADJS": "кр.прил", "COMP": "сравн", "VERB": "гл",
    "INFN": "инф", "PRTF": "прич", "PRTS": "кр.прич", "GRND": "деепр", "ADVB": "нареч",
    "NUMR": "числ", "NPRO": "мест", "PREP": "предл", "CONJ": "союз", "PRCL": "част",
    "masc": "м.р.", "femn": "ж.р.", "neut": "ср.р.", "ms-f": "общ.р.",
    "sing": "ед", "plur": "мн",
    "nomn": "им", "gent": "род", "datv": "дат", "accs": "вин", "ablt": "тв", "loct": "пр",
    "voct": "зв", "gen2": "род2", "loc2": "пр2", "acc2": "вин2",
    "anim": "одуш", "inan": "неодуш", "perf": "сов", "impf": "несов",
    "tran": "перех", "intr": "неперех", "past": "прош", "pres": "наст", "futr": "буд",
    "1per": "1л", "2per": "2л", "3per": "3л", "indc": "изъяв", "impr": "повел",
    "actv": "действ", "pssv": "страд", "Qual": "кач", "Name": "имя", "Surn": "фам",
    "Patr": "отч", "Geox": "гео", "Orgn": "орг", "Fixd": "неизм",
}


def ru(tag):
    parts = tag.replace(" ", ",").split(",")
    return " ".join(RU.get(p, p) for p in parts if p)


class HybridAnalyzer:
    def __init__(self, use_dictionary=True):
        ck = torch.load(os.path.join(HERE, "cache", "model.pt"), map_location="cpu")
        self.classes, self.enc = ck["classes"], Encoder(ck["chars"])
        self.model = CharBiLSTM(len(ck["chars"]), len(self.classes))
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        with open(os.path.join(HERE, "cache", "sa.pkl"), "rb") as f:
            self.sa = pickle.load(f)
        self.ea = EndingAutomaton(self.classes)
        self.morph = None
        if use_dictionary:
            import pymorphy3
            self.morph = pymorphy3.MorphAnalyzer()

    # ------------------------------------------------------------ шаг 1
    def from_dictionary(self, word):
        if self.morph is None or not self.morph.word_is_known(word):
            return []
        out = []
        for p in self.morph.parse(word):
            if type(p.methods_stack[0][0]).__name__ == "DictionaryAnalyzer":
                out.append((p.normal_form, str(p.tag), round(p.score, 3), "словарь (DAWG)"))
        return out

    # -------------------------------------------------------- шаги 2 и 3
    @torch.no_grad()
    def predict(self, word, k=3, explain=False, weight=0.25):
        """Лучший вариант из compare_hybrids.py (E):
        1) автомат возможных окончаний оставляет только классы, чьё
           отрезаемое окончание действительно стоит в конце слова;
        2) оценка = log p_нейросети + weight * log p_автомата_окончаний."""
        cands, suffix = self.sa.candidates(word)
        x, L = self.enc.batch([word])
        logp = torch.log_softmax(self.model(x, L)[0], 0)
        ids = sorted(self.ea.possible(word)) or list(range(len(self.classes)))
        t = torch.tensor(ids)
        total = sum(cands.values())
        prior = torch.tensor([(cands.get(i, 0) + 0.1) / (total + 0.1 * len(ids)) for i in ids])
        probs = torch.softmax(logp[t] + weight * torch.log(prior), 0)
        order = probs.argsort(descending=True)
        if explain:
            print(f"  автомат возможных окончаний: {len(ids)} классов из {len(self.classes)}")
            print(f"  автомат окончаний (статистика): окончание '-{suffix}', "
                  f"встречалось {len(cands)} вариантов, чаще всего: "
                  f"{ru(self.classes[cands.most_common(1)[0][0]][0]) if cands else '—'}")
            print(f"  нейросеть одна выбрала бы: {ru(self.classes[int(logp.argmax())][0])}"
                  f"  (p={float(logp.max().exp()):.2f})")
        out, seen = [], set()
        for j in order.tolist():
            tag, cut, add = self.classes[ids[j]]
            lemma = apply_rule(word, cut, add)
            if lemma is None or (lemma, tag) in seen:
                continue
            seen.add((lemma, tag))
            out.append((lemma, tag, round(float(probs[j]), 3), "автомат окончаний + нейросеть"))
            if len(out) == k:
                break
        return out

    def parse(self, word, k=3, explain=False):
        word = word.lower().replace("ё", "е")
        return self.from_dictionary(word) or self.predict(word, k, explain)


def main():
    explain = "--explain" in sys.argv
    words = [w for w in sys.argv[1:] if not w.startswith("--")] or [
        "кошками", "зумеров", "кринжовый", "загуглила", "хайпанули", "вайбовые", "лайкнуть"]
    an = HybridAnalyzer()
    for w in words:
        print(f"\n{w}:")
        for lemma, tag, p, src in an.parse(w, explain=explain):
            print(f"    {lemma:<14} {ru(tag):<34} p={p:<6} [{src}]")


if __name__ == "__main__":
    main()
