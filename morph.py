"""
Морфологический анализатор = словарь, скомпилированный в минимальный автомат.

Как в pymorphy2: в автомат кладутся строки
        "словоформа" + SEP + "номер_парадигмы" + ":" + "номер_формы"
Анализ слова: пройти по автомату буквами слова, затем по SEP и
перечислить все продолжения — это и есть все разборы (омонимия).
Номер парадигмы + номер формы -> теги и лемма.
"""
from collections import Counter, defaultdict

from automata import NFA, EPS, build_dawg, determinize
from lexicon import PARADIGMS, PARADIGM_NAMES, LEMMAS, VERB_FIXES, generate, form_ending

SEP = "|"


class MorphAnalyzer:
    def __init__(self):
        self.rows = generate()
        keys = [f"{w}{SEP}{pid}:{fi}" for w, _, pid, fi, _ in self.rows]
        self.dawg = build_dawg(keys)
        # лемма восстанавливается по (основа o1, парадигма); храним как
        # отдельный маленький словарь — как "словарь лемм" в pymorphy2
        self.lemma_of = {(w, pid, fi): lemma for w, lemma, pid, fi, _ in self.rows}
        self._build_suffix_index()

    # ---------------------------------------------------------------- анализ
    def lookup(self, word):
        """Возвращает список (pid, fidx) для словарного слова."""
        s = self.dawg.start
        for ch in word + SEP:
            s = s.edges.get(ch)
            if s is None:
                return []
        out = []

        def dfs(st, acc):
            if st.final:
                pid, fi = acc.split(":")
                out.append((int(pid), int(fi)))
            for c, t in st.edges.items():
                dfs(t, acc + c)
        dfs(s, "")
        return sorted(out)

    def parse(self, word):
        word = word.lower().replace("ё", "е")
        res = []
        for pid, fi in self.lookup(word):
            tags = PARADIGMS[PARADIGM_NAMES[pid]][fi][2]
            res.append((self.lemma_of[(word, pid, fi)], tags, "словарь"))
        return res or self.predict(word)

    # ---------------------------------------------------- предсказание (OOV)
    def _build_suffix_index(self, max_len=4):
        """Окончание слова (до max_len букв) -> какие (парадигма, форма) там встречались."""
        self.suffix_index = defaultdict(Counter)
        for w, _, pid, fi, _ in self.rows:
            for k in range(1, min(max_len, len(w)) + 1):
                self.suffix_index[w[-k:]][(pid, fi)] += 1

    def predict(self, word, max_len=4):
        """Разбор незнакомого слова по самому длинному известному концу."""
        for k in range(min(max_len, len(word) - 1), 0, -1):
            cands = self.suffix_index.get(word[-k:])
            if not cands:
                continue
            res = []
            for (pid, fi), _ in cands.most_common():
                stem_key, ending = form_ending(pid, fi)
                if stem_key != "o1" or not word.endswith(ending):
                    continue      # форму с чередованием в основе не угадать
                stem = word[: len(word) - len(ending)] if ending else word
                lemma_key, lemma_end = form_ending(pid, 0)
                if lemma_key != "o1":
                    continue
                lemma = stem + lemma_end
                tags = PARADIGMS[PARADIGM_NAMES[pid]][fi][2]
                item = (lemma, tags, f"предсказано по '-{word[-k:]}'")
                if item not in res:
                    res.append(item)
            if res:
                return res
        return []


# ------------------------------------------------------------------------
# НКА "основа · окончания": так автомат удобно ОПИСЫВАТЬ (по словарю),
# а затем детерминизировать для быстрого ИСПОЛНЕНИЯ.
# ------------------------------------------------------------------------
def build_stem_ending_nfa(lemmas=LEMMAS):
    nfa = NFA()
    # 1) по одному подавтомату окончаний на парадигму
    entry = {}
    used = {p for _, _, p in lemmas}
    for pname, forms in PARADIGMS.items():
        if pname not in used:
            continue
        q_in = nfa.new()
        q_fin = nfa.new()
        nfa.finals.add(q_fin)
        for stem_key, ending, _ in forms:
            if stem_key == "o1":
                nfa.add_word(q_in, ending, q_fin) if ending else nfa.add(q_in, EPS, q_fin)
        entry[pname] = q_in
    # 2) основы: путь из старта, затем ε-переход во вход парадигмы
    for o1, o2, pname in lemmas:
        if o1 in VERB_FIXES:          # нерегулярный глагол — отдельными словами
            q = nfa.add_word(nfa.start, o1)
            nfa.finals.add(q)
            for e in VERB_FIXES[o1].values():
                nfa.finals.add(nfa.add_word(q, e))
        else:
            q = nfa.add_word(nfa.start, o1)
            nfa.add(q, EPS, entry[pname])
        # формы с чередованием в основе (кошек) — целым словом
        if any(k == "o2" for k, _, _ in PARADIGMS[pname]):
            nfa.finals.add(nfa.add_word(nfa.start, o2))
    return nfa


def surface_forms():
    return sorted({w for w, *_ in generate()})


if __name__ == "__main__":
    nfa = build_stem_ending_nfa()
    dfa, _ = determinize(nfa)
    assert set(dfa.language()) == set(surface_forms())
    print("НКА и словарь задают один язык: OK")
