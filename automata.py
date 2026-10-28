"""
Конечные автоматы для морфологии:
  * Trie            — префиксное дерево (ДКА в форме дерева)
  * build_dawg      — минимальный ациклический ДКА (алгоритм Дачука и др., 2000)
  * NFA             — недетерминированный автомат с ε-переходами
  * determinize     — построение подмножеств (Рабин–Скотт)
  * minimize_brzozowski — минимизация: reverse -> determinize -> reverse -> determinize
"""
from itertools import count

EPS = ""  # ε-переход


# --------------------------------------------------------------------------
# ДКА: состояние = словарь переходов + флаг "допускающее"
# --------------------------------------------------------------------------
class State:
    _ids = count()

    def __init__(self):
        self.id = next(State._ids)
        self.edges = {}       # символ -> State
        self.final = False

    def signature(self):
        """Два состояния эквивалентны, если у них одинаковое 'будущее'."""
        return (self.final, tuple(sorted((c, s.id) for c, s in self.edges.items())))


class DFA:
    def __init__(self, start):
        self.start = start

    def accepts(self, word):
        s = self.start
        for ch in word:
            s = s.edges.get(ch)
            if s is None:
                return False
        return s.final

    def states(self):
        seen, stack = {self.start.id: self.start}, [self.start]
        while stack:
            for t in stack.pop().edges.values():
                if t.id not in seen:
                    seen[t.id] = t
                    stack.append(t)
        return list(seen.values())

    def n_states(self):
        return len(self.states())

    def n_edges(self):
        return sum(len(s.edges) for s in self.states())

    def language(self):
        """Все слова языка (автомат ациклический)."""
        out = []

        def dfs(s, prefix):
            if s.final:
                out.append(prefix)
            for c, t in sorted(s.edges.items()):
                dfs(t, prefix + c)
        dfs(self.start, "")
        return out


# --------------------------------------------------------------------------
# 1. Trie
# --------------------------------------------------------------------------
def build_trie(words):
    root = State()
    for w in words:
        s = root
        for ch in w:
            s = s.edges.setdefault(ch, State())
        s.final = True
    return DFA(root)


# --------------------------------------------------------------------------
# 2. Минимальный ДКА за один проход по отсортированным словам
#    (Daciuk, Mihov, Watson, Watson, 2000). Идея: как только ветка trie
#    больше не может измениться, её состояния заменяются на уже
#    известные эквивалентные ("регистр").
# --------------------------------------------------------------------------
def build_dawg(words):
    words = sorted(set(words))
    root = State()
    register = {}                      # сигнатура -> State

    def replace_or_register(state):
        last_char = max(state.edges)   # последний добавленный ребёнок
        child = state.edges[last_char]
        if child.edges:
            replace_or_register(child)
        sig = child.signature()
        if sig in register:
            state.edges[last_char] = register[sig]
        else:
            register[sig] = child

    for w in words:
        # общий префикс с тем, что уже в автомате
        s, i = root, 0
        while i < len(w) and w[i] in s.edges:
            s = s.edges[w[i]]
            i += 1
        if s.edges:
            replace_or_register(s)
        for ch in w[i:]:
            t = State()
            s.edges[ch] = t
            s = t
        s.final = True
    if root.edges:
        replace_or_register(root)
    return DFA(root)


# --------------------------------------------------------------------------
# 3. НКА с ε-переходами
# --------------------------------------------------------------------------
class NFA:
    def __init__(self):
        self.n = 0
        self.edges = {}      # (q, символ) -> set(q')
        self.start = self.new()
        self.finals = set()

    def new(self):
        q = self.n
        self.n += 1
        return q

    def add(self, q, ch, r):
        self.edges.setdefault((q, ch), set()).add(r)

    def add_word(self, q_from, word, q_to=None):
        """Цепочка состояний для слова; возвращает последнее состояние."""
        q = q_from
        for i, ch in enumerate(word):
            r = q_to if (i == len(word) - 1 and q_to is not None) else self.new()
            self.add(q, ch, r)
            q = r
        if not word and q_to is not None:
            self.add(q, EPS, q_to)
            q = q_to
        return q

    def n_edges(self):
        return sum(len(v) for v in self.edges.values())

    def eps_closure(self, qs):
        stack, seen = list(qs), set(qs)
        while stack:
            q = stack.pop()
            for r in self.edges.get((q, EPS), ()):
                if r not in seen:
                    seen.add(r)
                    stack.append(r)
        return frozenset(seen)

    def alphabet(self):
        return {c for (_, c) in self.edges if c != EPS}

    def accepts(self, word):
        cur = self.eps_closure({self.start})
        for ch in word:
            nxt = set()
            for q in cur:
                nxt |= self.edges.get((q, ch), set())
            cur = self.eps_closure(nxt)
        return bool(cur & self.finals)


def determinize(nfa):
    """Построение подмножеств: состояние ДКА = множество состояний НКА."""
    sigma = sorted(nfa.alphabet())
    start_set = nfa.eps_closure({nfa.start})
    dstates = {start_set: State()}
    stack = [start_set]
    while stack:
        S = stack.pop()
        dstates[S].final = bool(S & nfa.finals)
        for ch in sigma:
            T = set()
            for q in S:
                T |= nfa.edges.get((q, ch), set())
            if not T:
                continue
            T = nfa.eps_closure(T)
            if T not in dstates:
                dstates[T] = State()
                stack.append(T)
            dstates[S].edges[ch] = dstates[T]
    return DFA(dstates[start_set]), len(dstates)


def dfa_to_reversed_nfa(dfa):
    """Обращение: все стрелки в обратную сторону, старт <-> финалы."""
    nfa = NFA()
    ids = {}
    for s in dfa.states():
        ids[s.id] = nfa.new()
    for s in dfa.states():
        for ch, t in s.edges.items():
            nfa.add(ids[t.id], ch, ids[s.id])
        if s.final:
            nfa.add(nfa.start, EPS, ids[s.id])
    nfa.finals = {ids[dfa.start.id]}
    return nfa


def minimize_brzozowski(dfa):
    """Теорема Бжозовского: det(rev(det(rev(A)))) — минимальный ДКА."""
    d1, _ = determinize(dfa_to_reversed_nfa(dfa))
    d2, _ = determinize(dfa_to_reversed_nfa(d1))
    return d2
