"""
Автомат окончаний — та часть системы, которая говорит, ЧТО ВОЗМОЖНО.

Это ДКА (префиксное дерево), построенный по ПЕРЕВЁРНУТЫМ окончаниям
слов словаря. Слово `кружками` читается с конца: и, м, а, к, ...
В каждом состоянии хранится статистика: какие классы (тег + правило
леммы) встречались у словарных слов с таким окончанием.

Так же устроен "предсказатель" в pymorphy2 (там это DAWG окончаний).

Использование:
  * сам по себе — базовый метод: самый частый класс для окончания;
  * в гибриде — фильтр: нейросеть выбирает только среди классов,
    которые автомат считает возможными.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from automata import State  # noqa: E402  (наш класс состояния ДКА)


class SuffixAutomaton:
    def __init__(self, examples, classes, max_suffix=5):
        self.classes = classes
        self.max_suffix = max_suffix
        self.root = State()
        self.root.stats = Counter()
        for word, ids, _ in examples:
            s = self.root
            s.stats.update(ids)
            for ch in reversed(word[-max_suffix:]):
                if ch not in s.edges:
                    t = State()
                    t.stats = Counter()
                    s.edges[ch] = t
                s = s.edges[ch]
                s.stats.update(ids)

    def n_states(self):
        n, stack = 0, [self.root]
        while stack:
            s = stack.pop()
            n += 1
            stack.extend(s.edges.values())
        return n

    def _path(self, word):
        """Все состояния, через которые проходит перевёрнутое слово."""
        path, s = [self.root], self.root
        for ch in reversed(word[-self.max_suffix:]):
            s = s.edges.get(ch)
            if s is None:
                break
            path.append(s)
        return path

    def candidates(self, word):
        """Возможные классы: самое длинное известное окончание, у которого
        есть хотя бы один класс, применимый к слову (отрезаемая часть
        действительно стоит в конце слова)."""
        for depth, s in reversed(list(enumerate(self._path(word)))):
            cands = Counter({c: n for c, n in s.stats.items()
                             if word.endswith(self.classes[c][1])})
            if cands:
                return cands, word[len(word) - depth:] if depth else ""
        return Counter(), ""

    def predict(self, word):
        cands, _ = self.candidates(word)
        return cands.most_common(1)[0][0] if cands else None
