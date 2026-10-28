"""
Автомат "возможных окончаний".

Каждый класс нейросети содержит правило леммы "отрезать с конца X".
Класс грамматически возможен для слова, только если слово кончается на X.
Строим ДКА (префиксное дерево) по ПЕРЕВЁРНУТЫМ окончаниям X всех классов.
Читая перевёрнутое слово, автомат в каждом пройденном состоянии
собирает классы, чьё окончание целиком прочитано, — это и есть все
возможные классы. Один проход по слову, O(длина слова).
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from automata import State  # noqa: E402


class EndingAutomaton:
    def __init__(self, classes):
        self.root = State()
        self.out = defaultdict(list)          # id состояния -> классы с таким окончанием
        for cid, (_, cut, _) in enumerate(classes):
            s = self.root
            for ch in reversed(cut):
                s = s.edges.setdefault(ch, State())
            s.final = True
            self.out[s.id].append(cid)

    def possible(self, word):
        s, res = self.root, list(self.out[self.root.id])
        for ch in reversed(word):
            s = s.edges.get(ch)
            if s is None:
                break
            res.extend(self.out[s.id])
        return res
