"""Рисование автоматов через Graphviz (нужна утилита `dot`)."""
import subprocess

from automata import EPS

STYLE = ('graph [rankdir=LR, fontname="DejaVu Sans", bgcolor=white, nodesep=0.25, ranksep=0.35];\n'
         'node [shape=circle, fontname="DejaVu Sans", fontsize=11, width=0.35, fixedsize=true, '
         'style=filled, fillcolor="#EEEDFE", color="#534AB7"];\n'
         'edge [fontname="DejaVu Sans", fontsize=13, color="#5F5E5A", arrowsize=0.6];\n'
         '__start [shape=point, width=0.05, style=invis];\n')


def dfa_to_dot(dfa, title=""):
    states = dfa.states()
    num = {s.id: i for i, s in enumerate(sorted(states, key=lambda s: s.id))}
    lines = ["digraph A {", STYLE, f'label="{title}"; labelloc=t; fontsize=16;']
    for s in states:
        extra = ', shape=doublecircle, fillcolor="#E1F5EE", color="#0F6E56"' if s.final else ""
        lines.append(f'{num[s.id]} [label="{num[s.id]}"{extra}];')
    lines.append(f"__start -> {num[dfa.start.id]};")
    for s in states:
        groups = {}
        for c, t in s.edges.items():
            groups.setdefault(t.id, []).append(c)
        for tid, cs in groups.items():
            lines.append(f'{num[s.id]} -> {num[tid]} [label="{",".join(sorted(cs))}"];')
    lines.append("}")
    return "\n".join(lines)


def nfa_to_dot(nfa, title=""):
    lines = ["digraph A {", STYLE, f'label="{title}"; labelloc=t; fontsize=16;']
    for q in range(nfa.n):
        extra = ', shape=doublecircle, fillcolor="#E1F5EE", color="#0F6E56"' if q in nfa.finals else ""
        lines.append(f'{q} [label="{q}"{extra}];')
    lines.append(f"__start -> {nfa.start};")
    for (q, c), rs in nfa.edges.items():
        for r in rs:
            if c == EPS:
                lines.append(f'{q} -> {r} [label="ε", style=dashed, color="#D85A30", fontcolor="#D85A30"];')
            else:
                lines.append(f'{q} -> {r} [label="{c}"];')
    lines.append("}")
    return "\n".join(lines)


def render(dot_src, path):
    """path без расширения: сохраняет .dot и .png."""
    with open(path + ".dot", "w", encoding="utf-8") as f:
        f.write(dot_src)
    subprocess.run(["dot", "-Tpng", "-Gdpi=150", path + ".dot", "-o", path + ".png"], check=True)
    return path + ".png"
