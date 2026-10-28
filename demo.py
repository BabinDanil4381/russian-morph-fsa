"""
Демонстрация к докладу «Конечные автоматы и морфологический анализ».

    python demo.py            # всё подряд
    python demo.py --step     # с паузой (Enter) между шагами — для доклада
    python demo.py слово ...  # разобрать свои слова
"""
import os
import sys

from automata import build_trie, build_dawg, determinize, minimize_brzozowski
from lexicon import LEMMAS, PARADIGMS
from morph import MorphAnalyzer, build_stem_ending_nfa, surface_forms
from viz import dfa_to_dot, nfa_to_dot, render

STEP = "--step" in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pictures")
os.makedirs(OUT, exist_ok=True)


def header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)
    if STEP:
        input("  [Enter] ")


def fmt(parses):
    return "\n".join(f"    {lemma:<8} {tags:<22} ({src})" for lemma, tags, src in parses) or "    — нет разбора"


def main():
    words = [a for a in sys.argv[1:] if not a.startswith("--")]
    morph = MorphAnalyzer()
    if words:
        for w in words:
            print(f"{w}:\n{fmt(morph.parse(w))}")
        return

    forms = surface_forms()

    # ------------------------------------------------------------ шаг 1
    header("Шаг 1. Словарь: леммы + парадигмы -> словоформы")
    print(f"  лемм: {len(LEMMAS)}, парадигм: {len(PARADIGMS)}, "
          f"разных словоформ: {len(forms)}")
    print("  пример — парадигма 'кошка':",
          ", ".join(w for w, l, *_ in morph.rows if l == "кошка"))

    # ------------------------------------------------------------ шаг 2
    header("Шаг 2. ДКА-распознаватель: trie против минимального автомата (DAWG)")
    frag = [w for w, l, *_ in morph.rows if l in ("кошка", "мышка")]
    trie, dawg = build_trie(frag), build_dawg(frag)
    print(f"  фрагмент кошка+мышка: trie {trie.n_states()} сост., "
          f"DAWG {dawg.n_states()} сост.")
    print("  картинки:", render(dfa_to_dot(trie, "Trie: кошка + мышка"), f"{OUT}/1_trie"))
    print("           ", render(dfa_to_dot(dawg, "Минимальный ДКА: общие окончания склеены"), f"{OUT}/2_dawg"))
    trie_all, dawg_all = build_trie(forms), build_dawg(forms)
    print(f"  весь словарь: trie {trie_all.n_states()} сост. / {trie_all.n_edges()} переходов; "
          f"DAWG {dawg_all.n_states()} сост. / {dawg_all.n_edges()} переходов")
    for w in ["кошками", "мышек", "кошкой", "кошкаа", "кош"]:
        print(f"    accepts({w!r}) = {dawg_all.accepts(w)}")

    # ------------------------------------------------------------ шаг 3
    header("Шаг 3. НКА 'основа · окончание' -> детерминизация -> минимизация")
    frag_lemmas = [l for l in LEMMAS if l[0] in ("кошк", "мышк")]
    nfa = build_stem_ending_nfa(frag_lemmas)
    dfa, n_subsets = determinize(nfa)
    mdfa = minimize_brzozowski(dfa)
    print(f"  НКА: {nfa.n} сост., {nfa.n_edges()} переходов (с ε)")
    print(f"  после построения подмножеств: {n_subsets} сост.")
    print(f"  после минимизации (Бжозовский): {mdfa.n_states()} сост. "
          f"(= DAWG: {dawg.n_states()})")
    assert set(mdfa.language()) == set(frag), "языки должны совпадать"
    print("  язык НКА совпадает со словарём фрагмента: OK")
    print("  картинки:", render(nfa_to_dot(nfa, "НКА: основы --ε--> общая парадигма окончаний"), f"{OUT}/3_nfa"))
    print("           ", render(dfa_to_dot(dfa, "ДКА после построения подмножеств"), f"{OUT}/4_dfa"))

    full_nfa = build_stem_ending_nfa()
    full_dfa, _ = determinize(full_nfa)
    assert set(full_dfa.language()) == set(forms)
    print(f"  весь словарь: НКА {full_nfa.n} сост. -> ДКА {full_dfa.n_states()} "
          f"-> минимальный {minimize_brzozowski(full_dfa).n_states()}")

    # ------------------------------------------------------------ шаг 4
    header("Шаг 4. Анализатор: словарь, скомпилированный в автомат (как pymorphy2)")
    print(f"  ключи вида 'словоформа|парадигма:форма', DAWG: "
          f"{morph.dawg.n_states()} сост.")
    for w in ["кошками", "кошки", "кошке", "мышек", "книги", "лампы"]:
        print(f"  {w}:\n{fmt(morph.parse(w))}")

    # ------------------------------------------------------------ шаг 5
    header("Шаг 5. Омонимия: одна словоформа — несколько разборов")
    for w in ["стекло", "стекла", "мыла", "мыл"]:
        print(f"  {w}:\n{fmt(morph.parse(w))}")

    # ------------------------------------------------------------ шаг 6
    header("Шаг 6. Незнакомые слова: предсказание по концу слова")
    for w in ["кружками", "фишкой", "плюшки", "зумерка"]:
        print(f"  {w}:\n{fmt(morph.parse(w))}")

    # ------------------------------------------------------------ шаг 7
    header("Шаг 7. Сравнение с настоящим анализатором (pymorphy3)")
    try:
        import pymorphy3
        m = pymorphy3.MorphAnalyzer()
        for w in ["кошки", "стекло", "мыла", "кружками"]:
            print(f"  {w}:")
            for p in m.parse(w)[:4]:
                print(f"    {p.normal_form:<8} {str(p.tag):<28} score={p.score:.3f}")
    except ImportError:
        print("  pymorphy3 не установлен:  pip install pymorphy3")

    # ------------------------------------------------------------ шаг 8
    header("Шаг 8. Гибрид: словарь-DAWG + автомат окончаний + нейросеть")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural", "cache", "model.pt")
    if not os.path.exists(model_path):
        print("  модель не обучена:  cd neural && python data.py && python train.py")
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural"))
    from analyzer import HybridAnalyzer, ru
    an = HybridAnalyzer()
    for w in ["кошками", "кружками", "зумеров", "кринжовый", "загуглила", "хайпанули"]:
        print(f"  {w}:")
        for lemma, tag, p, src in an.parse(w, explain=(w == "загуглила")):
            print(f"    {lemma:<12} {ru(tag):<34} p={p:<6} [{src}]")


if __name__ == "__main__":
    main()
