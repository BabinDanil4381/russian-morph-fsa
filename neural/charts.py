"""Графики для слайдов из cache/metrics.json и cache/size_*/metrics.json
-> ../pictures/5_comparison.png, 6_data_size.png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "pictures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, GREY, GREY_L = "#2a78d6", "#8a8984", "#c9c8c2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def load(path):
    m = json.load(open(path, encoding="utf-8"))
    v = m["variants"]
    best = next(k for k in v if k.startswith("E."))
    return m, {"automaton": v["A. Автомат окончаний"], "neural": v["B. Нейросеть"], "hybrid": v[best]}


main, res = load(os.path.join(HERE, "cache", "metrics.json"))
methods = [("Автомат окончаний", "automaton", GREY),
           ("Нейросеть", "neural", GREY_L),
           ("Автомат + нейросеть", "hybrid", ACCENT)]
metrics = [("Часть речи", "pos"), ("Лемма", "lemma"), ("Полный тег", "tag"), ("Лемма + тег", "full")]

# ---------------------------------------------------------------- 1. сравнение
fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
w, gap = 0.26, 0.03
for i, (name, key, color) in enumerate(methods):
    xs = [j + (i - 1) * (w + gap) for j in range(len(metrics))]
    vals = [res[key][k] for _, k in metrics]
    ax.bar(xs, vals, width=w, color=color, label=name, zorder=2)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=10.5, color=INK,
                fontweight="bold" if key == "hybrid" else "normal")
ax.set_xticks(range(len(metrics)), [n for n, _ in metrics], fontsize=12, color=INK)
ax.set_ylim(0, 108)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
h, a = res["hybrid"]["full"], res["automaton"]["full"]
fig.suptitle(f"На незнакомых словах гибрид прав в {h:.0f}% случаев, автомат окончаний — в {a:.0f}%",
             x=0.06, ha="left", fontsize=15, color=INK, fontweight="bold")
ax.set_title(f"Точность, %. {main['test_words']:,} слов, которых не было в обучении; ".replace(",", " ")
             + f"обучение на {main['train_words']:,} словоформах".replace(",", " "),
             loc="left", fontsize=11.5, color=INK2, pad=26)
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.07), ncol=3, frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "5_comparison.png"), facecolor="white")

# ------------------------------------------------------------ 2. размер данных
runs = []
for n in (10000, 50000):
    p = os.path.join(HERE, "cache", f"size_{n}", "metrics.json")
    if os.path.exists(p):
        runs.append((n, load(p)[1]))
runs.append((main["train_words"], res))
labels = [f"{n // 1000} тыс." for n, _ in runs]
xs = list(range(len(runs)))

fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
DY_END = {"hybrid": 2.0, "neural": -2.0, "automaton": 0}
DY_START = {"hybrid": 2.0, "automaton": -2.0, "neural": 0}
for name, key, color in methods:
    ys = [r[key]["full"] for _, r in runs]
    ax.plot(xs, ys, color=color, lw=2.5 if key == "hybrid" else 2, marker="o",
            ms=8, zorder=3 if key == "hybrid" else 2)
    ax.text(xs[-1] + 0.08, ys[-1] + DY_END[key], f"{name}  {ys[-1]:.1f}%", va="center", fontsize=11,
            color=ACCENT if key == "hybrid" else INK2,
            fontweight="bold" if key == "hybrid" else "normal")
    ax.text(xs[0] - 0.08, ys[0] + DY_START[key], f"{ys[0]:.1f}%", va="center", ha="right", fontsize=11,
            color=ACCENT if key == "hybrid" else INK2)
ax.set_xticks(xs, labels, fontsize=12, color=INK)
ax.set_xlabel("сколько словоформ видела нейросеть при обучении")
ax.set_xlim(-0.45, len(xs) - 1 + 1.15)
ax.set_ylim(30, 85)
ax.grid(axis="y", color=GRID, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
fig.suptitle("Мало данных — выручает автомат, много — нейросеть.\nГибрид всегда рядом с лучшим из двух",
             x=0.06, ha="left", fontsize=15, color=INK, fontweight="bold")
ax.set_title("Доля незнакомых слов с правильными леммой и тегом", loc="left", fontsize=11.5, color=INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "6_data_size.png"), facecolor="white")
print("saved")
