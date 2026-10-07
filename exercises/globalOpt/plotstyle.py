"""Shared matplotlib style for the TutorialGO figures (static PDFs for answers.tex)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots, fixed order (dataviz reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
SEQ = "Blues"   # sequential: one hue, light -> dark

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "lines.linewidth": 2,
    "legend.frameon": False,
})


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figs/{name}")
