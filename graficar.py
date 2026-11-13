"""
Dibuja la curva de aprendizaje a partir de curva.csv (y opcionalmente log.csv).

Uso:
    python graficar.py runs/desde_cero
    python graficar.py runs/desde_cero runs/con_imitacion      # compara corridas
    python graficar.py runs/desde_cero runs/desde_cero/curva_d2.csv   # misma red, otra profundidad

Guarda curva.png en la carpeta de la primera corrida.
"""
import csv
import os
import sys

import matplotlib.pyplot as plt


def csv_of(arg):
    return arg if arg.endswith(".csv") else os.path.join(arg, "curva.csv")


def label_of(arg):
    if arg.endswith(".csv"):
        return os.path.basename(os.path.dirname(arg)) + " " + os.path.basename(arg)[:-4].replace("curva_", "")
    return os.path.basename(arg.rstrip("/\\"))


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    runs = sys.argv[1:]
    if not runs:
        print(__doc__)
        return
    rivals = []
    for name in read(csv_of(runs[0]))[0].keys():
        if name.startswith("puntaje_"):
            rivals.append(name[len("puntaje_"):])

    fig, axes = plt.subplots(1, len(rivals), figsize=(5 * len(rivals), 4), sharey=True, squeeze=False)
    for ax, rival in zip(axes[0], rivals):
        for run in runs:
            rows = read(csv_of(run))
            x = [int(r["partidas_entrenadas"]) for r in rows if r.get(f"puntaje_{rival}")]
            y = [float(r[f"puntaje_{rival}"]) for r in rows if r.get(f"puntaje_{rival}")]
            ax.plot(x, y, marker="o", label=label_of(run))
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"contra {rival}")
        ax.set_xlabel("partidas de autojuego")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("puntaje del agente (1 = gana todo)")
    if len(runs) > 1:
        axes[0][0].legend()
    fig.suptitle("Curva de aprendizaje")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(csv_of(runs[0])), "curva.png")
    fig.savefig(out, dpi=150)
    print(f"guardado -> {out}")


if __name__ == "__main__":
    main()
