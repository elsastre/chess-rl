"""
Grafico resumen de las mediciones para atribuir al RL (CLAUDE.md, seccion 4.4), a partir de los CSV.

  izquierda: enfrentamientos directos (runs/mediciones/duelo_*.csv), diferencia de Elo con IC 95 % por pares.
  derecha:   prediccion del resultado (runs/mediciones/valor.csv), mejora en AUC respecto de la referencia,
             con IC 95 % pareado (las mismas posiciones para todas las redes).

Uso:  python graficar_mediciones.py [--dir runs/mediciones]   -> resumen_atribucion.png
"""
import argparse
import csv
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
AZUL, NAR, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baa7a", "#8a8984"


def color(nombre):
    n = nombre.lower()
    if "sin_rl" in n or "material" in n:
        return GRIS
    if n.startswith("rl_"):
        return AZUL
    if n.startswith("mej"):
        return VERDE
    return NAR


def elo(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return 400 * math.log10(p / (1 - p))


def duelo(path):
    """Devuelve (nombre_a, nombre_b, elo, elo_lo, elo_hi, partidas) con IC por pares."""
    base = os.path.basename(path)[len("duelo_"):-4]
    na, nb = base.split("_vs_", 1)
    pares = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            pares.setdefault(r["par"], []).append(float(r["puntaje_a"]))
    xs = [sum(v) for v in pares.values() if len(v) == 2]
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) / 2
    p = m / 2
    return na, nb, elo(p), elo(p - 1.96 * se), elo(p + 1.96 * se), 2 * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/mediciones")
    args = ap.parse_args()

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": .6, "font.size": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold"})
    duelos = [d for d in (duelo(p) for p in sorted(glob.glob(os.path.join(args.dir, "duelo_*.csv")))) if d]
    valor = os.path.join(args.dir, "valor.csv")
    filas = list(csv.DictReader(open(valor))) if os.path.exists(valor) else []
    if not duelos and not filas:
        print("no hay mediciones en", args.dir); return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    # --- izquierda: enfrentamientos
    for i, (na, nb, e, lo, hi, n) in enumerate(duelos):
        a1.barh(i, e, color=color(na), height=.5)
        a1.errorbar(e, i, xerr=[[e - lo], [hi - e]], color=INK, capsize=5, lw=1.4)
        a1.text(max(hi, 0) + 2, i, f"{e:+.0f} Elo  [{lo:+.0f}; {hi:+.0f}]", va="center", fontsize=9.5)
    a1.axvline(0, color=INK2, lw=1)
    a1.set_yticks(range(len(duelos)))
    a1.set_yticklabels([f"{na}\ncontra {nb}" for na, nb, *_ in duelos])
    a1.invert_yaxis()
    tope = max([hi for *_, hi, _ in duelos] + [10])
    a1.set_xlim(min([lo for *_, lo, _, _ in duelos] + [0]) - 5, tope * 1.6)
    a1.set_xlabel("diferencia de Elo (IC 95 %); 0 = misma fuerza")
    n_part = sorted({n for *_, n in duelos})
    a1.set_title(f"Enfrentamiento directo ({' / '.join(str(n) for n in n_part)} partidas)")
    a1.grid(axis="y", visible=False)

    # --- derecha: AUC respecto de la referencia
    if filas:
        ref = next(f["modelo"] for f in filas if abs(float(f["auc_vs_ref"])) < 1e-12 and f["modelo"] != "material (sin red)")
        puntos = [f for f in filas if f["modelo"] != ref]
        for i, f in enumerate(puntos):
            v, lo, hi = float(f["auc_vs_ref"]), float(f["auc_vs_ref_lo"]), float(f["auc_vs_ref_hi"])
            a2.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="o", color=color(f["modelo"]), ms=8, capsize=5, lw=1.6)
            a2.text(i + .12, v, f"{v:+.3f}", va="center", fontsize=9)
        a2.axhline(0, color=INK2, lw=1)
        a2.set_xticks(range(len(puntos)))
        a2.set_xticklabels([f["modelo"].replace(" (sin red)", "\n(sin red)") for f in puntos])
        a2.set_xlim(-.4, len(puntos) - .4)
        auc_ref = next(float(f["auc"]) for f in filas if f["modelo"] == ref)
        a2.set_ylabel(f"mejora en AUC respecto de {ref} (IC 95 %)")
        a2.set_title(f"Predicción de quién gana (AUC de {ref} = {auc_ref:.3f})")
        a2.grid(axis="x", visible=False)
    else:
        a2.set_visible(False)
    fig.text(.5, .005, "Gris: sin RL · azul: tras el RL (0 a 150.000) · naranja: continuación (150.000 a 300.000) · "
             "verde: corrida con mejoras. IC 95 %: por pares de aperturas (izq.) y bootstrap pareado por partida (der.).",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, .04, 1, 1))
    out = os.path.join(args.dir, "resumen_atribucion.png")
    fig.savefig(out, dpi=160)
    print("guardado ->", out)


if __name__ == "__main__":
    main()
