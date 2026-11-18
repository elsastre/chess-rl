"""
Medicion 1: que tan bien predice cada red QUIEN GANA, en partidas humanas que nunca vio (sin jugar).

El objetivo del TD es justamente ese: que el valor de una posicion anticipe el resultado. Si el RL aprendio
algo mas que el material, la red con RL tiene que predecir mejor que la red de material de la que partio.

Para cada posicion (de partidas de Lichess de OTRO mes que el usado para entrenar):
  * se resuelven los cambios de piezas con la quiescencia (como en las hojas de la busqueda),
  * cada red da su valor v en [-1, 1] para el bando que mueve; chance de ganar p = (v + 1) / 2,
  * el resultado real desde ese bando: z = 1 gano, 0 tablas, -1 perdio; y = (z + 1) / 2.

Metricas:
  Brier     promedio de (p - y)^2. Mas bajo es mejor. Mide todo junto: acertar y estar bien calibrado.
  LogLoss   entropia cruzada. Mas bajo es mejor. Castiga mucho estar muy seguro y equivocarse.
  AUC       en partidas con ganador: probabilidad de que una posicion del bando que termino ganando tenga
            mas valor que una del que termino perdiendo. 0,5 = azar, 1 = perfecto. NO depende de la escala
            de v: si sube, la red ordena mejor las posiciones (sabe mas ajedrez), no solo "se reescalo".
Intervalos de confianza al 95 % con bootstrap POR PARTIDA (las posiciones de una misma partida comparten el
resultado; remuestrear posiciones sueltas daria intervalos falsamente angostos).
Las diferencias contra la referencia (--ref) son pareadas: mismas posiciones, mismo remuestreo.

Uso:
    python medir_valor.py --pgn data/lichess_db_standard_rated_2013-02.pgn.zst \\
        --ck material=checkpoints/material_regen.pt rl_150k=runs/final/ck_150000.pt --out runs/mediciones
"""
import argparse
import math
import os
import random
import time

import chess
import chess.pgn
import numpy as np
import torch

from dataset_lichess import open_pgn, RESULT
from encoding import encode
from model import load_checkpoint
from search import qsearch, INF, MATE


# ---------------------------------------------------------------- datos
def leer_posiciones(pgn, max_partidas, por_partida, saltear, min_elo, semilla):
    """Devuelve listas: planos tranquilos, signo (paridad de la secuencia de capturas), material, z, partida."""
    rng = random.Random(semilla)
    X, S, M, Z, G = [], [], [], [], []
    usadas = 0
    with open_pgn(pgn) as f:
        while usadas < max_partidas:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            h = game.headers
            if h.get("Result") not in RESULT:
                continue
            if h.get("Termination", "Normal") != "Normal":          # sin abandonos ni perdidas por tiempo
                continue
            try:
                if min(int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))) < min_elo:
                    continue
            except ValueError:
                continue
            moves = list(game.mainline_moves())
            if len(moves) <= saltear + 2:
                continue
            z_blancas = RESULT[h["Result"]]
            elegidas = set(rng.sample(range(saltear, len(moves)), min(por_partida, len(moves) - saltear)))
            b = game.board()
            for ply, mv in enumerate(moves):
                if ply in elegidas and b.outcome() is None:
                    cp, pv = qsearch(b, -INF, INF, 4)
                    for m in pv:
                        b.push(m)
                    terminal = b.outcome() is not None
                    planos = None if terminal else encode(b)
                    for _ in pv:
                        b.pop()
                    if not terminal:
                        X.append(planos)
                        S.append(1.0 if len(pv) % 2 == 0 else -1.0)
                        mat = 1.0 if cp >= MATE // 2 else -1.0 if cp <= -MATE // 2 else math.tanh(cp * math.log(10) / 800)
                        M.append(mat)
                        Z.append(z_blancas if b.turn == chess.WHITE else -z_blancas)
                        G.append(usadas)
                b.push(mv)
            usadas += 1
            if usadas % 1000 == 0:
                print(f"  {usadas} partidas, {len(X)} posiciones", flush=True)
    return np.stack(X), np.float32(S), np.float32(M), np.float32(Z), np.int64(G), usadas


@torch.no_grad()
def valores(path, X, S, device):
    model, extra = load_checkpoint(path, device)
    out = []
    for i in range(0, len(X), 4096):
        out.append(model(torch.from_numpy(X[i:i + 4096]).float().to(device)).float().cpu().numpy())
    return np.clip(np.concatenate(out) * S, -1, 1), extra.get("games")


# ---------------------------------------------------------------- metricas
def rangos(x):
    """Rangos promedio (con empates), como scipy.stats.rankdata."""
    orden = np.argsort(x, kind="mergesort")
    xs = x[orden]
    r = np.empty(len(x), np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[orden[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def metricas(v, z):
    p = np.clip((v + 1) / 2, 1e-4, 1 - 1e-4)
    y = (z + 1) / 2
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    dec = z != 0
    pos = z[dec] > 0
    r = rangos(v[dec])
    npos, nneg = pos.sum(), (~pos).sum()
    auc = float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
    return {"brier": brier, "logloss": logloss, "auc": auc}


def bootstrap(preds, z, g, n_boot, semilla):
    """preds: {nombre: v}. Devuelve {nombre: {metrica: [muestras]}} remuestreando partidas enteras."""
    rng = np.random.default_rng(semilla)
    partidas = np.unique(g)
    idx_por_partida = {k: np.flatnonzero(g == k) for k in partidas}
    out = {n: {m: [] for m in ("brier", "logloss", "auc")} for n in preds}
    for _ in range(n_boot):
        elegidas = rng.choice(partidas, size=len(partidas), replace=True)
        idx = np.concatenate([idx_por_partida[k] for k in elegidas])
        for n, v in preds.items():
            for m, val in metricas(v[idx], z[idx]).items():
                out[n][m].append(val)
    return out


def ic(xs):
    return float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))


def calibracion(preds, z, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = (z + 1) / 2
    bordes = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--", label="calibración perfecta")
    for n, v in preds.items():
        p = (v + 1) / 2
        xs, ys = [], []
        for a, b in zip(bordes[:-1], bordes[1:]):
            sel = (p >= a) & (p < b if b < 1 else p <= b)
            if sel.sum() >= 30:
                xs.append(p[sel].mean()); ys.append(y[sel].mean())
        ax.plot(xs, ys, marker="o", lw=2, ms=4, label=n)
    ax.set_xlabel("chance de ganar que da la red")
    ax.set_ylabel("proporción real de puntos (victoria 1, tablas 0,5)")
    ax.set_title("Calibración: ¿cuando dice 70 %, gana el 70 %?")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(ruta, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="partidas de un mes NO usado para entrenar (.pgn o .pgn.zst)")
    ap.add_argument("--ck", nargs="+", required=True, help="nombre=ruta.pt (el primero es la referencia)")
    ap.add_argument("--out", default="runs/mediciones")
    ap.add_argument("--partidas", type=int, default=4000)
    ap.add_argument("--por-partida", type=int, default=8)
    ap.add_argument("--saltear", type=int, default=10, help="jugadas de apertura que no se usan")
    ap.add_argument("--min-elo", type=int, default=1200)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    print(f"Leyendo {args.pgn} ...", flush=True)
    X, S, M, Z, G, n = leer_posiciones(args.pgn, args.partidas, args.por_partida, args.saltear, args.min_elo, args.semilla)
    print(f"{len(X)} posiciones de {n} partidas ({time.time() - t0:.0f} s). "
          f"Resultados: {np.mean(Z > 0):.0%} gana el que mueve, {np.mean(Z == 0):.0%} tablas", flush=True)

    preds = {"material (sin red)": M}
    partidas_ent = {"material (sin red)": None}
    for item in args.ck:
        nombre, ruta = item.split("=", 1) if "=" in item else (os.path.basename(item)[:-3], item)
        preds[nombre], partidas_ent[nombre] = valores(ruta, X, S, args.device)
    ref = list(preds)[1] if len(preds) > 1 else list(preds)[0]

    print(f"Bootstrap por partida ({args.boot} remuestreos)...", flush=True)
    bs = bootstrap(preds, Z, G, args.boot, args.semilla)
    filas = []
    for nom, v in preds.items():
        base = metricas(v, Z)
        fila = {"modelo": nom, "partidas_entrenadas": partidas_ent[nom]}
        for m in ("brier", "logloss", "auc"):
            lo, hi = ic(bs[nom][m])
            d = np.array(bs[nom][m]) - np.array(bs[ref][m])
            dlo, dhi = ic(d)
            fila.update({m: base[m], f"{m}_lo": lo, f"{m}_hi": hi,
                         f"{m}_vs_ref": base[m] - metricas(preds[ref], Z)[m], f"{m}_vs_ref_lo": dlo, f"{m}_vs_ref_hi": dhi})
        filas.append(fila)

    import csv
    with open(os.path.join(args.out, "valor.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    calibracion(preds, Z, os.path.join(args.out, "valor_calibracion.png"))

    lineas = [f"Medicion 1: prediccion del resultado ({len(X)} posiciones de {n} partidas, {args.pgn})",
              f"Referencia para las diferencias: {ref}. IC 95 % por bootstrap de partidas.",
              "Brier y LogLoss: mas bajo es mejor. AUC: mas alto es mejor. Diferencia significativa si su IC no incluye el 0.", ""]
    for fl in filas:
        lineas.append(f"{fl['modelo']}")
        for m, mejor in (("brier", "baja"), ("logloss", "baja"), ("auc", "sube")):
            sig = "" if fl["modelo"] == ref else (
                "  <-- significativa" if (fl[f"{m}_vs_ref_lo"] > 0 or fl[f"{m}_vs_ref_hi"] < 0) else "  (no significativa)")
            lineas.append(f"   {m:8s} {fl[m]:.4f} [{fl[m + '_lo']:.4f}, {fl[m + '_hi']:.4f}]   "
                          f"vs ref {fl[m + '_vs_ref']:+.4f} [{fl[m + '_vs_ref_lo']:+.4f}, {fl[m + '_vs_ref_hi']:+.4f}]{sig}")
    txt = "\n".join(lineas)
    open(os.path.join(args.out, "valor_resumen.txt"), "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\nlisto en {time.time() - t0:.0f} s -> {args.out}/valor.csv, valor_resumen.txt, valor_calibracion.png")


if __name__ == "__main__":
    main()
