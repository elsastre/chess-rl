"""
Medicion 2: enfrentamiento directo entre dos redes, con la MISMA busqueda, desde aperturas pareadas.

Por que asi:
  * Directo (A contra B) es mucho mas sensible que comparar a las dos contra un tercero (Stockfish):
    la diferencia se mide sin el ruido del rival.
  * Las dos redes juegan de forma determinista. Desde la posicion inicial, las partidas se repetirian
    identicas. Por eso cada par arranca de una posicion distinta (de partidas reales, con material parejo).
  * Cada posicion se juega DOS veces cambiando los colores: la ventaja o desventaja de la apertura se
    cancela. La incertidumbre se calcula por par, que es la unidad independiente.

Resultado: puntaje de A (victoria 1, tablas 0,5), diferencia de Elo con IC 95 %, y victorias/tablas/derrotas.
Se guarda una fila por partida en el CSV; si se corta, al relanzar sigue con los pares que faltan.

Uso:
    python duelo.py --a runs/final/ck_150000.pt --b checkpoints/material_regen.pt --pares 300 \\
        --out runs/mediciones/duelo_rl150k_vs_material.csv --workers 6
"""
import argparse
import csv
import math
import os
import random
import time

import chess
import torch.multiprocessing as mp

from search import qsearch, INF

MAX_PLIES = 300


def elegir_aperturas(fens_path, n, semilla, max_cp=100):
    """Posiciones de partidas reales: entre la jugada 5 y la 15, sin jaque y con material parejo (±1 peon)."""
    rng = random.Random(semilla)
    with open(fens_path) as f:
        fens = [l.strip() for l in f if l.strip()]
    rng.shuffle(fens)
    out = []
    for fen in fens:
        b = chess.Board(fen)
        if not (5 <= b.fullmove_number <= 15) or b.is_check() or b.outcome() is not None:
            continue
        cp, _ = qsearch(b, -INF, INF, 4)
        if abs(cp) <= max_cp:
            out.append(fen)
        if len(out) >= n:
            break
    return out


def jugar(fen, blancas, negras):
    """Devuelve (puntaje de las blancas, jugadas, motivo)."""
    b = chess.Board(fen)
    while True:
        if b.is_game_over(claim_draw=True):
            r = b.result(claim_draw=True)
            return {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5), b.ply(), b.outcome(claim_draw=True).termination.name.lower()
        if b.ply() >= MAX_PLIES:
            return 0.5, b.ply(), "tope_jugadas"
        jugador = blancas if b.turn == chess.WHITE else negras
        b.push(jugador.choose(b))


def trabajador(args, tareas, resultados):
    import torch
    torch.set_num_threads(1)
    from model import load_checkpoint
    from search import SearchPlayer
    jugadores = {}
    for lado, ruta in (("A", args["a"]), ("B", args["b"])):
        m, _ = load_checkpoint(ruta, "cpu")
        jugadores[lado] = SearchPlayer(m, "cpu", args["depth"], args["q"], name=lado,
                                       mates=args["mates"], poda=args["poda"])
    while True:
        t = tareas.get()
        if t is None:
            break
        par, fen, a_blancas = t
        t0 = time.time()
        if a_blancas:
            s_blancas, plies, motivo = jugar(fen, jugadores["A"], jugadores["B"])
            s_a = s_blancas
        else:
            s_blancas, plies, motivo = jugar(fen, jugadores["B"], jugadores["A"])
            s_a = 1 - s_blancas
        resultados.put({"par": par, "a_blancas": int(a_blancas), "fen": fen, "puntaje_a": s_a,
                        "jugadas": plies, "motivo": motivo, "segundos": round(time.time() - t0, 1)})


def resumen(filas, nombre_a, nombre_b):
    pares = {}
    for f in filas:
        pares.setdefault(int(f["par"]), []).append(float(f["puntaje_a"]))
    completos = [sum(v) for v in pares.values() if len(v) == 2]           # 0, 0,5, 1, 1,5 o 2 por par
    n = len(completos)
    if n < 2:
        return "todavia no hay pares completos"
    media = sum(completos) / n
    var = sum((x - media) ** 2 for x in completos) / (n - 1)
    p, se = media / 2, math.sqrt(var / n) / 2
    elo = lambda q: 400 * math.log10(min(max(q, 1e-3), 1 - 1e-3) / (1 - min(max(q, 1e-3), 1 - 1e-3)))
    lo, hi = p - 1.96 * se, p + 1.96 * se
    pts = [float(f["puntaje_a"]) for f in filas]
    g, e, d = sum(x == 1 for x in pts), sum(x == .5 for x in pts), sum(x == 0 for x in pts)
    sig = "SIGNIFICATIVA" if (lo > .5 or hi < .5) else "no significativa"
    return (f"{nombre_a} contra {nombre_b}: {n} pares ({2 * n} partidas)\n"
            f"  puntaje de {nombre_a}: {p:.3f}  IC 95 % [{lo:.3f}, {hi:.3f}]  -> diferencia {sig}\n"
            f"  Elo de {nombre_a} respecto de {nombre_b}: {elo(p):+.0f}  [{elo(lo):+.0f}, {elo(hi):+.0f}]\n"
            f"  partidas: {g} victorias, {e} tablas, {d} derrotas de {nombre_a}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint A (el que se mide)")
    ap.add_argument("--b", required=True, help="checkpoint B (la referencia)")
    ap.add_argument("--nombre-a", default=None)
    ap.add_argument("--nombre-b", default=None)
    ap.add_argument("--pares", type=int, default=300, help="cada par = 2 partidas desde la misma posicion")
    ap.add_argument("--fens", default="data/fens.txt")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--poda", action="store_true", help="profundidad 2 con poda alfa-beta (mismo valor, mas rapido)")
    ap.add_argument("--mates", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", required=True)
    ap.add_argument("--semilla", type=int, default=0, help="misma semilla = mismas aperturas en todos los duelos")
    args = ap.parse_args()
    na = args.nombre_a or os.path.basename(args.a)[:-3]
    nb = args.nombre_b or os.path.basename(args.b)[:-3]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    aperturas = elegir_aperturas(args.fens, args.pares, args.semilla)
    hechas, filas = set(), []
    if os.path.exists(args.out):
        with open(args.out) as f:
            filas = list(csv.DictReader(f))
        hechas = {(int(r["par"]), int(r["a_blancas"])) for r in filas}
    tareas_lista = [(i, fen, ab) for i, fen in enumerate(aperturas) for ab in (True, False) if (i, int(ab)) not in hechas]
    print(f"{na} contra {nb} | profundidad {args.depth}{' con poda' if args.poda else ''}"
          f"{' y mates' if args.mates else ''} | {len(aperturas)} aperturas, faltan {len(tareas_lista)} partidas, "
          f"{args.workers} procesos", flush=True)
    if not tareas_lista:
        print(resumen(filas, na, nb)); return

    ctx = mp.get_context("spawn")
    tareas, resultados = ctx.Queue(), ctx.Queue()
    for t in tareas_lista:
        tareas.put(t)
    cfg = {"a": args.a, "b": args.b, "depth": args.depth, "q": args.q, "poda": args.poda, "mates": args.mates}
    procs = [ctx.Process(target=trabajador, args=(cfg, tareas, resultados), daemon=True) for _ in range(args.workers)]
    for _ in procs:
        tareas.put(None)
    for p in procs:
        p.start()

    campos = ["par", "a_blancas", "fen", "puntaje_a", "jugadas", "motivo", "segundos"]
    nuevo = not os.path.exists(args.out)
    t0 = time.time()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        if nuevo:
            w.writeheader()
        for k in range(1, len(tareas_lista) + 1):
            r = resultados.get()
            w.writerow(r); f.flush()
            filas.append(r)
            if k % 20 == 0 or k == len(tareas_lista):
                falta = (time.time() - t0) / k * (len(tareas_lista) - k) / 60
                print(f"[{k}/{len(tareas_lista)}] faltan ~{falta:.0f} min | " + resumen(filas, na, nb).split("\n")[1].strip(),
                      flush=True)
    for p in procs:
        p.join(timeout=10)
    txt = resumen(filas, na, nb)
    print("\n" + txt)
    with open(args.out[:-4] + "_resumen.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
