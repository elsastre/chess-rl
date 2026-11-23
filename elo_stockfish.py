"""
Elo absoluto contra Stockfish 1320, con el MISMO protocolo con el que se midio el ~1240 de ck_150000 (seccion 4.1
de CLAUDE.md): partidas desde la posicion inicial alternando colores, el agente a profundidad 2 con quiescencia 4,
Stockfish con UCI_LimitStrength y UCI_Elo 1320, 0,05 s por jugada (arena.StockfishPlayer).

Elo del agente = 1320 + 400 log10(s / (1 - s)), con s el puntaje medio; el intervalo sale del intervalo de Wilson
del puntaje (como en evaluar.py).

Por que medir tambien ck_150000: con tiempo fijo la fuerza de Stockfish depende de la maquina (y de cuantas
partidas corren a la vez). Si en esta maquina 150k vuelve a dar ~1240, la medicion es comparable con la vieja; si
no, lo que vale es la diferencia entre las dos redes medidas aca, en las mismas condiciones.

Uso (Mac):
    python elo_stockfish.py --ck rl_150k=runs/final/ck_150000.pt mej_210k=runs/mejoras/ck_210000.pt \\
        --stockfish /opt/homebrew/bin/stockfish --partidas 300 --out runs/elo_stockfish
Se puede cortar y relanzar: sigue desde donde quedo (un CSV por red).
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import time

import chess

MAX_PLIES = 300


def wilson(s, n, z=1.96):
    """Intervalo de Wilson para una proporcion (las tablas cuentan como medio punto)."""
    if n == 0:
        return 0.0, 1.0
    c = (s + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(s * (1 - s) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - m), min(1.0, c + m)


def elo(s, ref=1320):
    s = min(max(s, 1e-3), 1 - 1e-3)
    return ref + 400 * math.log10(s / (1 - s))


def jugar(blancas, negras):
    b = chess.Board()
    while not b.is_game_over(claim_draw=True) and b.ply() < MAX_PLIES:
        b.push((blancas if b.turn == chess.WHITE else negras).choose(b))
    r = b.result(claim_draw=True)
    motivo = "tope_jugadas" if b.ply() >= MAX_PLIES and not b.is_game_over(claim_draw=True) else \
        (b.outcome(claim_draw=True).termination.name.lower() if b.outcome(claim_draw=True) else "?")
    return {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5), b.ply(), motivo


def trabajador(cfg, tareas, resultados):
    import torch
    torch.set_num_threads(1)
    from model import load_checkpoint
    from search import SearchPlayer
    from arena import StockfishPlayer
    m, _ = load_checkpoint(cfg["ck"], "cpu")
    agente = SearchPlayer(m, "cpu", cfg["depth"], cfg["q"], mates=cfg["mates"], poda=cfg["poda"])
    if cfg["prueba"]:                                   # sin Stockfish: solo para probar el script
        from arena import MinimaxPlayer
        rival = MinimaxPlayer(2)
    else:
        rival = StockfishPlayer(cfg["stockfish"], elo=cfg["sf_elo"], nodes=cfg["sf_nodes"])
    try:
        while True:
            t = tareas.get()
            if t is None:
                break
            g = t
            t0 = time.time()
            if g % 2 == 0:
                s, plies, motivo = jugar(agente, rival)
            else:
                s, plies, motivo = jugar(rival, agente)
                s = 1 - s
            resultados.put({"partida": g, "agente_blancas": int(g % 2 == 0), "puntaje": s, "jugadas": plies,
                            "motivo": motivo, "segundos": round(time.time() - t0, 1)})
    finally:
        rival.close() if hasattr(rival, "close") else None


def resumen(nombre, filas, ref):
    n = len(filas)
    if n == 0:
        return f"{nombre}: sin partidas"
    pts = [float(f["puntaje"]) for f in filas]
    s = sum(pts) / n
    lo, hi = wilson(s, n)
    g, e, d = sum(x == 1 for x in pts), sum(x == .5 for x in pts), sum(x == 0 for x in pts)
    bl = [float(f["puntaje"]) for f in filas if int(f["agente_blancas"])]
    ng = [float(f["puntaje"]) for f in filas if not int(f["agente_blancas"])]
    return (f"{nombre}: {n} partidas contra Stockfish {ref}\n"
            f"  puntaje {s:.3f}  IC 95 % (Wilson) [{lo:.3f}, {hi:.3f}]\n"
            f"  Elo ≈ {elo(s, ref):.0f}  [{elo(lo, ref):.0f}, {elo(hi, ref):.0f}]\n"
            f"  {g} victorias, {e} tablas, {d} derrotas"
            f" | con blancas {sum(bl) / max(1, len(bl)):.2f}, con negras {sum(ng) / max(1, len(ng)):.2f}")


def medir(nombre, ruta, args):
    out = os.path.join(args.out, f"elo_{nombre}.csv")
    filas = []
    if os.path.exists(out):
        with open(out) as f:
            filas = list(csv.DictReader(f))
    hechas = {int(r["partida"]) for r in filas}
    faltan = [g for g in range(args.partidas) if g not in hechas]
    print(f"\n== {nombre} ({ruta}): faltan {len(faltan)} de {args.partidas} partidas, {args.workers} procesos ==",
          flush=True)
    if faltan:
        ctx = mp.get_context("spawn")
        tareas, resultados = ctx.Queue(), ctx.Queue()
        for g in faltan:
            tareas.put(g)
        cfg = {"ck": ruta, "depth": args.depth, "q": args.q, "poda": args.poda, "mates": args.mates,
               "stockfish": args.stockfish, "sf_elo": args.sf_elo, "sf_nodes": args.sf_nodes, "prueba": args.prueba}
        procs = [ctx.Process(target=trabajador, args=(cfg, tareas, resultados), daemon=True)
                 for _ in range(args.workers)]
        for _ in procs:
            tareas.put(None)
        for p in procs:
            p.start()
        campos = ["partida", "agente_blancas", "puntaje", "jugadas", "motivo", "segundos"]
        nuevo = not os.path.exists(out)
        t0 = time.time()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            if nuevo:
                w.writeheader()
            for k in range(1, len(faltan) + 1):
                r = resultados.get()
                w.writerow(r)
                f.flush()
                filas.append(r)
                if k % 20 == 0 or k == len(faltan):
                    resta = (time.time() - t0) / k * (len(faltan) - k) / 60
                    lin = resumen(nombre, filas, args.sf_elo).split("\n")
                    print(f"[{k}/{len(faltan)}] faltan ~{resta:.0f} min |{lin[1]} |{lin[2]}", flush=True)
        for p in procs:
            p.join(timeout=10)
    return resumen(nombre, filas, args.sf_elo)


def main():
    ap = argparse.ArgumentParser(description="Elo absoluto contra Stockfish 1320 (mismo protocolo que el ~1240)")
    ap.add_argument("--ck", nargs="+", required=True, help="nombre=ruta")
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--partidas", type=int, default=300)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--poda", action="store_true", default=True)
    ap.add_argument("--mates", action="store_true", help="como el robot (el ~1240 se midio sin --mates)")
    ap.add_argument("--sf-elo", type=int, default=1320)
    ap.add_argument("--sf-nodes", type=int, default=None, help="Stockfish por nodos en vez de 0,05 s")
    ap.add_argument("--workers", type=int, default=4,
                    help="pocos: con tiempo fijo, si Stockfish comparte la CPU juega algo mas flojo")
    ap.add_argument("--out", default="runs/elo_stockfish")
    ap.add_argument("--prueba", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    textos = []
    for par in args.ck:
        nombre, ruta = par.split("=", 1)
        textos.append(medir(nombre, ruta, args))
    cab = (f"Elo contra Stockfish {args.sf_elo} ("
           f"{f'{args.sf_nodes} nodos' if args.sf_nodes else '0,05 s'} por jugada), agente a profundidad "
           f"{args.depth}, quiescencia {args.q}{', mates' if args.mates else ''}, desde la posicion inicial")
    txt = cab + "\n\n" + "\n\n".join(textos)
    print("\n" + txt)
    open(os.path.join(args.out, "resumen.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
