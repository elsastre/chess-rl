"""
Curva de aprendizaje: evalua checkpoints de una corrida contra la escalera fija.

Uso:
    python evaluar.py --run runs/desde_cero --every 10 --games 20
    python evaluar.py --run runs/desde_cero --every 10 --games 20 --rivals aleatorio,codicioso
    python evaluar.py --run runs/desde_cero --stockfish "C:\\ruta\\stockfish.exe"
    python evaluar.py --run runs/final --depth 2 --q 4      (busqueda nueva con capturas)

--every N   evalua uno de cada N checkpoints (siempre incluye el primero y el ultimo)
--rivals    lista separada por comas: aleatorio, codicioso, minimax2 (el mas lento)

Escribe runs/<corrida>/curva.csv. Los checkpoints ya evaluados se saltean,
asi que se puede volver a correr para completar huecos.
"""
import argparse
import csv
import glob
import os
import time

from arena import RandomPlayer, GreedyPlayer, MinimaxPlayer, StockfishPlayer, match, elo_diff
from model import load_checkpoint, default_device
from agent import NetPlayer
from search import SearchPlayer

RIVALS = {"aleatorio": RandomPlayer, "codicioso": GreedyPlayer, "minimax2": lambda: MinimaxPlayer(2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--q", type=int, default=None, help="usa la busqueda nueva con capturas de este largo")
    ap.add_argument("--rivals", default="aleatorio,codicioso,minimax2")
    ap.add_argument("--stockfish", default=None)
    args = ap.parse_args()

    device = default_device()
    rivals = [RIVALS[name.strip()]() for name in args.rivals.split(",")]
    if args.stockfish:
        rivals.append(StockfishPlayer(args.stockfish))

    paths = sorted(glob.glob(os.path.join(args.run, "ck_*.pt")))
    chosen = paths[::args.every]
    if paths and paths[-1] not in chosen:
        chosen.append(paths[-1])

    if args.q is not None:
        out = os.path.join(args.run, f"curva_d{args.depth}q{args.q}.csv")
    else:
        out = os.path.join(args.run, "curva.csv" if args.depth == 1 else f"curva_d{args.depth}.csv")
    rows, fields = {}, ["checkpoint", "partidas_entrenadas"]
    if os.path.exists(out):
        with open(out) as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames)
            rows = {row["checkpoint"]: row for row in reader}
    for r in rivals:                       # columnas nuevas sin perder las viejas
        for col in (f"puntaje_{r.name}", f"elo_vs_{r.name}"):
            if col not in fields:
                fields.append(col)

    # solo se juega lo que falta: checkpoint sin evaluar, o rival nuevo para ese checkpoint
    todo = [(p, [r for r in rivals if not rows.get(os.path.basename(p), {}).get(f"puntaje_{r.name}")])
            for p in chosen]
    todo = [(p, rs) for p, rs in todo if rs]
    print(f"{len(paths)} checkpoints, evaluo {len(todo)} ({args.games} partidas contra cada rival pendiente)")
    t0 = time.time()
    for i, (path, pending) in enumerate(todo, 1):
        name = os.path.basename(path)
        model, extra = load_checkpoint(path, device)
        agent = (SearchPlayer(model, device, args.depth, args.q) if args.q is not None
                 else NetPlayer(model, device, args.depth))
        row = rows.setdefault(name, {"checkpoint": name, "partidas_entrenadas": extra.get("games", "")})
        for r in pending:
            s = match(agent, r, args.games)
            row[f"puntaje_{r.name}"] = round(s, 3)
            row[f"elo_vs_{r.name}"] = round(elo_diff(s, args.games))
        ordered = sorted(rows.values(), key=lambda d: int(d["partidas_entrenadas"] or 0))
        with open(out, "w", newline="") as f:          # se reescribe ordenado en cada paso
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(ordered)
        el = time.time() - t0
        resumen = "  ".join(f"{r.name} {row[f'puntaje_{r.name}']:.2f}" for r in pending)
        print(f"[{i}/{len(todo)}] {name}: {resumen}  | faltan ~{el / i * (len(todo) - i) / 60:.0f} min")

    for r in rivals:
        r.close()
    print(f"listo -> {out}")


if __name__ == "__main__":
    main()
