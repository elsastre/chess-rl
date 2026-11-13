"""
Entrenador v2: el corazon de RL del proyecto.

Novedades respecto de selfplay_td.py (que queda como la version 1, para comparar):
  * Busqueda con capturas (search.py): la red evalua solo posiciones tranquilas.
  * Algoritmo a eleccion:
      treestrap  cada nodo que visita la busqueda se corrige hacia su valor de busqueda
                 (Veness, Silver et al. 2009, "Bootstrapping from Game Tree Search").
      tdleaf     TD-Leaf(lambda) sobre la hoja de la variante principal (Baxter et al. 1998).
      mixto      las dos senales juntas (la del resultado ancla, la de TreeStrap da volumen).
  * Partidas que arrancan desde posiciones reales de Lichess + 1 jugada al azar (Giraffe),
    jugando un tramo de N jugadas; al cortar, el objetivo se toma de la propia busqueda.
  * Rivales mezclados: el mismo agente, versiones viejas y Stockfish con curriculo
    (su nivel sube o baja para que el agente gane mas o menos la mitad).
  * Varios procesos jugando en paralelo (--workers), uno entrenando en la GPU.
  * Perdida de Huber (los errores de TD tienen valores atipicos, ver Giraffe).

Uso tipico:
  # desde cero, solo TreeStrap y autojuego (experimento de comparacion con TD-Leaf v1)
  python entrenar.py --out runs/treestrap_cero --algo treestrap --workers 6

  # bot final: arranca de la red supervisada, posiciones de Lichess, rivales mezclados
  python entrenar.py --out runs/final --init checkpoints/supervisado.pt --algo mixto ^
      --start lichess --fens data/fens.txt --stockfish "C:\\ruta\\stockfish.exe" --workers 6

  # seguir una corrida cortada
  python entrenar.py --out runs/final --init runs/final/ck_004000.pt --resume ...
"""
import argparse
import csv
import glob
import math
import os
import queue as queue_mod
import random
import time
from collections import deque

import chess
import chess.engine
import numpy as np
import torch
import torch.multiprocessing as mp

from encoding import pack, unpack, terminal_value
from model import ValueNet, save_checkpoint, load_checkpoint, default_device
from search import search, make_evaluator, pick_move

# Niveles de Stockfish para el curriculo: (Skill Level 0-20, limite de nodos por jugada)
SF_LEVELS = [(0, 30), (0, 100), (1, 300), (3, 700), (5, 1500), (7, 3000), (9, 6000),
             (11, 12000), (13, 25000), (16, 50000), (20, 100000)]


# =============================================================== una partida
def load_fens(path, limit=500_000):
    fens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                fens.append(line)
            if len(fens) >= limit:
                break
    return fens


def start_board(cfg, fens):
    if cfg["start"] == "lichess" and fens:
        board = chess.Board(random.choice(fens))
        moves = list(board.legal_moves)
        if moves:
            board.push(random.choice(moves))          # una jugada al azar, como Giraffe
        return board
    board = chess.Board()
    for _ in range(cfg["random_plies"]):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    return board


def lambda_returns(ws, z, lam):
    out, g, nxt = [0.0] * len(ws), z, z
    for t in range(len(ws) - 1, -1, -1):
        g = (1 - lam) * nxt + lam * g
        out[t] = g
        nxt = ws[t]
    return out


def play_game(evaluate, cfg, fens, opponent=None, agent_color=None):
    """opponent: None (autojuego) | ('red', evaluate_viejo) | ('sf', engine, limit).
    Devuelve (muestras, info). Muestra = (planos empaquetados, signo, objetivo)."""
    board = start_board(cfg, fens)
    collect = cfg["algo"] in ("treestrap", "mixto")
    use_td = cfg["algo"] in ("tdleaf", "mixto")
    # en autojuego desde Lichess se juegan tramos cortos; contra rivales, partidas completas
    # (para que el curriculo de Stockfish se base en resultados reales)
    max_plies = cfg["segment"] if (cfg["start"] == "lichess" and opponent is None) else cfg["max_plies"]
    samples, td_rec = [], []
    plies = 0
    while True:
        exact = terminal_value(board)
        if exact is not None:
            z_white = exact if board.turn == chess.WHITE else -exact
            finished = True
            break
        if plies >= max_plies:
            z_white = td_rec[-1][0] if td_rec else 0.0     # se corta: el objetivo sale de la propia busqueda
            finished = False
            break
        agent_turn = opponent is None or board.turn == agent_color
        if agent_turn:
            res = search(board, evaluate, cfg["depth"], cfg["qdepth"], collect=collect)
            if collect:
                samples.extend((pack(p), 1.0, t) for p, t in res.treestrap)
            color = 1.0 if board.turn == chess.WHITE else -1.0
            leaf = None if res.pv_leaf is None else (pack(res.pv_leaf[0]), color * res.pv_leaf[1])
            td_rec.append((color * res.value, leaf))
            move = pick_move(res.moves, cfg["temperature"])
        elif opponent[0] == "red":
            move = pick_move(search(board, opponent[1], cfg["depth"], cfg["qdepth"]).moves, 0.0)
        else:
            move = opponent[1].play(board, opponent[2]).move
        board.push(move)
        plies += 1
    if use_td and td_rec:
        targets = lambda_returns([w for w, _ in td_rec], z_white, cfg["lam"])
        for (w, leaf), y in zip(td_rec, targets):
            if leaf is not None:
                samples.append((leaf[0], leaf[1], float(y)))
    if opponent is None:
        result = z_white                                    # autojuego: resultado desde blancas
    else:
        result = z_white if agent_color == chess.WHITE else -z_white
    return samples, {"plies": plies, "result": result, "finished": finished}


# =============================================================== rivales
class Opponents:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.engine = None
        self.old_cache = {}
        if cfg["stockfish"] and cfg["p_stockfish"] > 0:
            self.engine = chess.engine.SimpleEngine.popen_uci(cfg["stockfish"])

    def pick(self, sf_level):
        r = random.random()
        cfg = self.cfg
        if self.engine and r < cfg["p_stockfish"]:
            skill, nodes = SF_LEVELS[sf_level]
            try:
                self.engine.configure({"Skill Level": skill})
            except chess.engine.EngineError:
                pass
            return ("sf", self.engine, chess.engine.Limit(nodes=nodes)), f"stockfish{sf_level}"
        if r < cfg["p_stockfish"] + cfg["p_viejos"]:
            cks = sorted(glob.glob(os.path.join(cfg["out"], "ck_*.pt")))
            if len(cks) >= 2:
                path = random.choice(cks[:-1])
                if path not in self.old_cache:
                    if len(self.old_cache) >= 3:
                        self.old_cache.pop(next(iter(self.old_cache)))
                    m, _ = load_checkpoint(path, self.device)
                    self.old_cache[path] = make_evaluator(m, self.device)
                return ("red", self.old_cache[path]), "viejo:" + os.path.basename(path)
        return None, "si_mismo"

    def close(self):
        if self.engine:
            self.engine.quit()


# =============================================================== procesos que juegan
def worker_main(wid, cfg, q, stop, sf_level, seed):
    random.seed(seed); np.random.seed(seed % (2 ** 31)); torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = cfg["worker_device"]
    latest = os.path.join(cfg["out"], "latest.pt")
    fens = load_fens(cfg["fens"]) if cfg["start"] == "lichess" else []
    model, mtime, evaluate = None, None, None
    opp = Opponents(cfg, device)
    try:
        while not stop.is_set():
            try:
                mt = os.path.getmtime(latest)
                if mt != mtime:
                    model, _ = load_checkpoint(latest, device)
                    evaluate, mtime = make_evaluator(model, device), mt
            except (OSError, RuntimeError, EOFError):
                if evaluate is None:
                    time.sleep(1)
                    continue
            opponent, name = opp.pick(sf_level.value)
            color = random.choice([chess.WHITE, chess.BLACK]) if opponent else None
            samples, info = play_game(evaluate, cfg, fens, opponent, color)
            info["opponent"], info["sf_level"] = name, sf_level.value
            q.put((samples, info))
    finally:
        opp.close()


# =============================================================== aprendiz
def train_steps(model, opt, buffer, n_steps, batch, device):
    losses = []
    model.train()
    for _ in range(n_steps):
        b = random.sample(buffer, min(batch, len(buffer)))
        x = torch.from_numpy(unpack(np.stack([s[0] for s in b]))).float().to(device)
        sign = torch.tensor([s[1] for s in b], dtype=torch.float32, device=device)
        y = torch.tensor([s[2] for s in b], dtype=torch.float32, device=device)
        loss = torch.nn.functional.smooth_l1_loss(sign * model(x), y, beta=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    model.eval()
    return float(np.mean(losses)) if losses else float("nan")


def safe_save(path, model, extra):
    tmp = path + ".tmp"
    save_checkpoint(tmp, model, extra)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="checkpoint inicial (supervisado, material o una corrida)")
    ap.add_argument("--resume", action="store_true", help="con --init: sigue contando partidas")
    ap.add_argument("--algo", choices=["treestrap", "tdleaf", "mixto"], default="mixto")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=1, help="profundidad de entrenamiento (1 rapido, 2 mas fuerte y ~30x mas lento)")
    ap.add_argument("--qdepth", type=int, default=4, help="largo maximo de la secuencia de capturas")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--start", choices=["inicial", "lichess"], default="inicial")
    ap.add_argument("--fens", default=None, help="archivo de posiciones (posiciones_lichess.py)")
    ap.add_argument("--segment", type=int, default=30, help="jugadas por tramo cuando --start lichess")
    ap.add_argument("--random-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--stockfish", default=None)
    ap.add_argument("--p-stockfish", type=float, default=0.25)
    ap.add_argument("--p-viejos", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0, help="0 = todo en un proceso")
    ap.add_argument("--worker-device", default="cpu", help="cpu o cuda para los procesos que juegan")
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--replay", type=float, default=2.0, help="cuantas veces se usa cada muestra en promedio")
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--sync-every", type=int, default=20, help="cada cuantas partidas los workers reciben la red nueva")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.start == "lichess" and not args.fens:
        ap.error("--start lichess necesita --fens (generalo con posiciones_lichess.py)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = default_device()
    os.makedirs(args.out, exist_ok=True)
    cfg = {k: v for k, v in vars(args).items()}

    start_games = 0
    if args.init:
        model, extra = load_checkpoint(args.init, device)
        if args.resume:
            start_games = int(extra.get("games", 0))
    else:
        model = ValueNet().to(device).eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    latest = os.path.join(args.out, "latest.pt")
    safe_save(latest, model, {"games": start_games})
    if not args.resume:
        save_checkpoint(os.path.join(args.out, f"ck_{start_games:06d}.pt"), model, {"games": start_games, "args": cfg})

    log_path = os.path.join(args.out, "log.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["partida", "segundos", "oponente", "nivel_sf", "plies", "resultado", "termino",
                         "muestras", "loss", "buffer"])

    buffer = deque(maxlen=args.buffer)
    sf_level = mp.Value("i", 0) if args.workers > 0 else type("V", (), {"value": 0})()
    sf_recent = deque(maxlen=30)
    pending_samples = 0.0
    loss = float("nan")
    t0 = time.time()

    procs, q, stop = [], None, None
    if args.workers > 0:
        ctx = mp.get_context("spawn")
        q, stop = ctx.Queue(maxsize=4 * args.workers), ctx.Event()
        sf_level = ctx.Value("i", 0)
        for w in range(args.workers):
            p = ctx.Process(target=worker_main, args=(w, cfg, q, stop, sf_level, args.seed * 1000 + w + 1), daemon=True)
            p.start(); procs.append(p)
        print(f"{args.workers} procesos jugando, entrenamiento en {device}")
    else:
        fens = load_fens(args.fens) if args.start == "lichess" else []
        opp = Opponents(cfg, device)
        evaluate = make_evaluator(model, device)

    g = start_games
    try:
        while g < start_games + args.games:
            if args.workers > 0:
                try:
                    samples, info = q.get(timeout=120)
                except queue_mod.Empty:
                    if not any(p.is_alive() for p in procs):
                        raise RuntimeError("los procesos que juegan se cerraron; revisa los mensajes de error")
                    continue
            else:
                opponent, name = opp.pick(sf_level.value)
                color = random.choice([chess.WHITE, chess.BLACK]) if opponent else None
                samples, info = play_game(evaluate, cfg, fens, opponent, color)
                info["opponent"], info["sf_level"] = name, sf_level.value
            g += 1
            buffer.extend(samples)
            pending_samples += len(samples) * args.replay / args.batch
            if len(buffer) >= args.batch and pending_samples >= 1:
                n = int(pending_samples); pending_samples -= n
                loss = train_steps(model, opt, buffer, n, args.batch, device)
                if args.workers == 0:
                    evaluate = make_evaluator(model, device)

            if info["opponent"].startswith("stockfish"):          # curriculo
                sf_recent.append((info["result"] + 1) / 2)
                if len(sf_recent) >= 20:
                    rate = sum(sf_recent) / len(sf_recent)
                    if rate > 0.6 and sf_level.value < len(SF_LEVELS) - 1:
                        sf_level.value += 1; sf_recent.clear()
                    elif rate < 0.4 and sf_level.value > 0:
                        sf_level.value -= 1; sf_recent.clear()

            writer.writerow([g, round(time.time() - t0, 1), info["opponent"], info["sf_level"], info["plies"],
                             info["result"], int(info["finished"]), len(samples), round(loss, 5), len(buffer)])
            if g % args.sync_every == 0:
                safe_save(latest, model, {"games": g})
                log.flush()
            if g % 20 == 0:
                rate = (g - start_games) / (time.time() - t0) * 3600
                print(f"partida {g:6d} | {time.time() - t0:7.0f}s | {rate:5.0f} partidas/h | loss {loss:.4f} "
                      f"| buffer {len(buffer)} | nivel stockfish {sf_level.value}")
            if g % args.checkpoint_every == 0:
                save_checkpoint(os.path.join(args.out, f"ck_{g:06d}.pt"), model,
                                {"games": g, "seconds": time.time() - t0, "sf_level": sf_level.value, "args": cfg})
    finally:
        if stop is not None:
            stop.set()
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
        elif args.workers == 0:
            opp.close()
        safe_save(latest, model, {"games": g})
        log.close()


if __name__ == "__main__":
    main()
