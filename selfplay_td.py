"""
Autojuego con TD-Leaf(lambda): el corazon de RL del proyecto.

Loop:
  1. La red juega contra si misma (con algo de exploracion).
  2. En cada jugada se guarda el valor de la busqueda y la hoja de la variante principal.
  3. Al terminar la partida se calculan los lambda-retornos (hacia atras desde el resultado)
     y la red aprende a que su valor en la hoja se acerque a ese objetivo.
  4. Cada tanto se guarda un checkpoint (para la curva de Elo con evaluar.py).

Objetivo TD (vista hacia adelante, todo desde el punto de vista de las blancas):
    G_T = z  (resultado: 1, 0 o -1)
    G_t = (1 - lambda) * w_{t+1} + lambda * G_{t+1}
lambda = 1 -> aprende solo del resultado final (Monte Carlo)
lambda = 0 -> aprende solo de su propia estimacion siguiente (TD(0))

Uso:
    python selfplay_td.py --out runs/desde_cero
    python selfplay_td.py --out runs/con_imitacion --init checkpoints/supervisado.pt
"""
import argparse
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch
import chess

from encoding import terminal_value
from model import ValueNet, save_checkpoint, load_checkpoint, default_device
from agent import NetSearcher, pick


def play_selfplay_game(searcher, temperature, random_plies, max_plies):
    board = chess.Board()
    record = []  # (w_t, planos_hoja, signo_blancas)

    # aperturas aleatorias para que no se repitan siempre las mismas partidas
    for _ in range(random_plies):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))

    while board.ply() < max_plies and terminal_value(board) is None:
        results = searcher.search(board)
        _, best_score, best_leaf = max(results, key=lambda t: t[1])
        color = 1.0 if board.turn == chess.WHITE else -1.0
        if best_leaf is not None:
            record.append((color * best_score, best_leaf[0], color * best_leaf[1]))
        else:
            record.append((color * best_score, None, None))
        board.push(pick(results, temperature))

    exact = terminal_value(board)
    if exact is None:          # tope de jugadas: tablas
        z = 0.0
    else:                      # exact es para el que mueve; pasamos a blancas
        z = exact if board.turn == chess.WHITE else -exact
    return record, z, board.ply()


def lambda_returns(record, z, lam):
    targets = [0.0] * len(record)
    g = z
    next_w = z
    for t in range(len(record) - 1, -1, -1):
        g = (1 - lam) * next_w + lam * g
        targets[t] = g
        next_w = record[t][0]
    return targets


def train_step(model, opt, buffer, batch_size, device):
    batch = random.sample(buffer, min(batch_size, len(buffer)))
    x = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device)
    sign = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
    y = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
    model.train()
    pred = sign * model(x)
    loss = torch.mean((pred - y) ** 2)
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.eval()
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="carpeta de la corrida")
    ap.add_argument("--init", default=None, help="checkpoint inicial (ej. el supervisado)")
    ap.add_argument("--resume", action="store_true", help="con --init: sigue contando partidas desde ese checkpoint")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=1, help="1 rapido, 2 mas fuerte y ~30x mas lento")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--random-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps-per-game", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = default_device()
    os.makedirs(args.out, exist_ok=True)

    start = 0
    if args.init:
        model, extra = load_checkpoint(args.init, device)
        if args.resume:
            start = int(extra.get("games", 0))
    else:
        model = ValueNet().to(device).eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    searcher = NetSearcher(model, device, args.depth)
    buffer = deque(maxlen=args.buffer)

    log_path = os.path.join(args.out, "log.csv")
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if os.path.getsize(log_path) == 0:
        writer.writerow(["partida", "segundos", "plies", "resultado", "loss", "buffer"])

    if not args.resume:
        save_checkpoint(os.path.join(args.out, "ck_000000.pt"), model, {"games": 0, "args": vars(args)})
    t0 = time.time()
    for g in range(start + 1, start + args.games + 1):
        record, z, plies = play_selfplay_game(searcher, args.temperature, args.random_plies, args.max_plies)
        targets = lambda_returns(record, z, args.lam)
        for (w, planes, sign), y in zip(record, targets):
            if planes is not None:
                buffer.append((planes, sign, y))

        loss = float("nan")
        if len(buffer) >= args.batch:
            loss = np.mean([train_step(model, opt, buffer, args.batch, device)
                            for _ in range(args.steps_per_game)])

        writer.writerow([g, round(time.time() - t0, 1), plies, z, round(loss, 5), len(buffer)])
        if g % 20 == 0:
            log.flush()
            print(f"partida {g:6d} | {time.time() - t0:7.0f}s | plies {plies:3d} | z {z:+.0f} | loss {loss:.4f}")
        if g % args.checkpoint_every == 0:
            path = os.path.join(args.out, f"ck_{g:06d}.pt")
            save_checkpoint(path, model, {"games": g, "seconds": time.time() - t0, "args": vars(args)})
    log.close()


if __name__ == "__main__":
    main()
