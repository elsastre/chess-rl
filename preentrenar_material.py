"""
Inicializacion "pura" (sin partidas humanas): la red aprende a predecir el balance de material,
resolviendo antes los cambios de piezas. Es el punto de partida de KnightCap (todos los
parametros en cero salvo el valor de las piezas) y de Giraffe.

Tarda pocos minutos. Despues se usa como --init en entrenar.py.

Uso:
    python preentrenar_material.py --out checkpoints/material.pt
    python preentrenar_material.py --out checkpoints/material.pt --fens data/fens.txt   (posiciones reales)
"""
import argparse
import math
import random

import chess
import numpy as np
import torch

from encoding import encode
from model import ValueNet, save_checkpoint, default_device
from search import qsearch, INF, MATE


def random_positions(n, max_plies=80):
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(random.randint(0, max_plies)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(random.choice(ms))
        if not b.is_game_over():
            out.append(b)
    return out


def target(board):
    cp, _ = qsearch(board, -INF, INF, 4)
    if abs(cp) >= MATE // 2:
        return 1.0 if cp > 0 else -1.0
    return math.tanh(cp * math.log(10) / 800)     # misma escala que una probabilidad de ganar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fens", default=None)
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    device = default_device()
    if args.fens:
        with open(args.fens) as f:
            fens = [l.strip() for l in f if l.strip()]
        boards = [chess.Board(x) for x in random.sample(fens, min(args.n, len(fens)))]
        # mezclar con posiciones al azar para que vea desequilibrios grandes
        boards += random_positions(len(boards) // 3)
    else:
        boards = random_positions(args.n)
    print(f"{len(boards)} posiciones, calculando objetivos...")
    X = np.stack([encode(b) for b in boards])
    Y = np.array([target(b) for b in boards], dtype=np.float32)
    idx = np.random.permutation(len(X)); cut = int(len(X) * .9)
    tr, va = idx[:cut], idx[cut:]

    model = ValueNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    best, best_state = float("inf"), None
    for ep in range(1, args.epochs + 1):
        model.train()
        for i in range(0, len(tr), args.batch):
            j = tr[i:i + args.batch]
            x = torch.from_numpy(X[j]).float().to(device); y = torch.from_numpy(Y[j]).to(device)
            loss = torch.mean((model(x) - y) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(X[va]).float().to(device)).cpu().numpy()
        mse = float(np.mean((pv - Y[va]) ** 2)); base = float(np.mean((Y[va] - Y[tr].mean()) ** 2))
        print(f"epoca {ep}: MSE val {mse:.4f} (linea base {base:.4f})")
        if mse < best:
            best, best_state = mse, {k: v.detach().clone() for k, v in model.state_dict().items()}
        np.random.shuffle(tr)
    model.load_state_dict(best_state)
    save_checkpoint(args.out, model.eval(), {"games": 0, "fuente": "material", "val_mse": best})
    print(f"mejor MSE val {best:.4f}")
    print(f"listo -> {args.out}")


if __name__ == "__main__":
    main()
