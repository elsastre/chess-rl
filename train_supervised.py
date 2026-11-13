"""
Arranque supervisado: la red aprende a predecir el resultado de partidas humanas.

Es el "paso de imitacion" de la receta tipo AlphaGo. Despues, selfplay_td.py
arranca desde este checkpoint (--init) y lo mejora jugando contra si mismo.

Validacion separada POR PARTIDA (ver dataset_lichess.py).
Reporta tambien el error de predecir siempre la media, como linea base.

Uso:
    python train_supervised.py data/lichess_2013_01.npz --out checkpoints/supervisado.pt
"""
import argparse
import numpy as np
import torch

from encoding import unpack
from model import ValueNet, save_checkpoint, default_device


def batches(idx, x, y, size, device, shuffle):
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), size):
        j = idx[i:i + size]
        xb = torch.from_numpy(unpack(x[j])).float().to(device)
        yb = torch.from_numpy(y[j]).to(device)
        yield xb, yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = default_device()
    d = np.load(args.data)
    x, y, game = d["x"], d["y"], d["game"]

    games = np.unique(game)
    val_games = np.random.choice(games, int(len(games) * args.val_frac), replace=False)
    is_val = np.isin(game, val_games)
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    base = float(np.mean((y[va] - y[tr].mean()) ** 2))
    print(f"train {len(tr)} | val {len(va)} posiciones | MSE linea base (predecir la media): {base:.4f}")

    model = ValueNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for xb, yb in batches(tr, x, y, args.batch, device, True):
            loss = torch.mean((model(xb) - yb) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(yb); n += len(yb)
        model.eval()
        va_loss, m = 0.0, 0
        with torch.no_grad():
            for xb, yb in batches(va, x, y, 4096, device, False):
                va_loss += torch.sum((model(xb) - yb) ** 2).item(); m += len(yb)
        tr_loss, va_loss = tr_loss / n, va_loss / m
        print(f"epoca {ep}: train {tr_loss:.4f} | val {va_loss:.4f} | base {base:.4f}")
        if va_loss < best:
            best = va_loss
            save_checkpoint(args.out, model.eval(), {"games": 0, "val_mse": va_loss, "fuente": args.data})
    print(f"mejor val MSE {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
