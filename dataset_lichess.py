"""
Arma el dataset del arranque supervisado a partir de partidas de Lichess.

Para cada partida toma algunas posiciones al azar (salteando la apertura) y les pone
como objetivo el resultado final DESDE EL BANDO QUE MUEVE: 1 si ese bando termino
ganando, 0 tablas, -1 si perdio.

Guarda tambien el id de la partida: el split train/val TIENE que ser por partida,
no por posicion. Dos posiciones de la misma partida comparten el resultado y casi
todas las piezas; si quedan una en train y otra en val, la validacion se infla.

Uso (acepta .pgn o .pgn.zst directo de database.lichess.org):
    python dataset_lichess.py lichess_db_standard_rated_2013-01.pgn.zst --out data/lichess_2013_01.npz
"""
import argparse
import io
import random

import numpy as np
import chess.pgn

from encoding import encode, pack

RESULT = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def open_pgn(path):
    if path.endswith(".zst"):
        import zstandard
        fh = open(path, "rb")
        stream = zstandard.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=200_000)
    ap.add_argument("--per-game", type=int, default=8, help="posiciones por partida")
    ap.add_argument("--skip-plies", type=int, default=10, help="saltear la apertura")
    ap.add_argument("--min-elo", type=int, default=1200, help="ambos jugadores por encima de esto")
    args = ap.parse_args()

    X, Y, G = [], [], []
    games_used = 0
    with open_pgn(args.pgn) as f:
        while games_used < args.max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            h = game.headers
            if h.get("Result") not in RESULT:
                continue
            try:
                if min(int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))) < args.min_elo:
                    continue
            except ValueError:
                continue
            if "abandoned" in h.get("Termination", "").lower():
                continue

            z_white = RESULT[h["Result"]]
            moves = list(game.mainline_moves())
            if len(moves) <= args.skip_plies:
                continue
            chosen = set(random.sample(range(args.skip_plies, len(moves)),
                                       min(args.per_game, len(moves) - args.skip_plies)))
            board = game.board()
            for ply, move in enumerate(moves):
                if ply in chosen:
                    z = z_white if board.turn == chess.WHITE else -z_white
                    X.append(pack(encode(board))); Y.append(z); G.append(games_used)
                board.push(move)

            games_used += 1
            if games_used % 5000 == 0:
                print(f"{games_used} partidas, {len(X)} posiciones")

    np.savez_compressed(args.out, x=np.stack(X), y=np.array(Y, np.float32), game=np.array(G, np.int32))
    print(f"listo: {games_used} partidas, {len(X)} posiciones -> {args.out}")


if __name__ == "__main__":
    main()
