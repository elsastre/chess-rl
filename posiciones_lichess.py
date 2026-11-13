"""
Extrae posiciones (FEN) de partidas de Lichess para que el autojuego arranque desde
situaciones reales y variadas (idea de Giraffe), en lugar de siempre desde la posicion inicial.

Uso:
    python posiciones_lichess.py lichess_db_standard_rated_2013-01.pgn.zst --out data/fens.txt
"""
import argparse
import random

import chess.pgn

from dataset_lichess import open_pgn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=100_000)
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--min-elo", type=int, default=1000)
    args = ap.parse_args()

    n_games, n_pos = 0, 0
    with open_pgn(args.pgn) as f, open(args.out, "w") as out:
        while n_games < args.max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            h = game.headers
            try:
                if min(int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))) < args.min_elo:
                    continue
            except ValueError:
                continue
            moves = list(game.mainline_moves())
            if len(moves) <= args.min_ply + 2:
                continue
            chosen = set(random.sample(range(args.min_ply, len(moves) - 1),
                                       min(args.per_game, len(moves) - 1 - args.min_ply)))
            board = game.board()
            for ply, m in enumerate(moves):
                if ply in chosen and not board.is_game_over():
                    out.write(board.fen() + "\n")
                    n_pos += 1
                board.push(m)
            n_games += 1
            if n_games % 10000 == 0:
                print(f"{n_games} partidas, {n_pos} posiciones")
    print(f"listo: {n_pos} posiciones de {n_games} partidas -> {args.out}")


if __name__ == "__main__":
    main()
