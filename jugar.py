"""
Jugar contra el bot en la terminal. Muestra el tablero, lo que piensa y como "se siente"
(la misma logica que va a usar la cara del robot).

Uso:
    python jugar.py --ck runs\\final\\ck_150000.pt                 (jugas con blancas)
    python jugar.py --ck runs\\final\\ck_150000.pt --color negras
    python jugar.py --ck runs\\final\\ck_150000.pt --depth 1        (mas rapido, mas debil)

Las jugadas se escriben en notacion algebraica (e4, Nf3, O-O, exd5, e8=Q) o UCI (e2e4).
Comandos: "atras" deshace tu ultima jugada, "salir" termina.
"""
import argparse
import sys

import chess

from model import load_checkpoint, default_device
from search import search, make_evaluator, pick_move

NAMES = {chess.KNIGHT: "Caballo", chess.BISHOP: "Alfil", chess.ROOK: "Torre", chess.QUEEN: "Dama",
         chess.KING: "Rey", chess.PAWN: "Peón"}


def emotion(value, td):
    """value: chance del bot en [-1, 1]; td: cambio de su estimacion tras tu jugada."""
    if abs(td) > 0.16:
        return "sorprendido"
    p = (value + 1) / 2
    if p >= 0.75:
        return "ganando"
    if p >= 0.5:
        return "confiado"
    if p >= 0.3:
        return "preocupado"
    return "perdiendo"


def speak(board, move):
    if board.is_castling(move):
        return "Enroque corto" if chess.square_file(move.to_square) == 6 else "Enroque largo"
    piece = board.piece_type_at(move.from_square)
    txt = NAMES[piece] + (" come en " if board.is_capture(move) else " a ") + chess.square_name(move.to_square)
    if move.promotion:
        txt += " y corona"
    board.push(move)
    if board.is_checkmate():
        txt += ". ¡Jaque mate!"
    elif board.is_check():
        txt += ", jaque"
    board.pop()
    return txt


def show(board, flip):
    rows = str(board.unicode(invert_color=True, borders=False, empty_square="·")).split("\n")
    ranks = list(range(8, 0, -1))
    if flip:
        rows = [r[::-1] for r in rows[::-1]]
        ranks = ranks[::-1]
    print()
    for r, row in zip(ranks, rows):
        print(f"  {r}  {row}")
    print("     " + (" ".join("hgfedcba") if flip else " ".join("abcdefgh")))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--color", choices=["blancas", "negras"], default="blancas")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--q", type=int, default=4)
    args = ap.parse_args()

    device = default_device()
    model, extra = load_checkpoint(args.ck, device)
    evaluate = make_evaluator(model, device)
    human = chess.WHITE if args.color == "blancas" else chess.BLACK
    board = chess.Board()
    last_value = 0.0
    print(f"Bot cargado ({extra.get('games', '?')} partidas de entrenamiento). Jugás con {args.color}.")

    while not board.is_game_over(claim_draw=True):
        show(board, flip=(human == chess.BLACK))
        if board.turn == human:
            txt = input("Tu jugada: ").strip()
            if txt in ("salir", "exit", "quit"):
                return
            if txt == "atras":
                if len(board.move_stack) >= 2:
                    board.pop(); board.pop()
                continue
            try:
                move = board.parse_san(txt)
            except ValueError:
                try:
                    move = chess.Move.from_uci(txt)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("  No entiendo esa jugada o no es legal. Ejemplos: e4, Nf3, O-O, e2e4")
                    continue
            board.push(move)
        else:
            print("  El bot está pensando...")
            res = search(board, evaluate, args.depth, args.q)
            td = res.value - last_value
            move = pick_move(res.moves, 0.0)
            best = sorted(res.moves, key=lambda t: -t[1])[:3]
            print(f"  Chance de ganar del bot: {round((res.value + 1) / 2 * 100)} %   "
                  f"sorpresa (error TD): {td:+.2f}   emoción: {emotion(res.value, td)}")
            print("  Lo que consideró: " + ", ".join(f"{board.san(m)} ({round((s + 1) / 2 * 100)} %)" for m, s in best))
            print(f"  Bot: {speak(board, move)}  [{board.san(move)}]")
            board.push(move)
            # lo que espera despues de su jugada (desde su punto de vista) para medir la sorpresa
            last_value = res.value

    show(board, flip=(human == chess.BLACK))
    out = board.outcome(claim_draw=True)
    if out.winner is None:
        print("Tablas.")
    else:
        print("¡Ganaste!" if out.winner == human else "Ganó el bot.")


if __name__ == "__main__":
    main()
