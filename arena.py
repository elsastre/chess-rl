"""
Etapa 0: arena de evaluacion para el bot de ajedrez.

Define rivales de referencia, juega matches y estima diferencias de Elo.
Todo agente futuro (el de RL) solo tiene que implementar choose(board) -> move.

Uso:
    python arena.py                         # escalera sin Stockfish
    python arena.py --stockfish RUTA.exe    # agrega Stockfish limitado como ancla
    python arena.py --games 40              # mas partidas por match (menos ruido)
"""
import argparse
import math
import random
import chess
import chess.engine

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


# ---------------------------------------------------------------- jugadores
class Player:
    name = "base"

    def choose(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError

    def close(self):
        pass


class RandomPlayer(Player):
    name = "aleatorio"

    def choose(self, board):
        return random.choice(list(board.legal_moves))


def material(board: chess.Board, color: bool) -> int:
    """Material de `color` menos material del rival."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, color))
        total -= value * len(board.pieces(piece_type, not color))
    return total


def evaluate(board: chess.Board, color: bool) -> float:
    """Evaluacion simple desde el punto de vista de `color`."""
    if board.is_checkmate():
        # le toca mover al que recibio mate
        return -1000 if board.turn == color else 1000
    if board.is_game_over():
        return 0
    return material(board, color)


class GreedyPlayer(Player):
    """Mira 1 jugada: elige la que deja mas material (desempata al azar)."""
    name = "codicioso"

    def choose(self, board):
        me = board.turn
        best, best_moves = -math.inf, []
        for move in board.legal_moves:
            board.push(move)
            score = evaluate(board, me)
            board.pop()
            if score > best:
                best, best_moves = score, [move]
            elif score == best:
                best_moves.append(move)
        return random.choice(best_moves)


class MinimaxPlayer(Player):
    """Alfa-beta a profundidad fija con evaluacion de material."""

    def __init__(self, depth=2):
        self.depth = depth
        self.name = f"minimax{depth}"

    def _search(self, board, depth, alpha, beta, me):
        if depth == 0 or board.is_game_over():
            return evaluate(board, me)
        maximizing = board.turn == me
        best = -math.inf if maximizing else math.inf
        for move in board.legal_moves:
            board.push(move)
            score = self._search(board, depth - 1, alpha, beta, me)
            board.pop()
            if maximizing:
                best = max(best, score)
                alpha = max(alpha, score)
            else:
                best = min(best, score)
                beta = min(beta, score)
            if beta <= alpha:
                break
        return best

    def choose(self, board):
        me = board.turn
        moves = list(board.legal_moves)
        random.shuffle(moves)  # variedad entre partidas
        best, best_move = -math.inf, moves[0]
        for move in moves:
            board.push(move)
            score = self._search(board, self.depth - 1, -math.inf, math.inf, me)
            board.pop()
            if score > best:
                best, best_move = score, move
        return best_move


class StockfishPlayer(Player):
    """Stockfish con fuerza limitada. Sirve como ancla de Elo conocida."""

    def __init__(self, path, elo=1320, think_time=0.05):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        opts = self.engine.options
        lo = opts["UCI_Elo"].min if "UCI_Elo" in opts else elo
        self.elo = max(elo, lo)  # Stockfish no acepta valores bajo su minimo
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
        self.limit = chess.engine.Limit(time=think_time)
        self.name = f"stockfish{self.elo}"

    def choose(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


# ---------------------------------------------------------------- partidas
def play_game(white: Player, black: Player, max_plies=300) -> float:
    """Devuelve el puntaje de las blancas: 1, 0.5 o 0."""
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        player = white if board.turn == chess.WHITE else black
        board.push(player.choose(board))
    result = board.result(claim_draw=True)
    return {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)


def match(a: Player, b: Player, games=20) -> float:
    """Puntaje promedio de `a` contra `b`, alternando colores."""
    score = 0.0
    for g in range(games):
        if g % 2 == 0:
            score += play_game(a, b)
        else:
            score += 1 - play_game(b, a)
    return score / games


def elo_diff(score: float, games: int) -> float:
    """Diferencia de Elo implicada por un puntaje (recortado para evitar infinitos)."""
    eps = 0.5 / games
    s = min(max(score, eps), 1 - eps)
    return 400 * math.log10(s / (1 - s))


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--stockfish", type=str, default=None)
    args = ap.parse_args()

    ladder = [RandomPlayer(), GreedyPlayer(), MinimaxPlayer(2)]
    if args.stockfish:
        ladder.append(StockfishPlayer(args.stockfish))

    print(f"{args.games} partidas por match\n")
    for weaker, stronger in zip(ladder, ladder[1:]):
        s = match(stronger, weaker, args.games)
        print(f"{stronger.name:>14} vs {weaker.name:<12} "
              f"puntaje {s:.2f}  ->  {elo_diff(s, args.games):+.0f} Elo")

    for p in ladder:
        p.close()
