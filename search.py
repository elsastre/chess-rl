"""
Busqueda v2: arbol a profundidad fija + busqueda de capturas (quiescence) + red en batch.

Como funciona, para una posicion y profundidad D:
  1. Se expande el arbol completo hasta D jugadas (python-chess, sin red).
  2. En cada hoja se resuelven los cambios de piezas con una quiescence que solo mira
     material (capturas y coronaciones). Asi la red nunca evalua una posicion "en medio
     de un cambio": evalua la posicion tranquila a la que se llega (truco de Texel).
  3. Todas esas posiciones tranquilas se evaluan con la red en UN solo batch (GPU).
  4. Se hace minimax hacia arriba. Mates y tablas usan el valor exacto.

Ademas de elegir la jugada, la busqueda devuelve lo que necesitan los algoritmos de RL:
  - TreeStrap: cada nodo del arbol con su valor de busqueda (objetivo de entrenamiento).
  - TD-Leaf:   la hoja de la variante principal, con su signo respecto de la raiz.

Convencion de valores: siempre desde el punto de vista del bando que mueve en ese nodo,
en [-1, 1] (igual que la red, ver encoding.py).
"""
import random
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from encoding import encode, terminal_value

VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
MATE = 100_000
INF = 10 ** 9


# ---------------------------------------------------------------- quiescence por material
def material(board: chess.Board) -> int:
    """Material en centipeones desde el bando que mueve."""
    me, op = board.occupied_co[board.turn], board.occupied_co[not board.turn]
    s = 0
    for bb, v in ((board.pawns, 100), (board.knights, 320), (board.bishops, 330), (board.rooks, 500), (board.queens, 900)):
        s += v * ((bb & me).bit_count() - (bb & op).bit_count())
    return s


def _gain(board, m):
    """Material que gana el que mueve con la jugada m (captura y/o coronacion)."""
    victim = board.piece_type_at(m.to_square)
    if victim is None and board.is_en_passant(m):
        victim = chess.PAWN
    g = VAL[victim] if victim else 0
    if m.promotion:
        g += VAL[m.promotion] - VAL[chess.PAWN]
    return g


def _order(board, moves):
    return sorted(moves, key=lambda m: _gain(board, m) * 10 - VAL[board.piece_type_at(m.from_square)], reverse=True)


DELTA = 200   # poda delta: no mirar capturas que ni ganando la pieza alcanzan alpha


def qsearch(board: chess.Board, alpha: int, beta: int, depth: int, mat=None):
    """Devuelve (puntaje material para el que mueve, jugadas de la variante principal).
    mat: material actual para el que mueve (se pasa incremental para no recalcularlo)."""
    if mat is None:
        mat = material(board)
    if board.is_check():
        moves = list(board.legal_moves)
        if not moves:
            return -MATE, []
        if depth <= 0:
            return mat, []
        best, pv, cand = -INF, [], _order(board, moves)
        stand = None
    else:
        stand = mat
        if depth <= 0 or stand >= beta:
            return stand, []
        best, pv = stand, []
        alpha = max(alpha, stand)
        cand = _order(board, list(board.generate_legal_captures()))
    for m in cand:
        g = _gain(board, m)
        if stand is not None and stand + g + DELTA < alpha:
            continue
        board.push(m)
        sc, sub = qsearch(board, -beta, -alpha, depth - 1, -(mat + g))
        board.pop()
        sc = -sc
        if sc > best:
            best, pv = sc, [m] + sub
        if sc > alpha:
            alpha = sc
        if alpha >= beta:
            break
    return best, pv


# ---------------------------------------------------------------- arbol con la red
@dataclass
class SearchResult:
    value: float                         # valor de la raiz para el que mueve
    moves: list                          # [(jugada, puntaje para el que mueve en la raiz)]
    pv_leaf: tuple                       # (planos, signo respecto de la raiz) o None si es exacta
    treestrap: list = field(default_factory=list)   # [(planos, objetivo)]
    n_leaves: int = 0


def make_evaluator(model, device, chunk=4096):
    @torch.no_grad()
    def evaluate(planes_list):
        if not planes_list:
            return np.zeros(0, dtype=np.float32)
        out = []
        for i in range(0, len(planes_list), chunk):
            x = torch.from_numpy(np.stack(planes_list[i:i + chunk])).float().to(device)
            out.append(model(x).float().cpu().numpy())
        return np.concatenate(out)
    return evaluate


def search(board: chess.Board, evaluate, depth=2, qdepth=4, collect=False, max_frontier_samples=32):
    leaves = []

    def build(d):
        exact = terminal_value(board)
        if exact is not None:
            return {"exact": exact}
        if d == 0:
            pv = qsearch(board, -INF, INF, qdepth)[1] if qdepth > 0 else []
            k, sign = len(pv), (1.0 if len(pv) % 2 == 0 else -1.0)
            for m in pv:
                board.push(m)
            ex = terminal_value(board) if k else None
            if ex is None:
                node = {"leaf": (len(leaves), sign)}
                leaves.append(encode(board))
            else:
                node = {"exact": sign * ex}
            for _ in pv:
                board.pop()
            if collect and k > 0 and "leaf" in node:   # hoja con cambios: aprende algo nuevo
                node["planes"] = encode(board)
            return node
        node = {"children": []}
        if collect:
            node["planes"] = encode(board)
        for m in board.legal_moves:
            board.push(m)
            child = build(d - 1)
            board.pop()
            child["move"] = m
            node["children"].append(child)
        return node

    root = build(depth)
    vals = evaluate(leaves)

    def backup(n):
        if "exact" in n:
            n["v"], n["pv"] = n["exact"], None
            return
        if "leaf" in n:
            idx, sign = n["leaf"]
            n["v"], n["pv"] = sign * float(vals[idx]), (idx, sign)
            return
        best, best_c = -INF, None
        for c in n["children"]:
            backup(c)
            if -c["v"] > best:
                best, best_c = -c["v"], c
        n["v"] = best
        n["pv"] = None if best_c["pv"] is None else (best_c["pv"][0], -best_c["pv"][1])

    backup(root)
    if "children" not in root:                      # la raiz ya es final o se pidio profundidad 0
        moves = []
    else:
        moves = [(c["move"], -c["v"]) for c in root["children"]]

    pv_leaf = None if root["pv"] is None else (leaves[root["pv"][0]], root["pv"][1])
    res = SearchResult(value=root["v"], moves=moves, pv_leaf=pv_leaf, n_leaves=len(leaves))

    if collect:
        interior, frontier = [], []

        def gather(n, is_root=True):
            if "planes" in n:
                (frontier if "leaf" in n else interior).append((n["planes"], float(np.clip(n["v"], -1, 1))))
            for c in n.get("children", []):
                gather(c, False)
        gather(root)
        if len(frontier) > max_frontier_samples:
            frontier = random.sample(frontier, max_frontier_samples)
        res.treestrap = interior + frontier
    return res


def pick_move(moves, temperature=0.0):
    scores = np.array([s for _, s in moves])
    if temperature <= 0:
        best = scores.max()
        return random.choice([m for (m, s) in moves if s >= best - 1e-9])
    p = np.exp((scores - scores.max()) / temperature)
    p /= p.sum()
    return moves[np.random.choice(len(moves), p=p)][0]


class SearchPlayer:
    """Jugador para la arena / el robot con la busqueda nueva."""

    def __init__(self, model, device="cpu", depth=2, qdepth=4, name="red"):
        self.evaluate = make_evaluator(model, device)
        self.depth, self.qdepth = depth, qdepth
        self.name = f"{name}-d{depth}q{qdepth}"

    def choose(self, board):
        res = search(board, self.evaluate, self.depth, self.qdepth)
        return pick_move(res.moves, 0.0)

    def close(self):
        pass
