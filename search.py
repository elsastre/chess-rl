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


def mate_en_1(board: chess.Board) -> bool:
    """True si el que mueve tiene un mate en 1.

    Solo se prueban (jugando la jugada) las que pueden dar jaque. Para saberlo sin jugarlas se calculan
    las casillas desde las que cada tipo de pieza atacaria al rey rival. Los casos raros (jaque a la
    descubierta, coronacion, al paso, enroque) se dejan pasar al chequeo completo."""
    ksq = board.king(not board.turn)
    if ksq is None:
        return False
    occ = board.occupied
    diag = chess.BB_DIAG_ATTACKS[ksq][chess.BB_DIAG_MASKS[ksq] & occ]
    line = (chess.BB_RANK_ATTACKS[ksq][chess.BB_RANK_MASKS[ksq] & occ]
            | chess.BB_FILE_ATTACKS[ksq][chess.BB_FILE_MASKS[ksq] & occ])
    desde = {chess.PAWN: chess.BB_PAWN_ATTACKS[not board.turn][ksq], chess.KNIGHT: chess.BB_KNIGHT_ATTACKS[ksq],
             chess.BISHOP: diag, chess.ROOK: line, chess.QUEEN: diag | line, chess.KING: 0}
    rayos = chess.BB_RAYS[ksq]
    for m in board.legal_moves:
        pt = board.piece_type_at(m.from_square)
        if not (chess.BB_SQUARES[m.to_square] & desde[pt]):
            raro = (rayos[m.from_square] or m.promotion or pt == chess.KING
                    or (pt == chess.PAWN and board.is_en_passant(m)))
            if not raro:
                continue
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


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


def search(board: chess.Board, evaluate, depth=2, qdepth=4, collect=False, max_frontier_samples=32, mates=False,
           poda=False):
    """mates=True: en cada hoja se mira si el que mueve tiene mate en 1 (valor exacto +1). Sin esto, la
    busqueda de profundidad 1 no ve un mate del rival que no sea captura (la quiescencia solo mira capturas)."""
    if poda and depth == 2 and terminal_value(board) is None:
        return _search_ab2(board, evaluate, qdepth, collect, max_frontier_samples, mates)
    leaves = []

    def build(d):
        exact = terminal_value(board)
        if exact is not None:
            return {"exact": exact}
        if d == 0:
            if mates and mate_en_1(board):
                return {"exact": 1.0}
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


# ---------------------------------------------------------------- profundidad 2 con poda alfa-beta
def _hoja(board, qdepth, mates, collect):
    """Una hoja del arbol: resuelve los cambios y devuelve (exacto, planos tranquilos, signo, planos previos).
    exacto: valor para el que mueve si no hace falta la red (si no, None). signo: +1 si en la posicion
    tranquila mueve el mismo bando que en la hoja. planos previos: solo si hubo cambios (muestra de TreeStrap)."""
    if mates and mate_en_1(board):
        return 1.0, None, 1.0, None
    pv = qsearch(board, -INF, INF, qdepth)[1] if qdepth > 0 else []
    sign = 1.0 if len(pv) % 2 == 0 else -1.0
    for m in pv:
        board.push(m)
    ex = terminal_value(board) if pv else None
    quiet = encode(board) if ex is None else None
    for _ in pv:
        board.pop()
    if ex is not None:
        return sign * ex, None, sign, None
    return None, quiet, sign, (encode(board) if collect and pv else None)


def _search_ab2(board, evaluate, qdepth, collect, max_frontier_samples, mates, chunk=4):
    """Busqueda de profundidad 2 con poda alfa-beta. Da el mismo valor de la raiz que el arbol completo,
    pero mira muchas menos posiciones: cuando una respuesta del rival ya deja a una jugada peor que la
    mejor encontrada, esa jugada queda refutada y no se miran sus otras respuestas.

    Para seguir evaluando con la red en lote se trabaja en tres pasos:
      1. La mejor candidata (segun una busqueda de profundidad 1) se mira entera: da el valor a superar.
      2. De todas las demas se miran las primeras `chunk` respuestas, juntas en un solo lote.
         La mayoria queda refutada ahi.
      3. Las que sobreviven se siguen mirando de a tandas hasta refutarlas o completarlas."""
    orden = search(board, evaluate, 1, qdepth, mates=mates)
    frontier, n_leaves = [], orden.n_leaves
    terminales, vivos = [], []
    for m, _ in sorted(orden.moves, key=lambda t: -t[1]):
        board.push(m)
        ex = terminal_value(board)
        if ex is not None:                                # la jugada termina la partida
            terminales.append({"m": m, "score": -ex, "pv": None, "completo": True})
        else:
            vivos.append({"m": m, "replies": list(board.legal_moves), "i": 0, "score": INF, "pv": None,
                          "mejor": None, "completo": False})
        board.pop()

    def ordenar(st, killers):
        board.push(st["m"])                               # refutaciones probables primero
        st["replies"].sort(key=lambda r: (r not in killers, -_gain(board, r)))
        board.pop()

    def avanzar(estados, cuantas):
        """Mira las siguientes `cuantas` respuestas (None = todas) de cada jugada, en un solo lote de la red."""
        nonlocal n_leaves
        trabajo, planos = [], []
        for st in estados:
            tanda = st["replies"][st["i"]:] if cuantas is None else st["replies"][st["i"]:st["i"] + cuantas]
            st["i"] += len(tanda)
            board.push(st["m"])
            for r in tanda:
                board.push(r)
                ex_r = terminal_value(board)
                if ex_r is not None:
                    trabajo.append((st, r, ex_r, None, 1.0, None))
                else:
                    e, quiet, sign, prev = _hoja(board, qdepth, mates, collect)
                    trabajo.append((st, r, e, quiet, sign, prev))
                    if e is None:
                        planos.append(quiet)
                board.pop()
            board.pop()
        vals, k = evaluate(planos), 0
        n_leaves += len(planos)
        for st, r, e, quiet, sign, prev in trabajo:
            if e is None:
                v = sign * float(vals[k]); k += 1          # valor para el que mueve en la hoja (el bando de la raiz)
                if prev is not None:
                    frontier.append((prev, float(np.clip(v, -1, 1))))
                leaf = (quiet, sign)
            else:
                v, leaf = e, None
            if v < st["score"]:
                st["score"], st["pv"], st["mejor"] = v, leaf, r
        for st in estados:
            st["completo"] = st["i"] >= len(st["replies"])

    alpha = max((t["score"] for t in terminales), default=-INF)
    if vivos:
        primera, resto = vivos[0], vivos[1:]
        ordenar(primera, ())
        avanzar([primera], None)                           # paso 1
        alpha = max(alpha, primera["score"])
        killers = (primera["mejor"],)
        for st in resto:
            ordenar(st, killers)
        if resto:
            avanzar(resto, chunk)                          # paso 2
        for st in resto:
            if st["completo"]:
                alpha = max(alpha, st["score"])
        for st in resto:                                   # paso 3
            while not st["completo"] and st["score"] > alpha:
                avanzar([st], 2 * chunk)
            if st["completo"]:
                alpha = max(alpha, st["score"])

    todas = terminales + vivos
    mejor = max((t for t in todas if t["completo"]), key=lambda t: t["score"])
    res = SearchResult(value=mejor["score"], moves=[(t["m"], t["score"]) for t in todas], pv_leaf=mejor["pv"],
                       n_leaves=n_leaves)
    if collect:
        interior = []
        for st in vivos:
            if st["completo"]:                             # valor exacto: sirve como muestra de TreeStrap
                board.push(st["m"])
                interior.append((encode(board), float(np.clip(-st["score"], -1, 1))))
                board.pop()
        if len(frontier) > max_frontier_samples:
            frontier = random.sample(frontier, max_frontier_samples)
        res.treestrap = [(encode(board), float(np.clip(mejor["score"], -1, 1)))] + interior + frontier
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

    def __init__(self, model, device="cpu", depth=2, qdepth=4, name="red", mates=False, poda=False):
        self.evaluate = make_evaluator(model, device)
        self.depth, self.qdepth, self.mates, self.poda = depth, qdepth, mates, poda
        self.name = f"{name}-d{depth}q{qdepth}" + ("m" if mates else "")

    def choose(self, board):
        res = search(board, self.evaluate, self.depth, self.qdepth, mates=self.mates, poda=self.poda)
        return pick_move(res.moves, 0.0)

    def close(self):
        pass
