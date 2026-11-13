"""
Agente que juega con la red de valor + busqueda negamax de profundidad fija.

search() devuelve, para cada jugada legal, su puntaje desde el punto de vista
del bando que mueve en la raiz, y la "hoja" de la variante principal:
(planos_de_la_hoja, signo). El valor de la raiz segun la red es
signo * red(hoja). Eso es exactamente lo que TD-Leaf necesita para entrenar.

Las hojas del ultimo nivel se evaluan en un solo batch en GPU.
Si una jugada termina la partida (mate, ahogado...), se usa el valor exacto.
"""
import random
import numpy as np
import torch
import chess

from encoding import encode, terminal_value
from arena import Player


class NetSearcher:
    def __init__(self, model, device="cpu", depth=1):
        self.model = model
        self.device = device
        self.depth = depth

    @torch.no_grad()
    def _eval(self, planes_list):
        x = torch.from_numpy(np.stack(planes_list)).float().to(self.device)
        return self.model(x).cpu().numpy()

    def search(self, board: chess.Board, depth=None):
        """Lista de (jugada, puntaje, hoja). hoja = (planos, signo) o None si es exacta."""
        depth = depth or self.depth
        moves, scores, leaves = [], [], []
        pending_planes, pending_idx = [], []

        for move in board.legal_moves:
            board.push(move)
            exact = terminal_value(board)
            if exact is not None:
                # exact es para el que mueve en el hijo; la raiz ve lo opuesto
                moves.append(move); scores.append(-exact); leaves.append(None)
            elif depth == 1:
                moves.append(move); scores.append(None); leaves.append(None)
                pending_planes.append(encode(board)); pending_idx.append(len(moves) - 1)
            else:
                sub = self.search(board, depth - 1)
                _, best_score, best_leaf = max(sub, key=lambda t: t[1])
                leaf = None if best_leaf is None else (best_leaf[0], -best_leaf[1])
                moves.append(move); scores.append(-best_score); leaves.append(leaf)
            board.pop()

        if pending_planes:
            values = self._eval(pending_planes)
            for v, i, p in zip(values, pending_idx, pending_planes):
                scores[i] = -float(v)   # la red valora para el rival (el que mueve en el hijo)
                leaves[i] = (p, -1.0)   # valor_raiz = -1 * red(hoja)
        return list(zip(moves, scores, leaves))


def pick(results, temperature=0.0):
    """Elige jugada: la mejor (temperature=0) o muestreo softmax para explorar."""
    scores = np.array([s for _, s, _ in results])
    if temperature <= 0:
        best = scores.max()
        idx = random.choice(np.flatnonzero(scores >= best - 1e-9))
    else:
        p = np.exp((scores - scores.max()) / temperature)
        p /= p.sum()
        idx = np.random.choice(len(results), p=p)
    return results[idx][0]


class NetPlayer(Player):
    """Envoltorio para usar el agente en la arena (misma interfaz que los rivales)."""

    def __init__(self, model, device="cpu", depth=1, name="red"):
        self.searcher = NetSearcher(model, device, depth)
        self.name = f"{name}-d{depth}"

    def choose(self, board):
        return pick(self.searcher.search(board), temperature=0.0)
