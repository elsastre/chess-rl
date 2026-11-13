"""
Codificacion del tablero para la red.

Convencion clave: el tablero se ve SIEMPRE desde el bando que mueve.
Si mueven las negras, se espeja (colores y filas invertidos), asi la red
aprende una sola cosa: "que tan buena es esta posicion para el que mueve".
El valor que devuelve la red esta en [-1, 1] desde ese punto de vista.

Salida: 12 planos de 8x8 (uint8).
  planos 0-5 : mis piezas      (peon, caballo, alfil, torre, dama, rey)
  planos 6-11: piezas del rival (mismo orden)
"""
import numpy as np
import chess

N_PLANES = 12
_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _mask_to_plane(mask: int) -> np.ndarray:
    # bit i = casilla i (a1=0, b1=1, ..., h8=63) -> matriz 8x8 con fila 0 = fila 1 del tablero
    bits = np.unpackbits(np.array([mask], dtype="<u8").view(np.uint8), bitorder="little")
    return bits.reshape(8, 8)


def encode(board: chess.Board) -> np.ndarray:
    b = board if board.turn == chess.WHITE else board.mirror()
    planes = np.empty((N_PLANES, 8, 8), dtype=np.uint8)
    for i, pt in enumerate(_PIECES):
        planes[i] = _mask_to_plane(b.pieces_mask(pt, chess.WHITE))
        planes[i + 6] = _mask_to_plane(b.pieces_mask(pt, chess.BLACK))
    return planes


def pack(planes: np.ndarray) -> np.ndarray:
    """12x8x8 bits -> 96 bytes (para guardar datasets grandes sin ocupar RAM)."""
    return np.packbits(planes.reshape(-1))


def unpack(packed: np.ndarray) -> np.ndarray:
    """Inverso de pack. Acepta (96,) o (N, 96)."""
    packed = np.asarray(packed)
    if packed.ndim == 1:
        return np.unpackbits(packed)[: N_PLANES * 64].reshape(N_PLANES, 8, 8)
    bits = np.unpackbits(packed, axis=1)[:, : N_PLANES * 64]
    return bits.reshape(-1, N_PLANES, 8, 8)


def terminal_value(board: chess.Board):
    """Valor exacto para el bando que mueve si la partida termino; None si sigue."""
    outcome = board.outcome()  # incluye mate, ahogado, material insuficiente, 75 jugadas, 5 repeticiones
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0
