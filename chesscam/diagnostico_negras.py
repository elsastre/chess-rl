"""
diagnostico_negras.py - Mide, sobre capturas guardadas, qué tan bien se detectan las jugadas según el color de
la pieza y de la casilla, y compara variantes del filtro de BoardVision sin volver a la cámara.

Las capturas las guarda ChessCam con --capturas (una por detección: imagen de referencia, imagen después de la
jugada, posición y la jugada real, la detectada o la que se cargó a mano) o prueba_robustez.py con --capturas.
Las que terminan en _deshecha.npz (se deshizo la jugada) no se usan.

Para cada variante del filtro y cada captura se recalculan los cambios por casilla y se elige la jugada con
pick_move (solo con la pista de cambio: el "aspecto de casilla vacía" depende de la historia de la partida y
no está en la captura, así que el acierto puede ser algo menor que en la aplicación). Se informa:
  * acierto: la jugada elegida es la real; avisos: no se animó a elegir; mal: eligió otra;
  * margen: puntaje de la jugada real menos el de la mejor otra (más alto = más seguro);
  * señal por grupo: cambio normalizado (z) de las casillas de la jugada real, separando pieza negra / blanca y
    casilla oscura / clara. Si "negra en oscura" queda muy por debajo del resto, ese es el problema.

Uso, desde la carpeta del repo:
    python chesscam/diagnostico_negras.py chesscam/capturas/20261120-153000
    python chesscam/diagnostico_negras.py chesscam/capturas --variante mia:gamma_sombras=0.5,peso_textura=1
    python chesscam/diagnostico_negras.py chesscam/capturas --imagenes chesscam/capturas/fallos   # ver fallos
"""
import argparse
import glob
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import chess
import cv2
import numpy as np

from board_vision import BoardVision, pick_move, normalize_scores, SQ, FILTRO_NEGRAS
from chesscam import candidate_moves, move_squares

VARIANTES = {
    "actual": {},
    "gamma 0,6": {"gamma_sombras": 0.6},
    "textura 0,5": {"peso_textura": 0.5},
    "gradiente 0,5": {"peso_grad": 0.5},
    "refuerzo oscuras 0,5": {"refuerzo_oscuras": 0.5},
    "filtro-negras": dict(FILTRO_NEGRAS),
}
GRUPOS = ["negra/oscura", "negra/clara", "blanca/oscura", "blanca/clara"]


def leer_capturas(rutas):
    archivos = []
    for r in rutas:
        if os.path.isdir(r):
            archivos += glob.glob(os.path.join(r, "**", "*.npz"), recursive=True)
        else:
            archivos.append(r)
    archivos = sorted(a for a in archivos if not a.endswith("_deshecha.npz"))
    caps = []
    for a in archivos:
        d = np.load(a, allow_pickle=False)
        caps.append({"ruta": a, "ref": d["ref"], "img": d["img"], "fen": str(d["fen"]),
                     "jugada": chess.Move.from_uci(str(d["jugada"])), "fuente": str(d["fuente"])})
    return caps


def grupo(board, mv, sq):
    """Grupo de una casilla de la jugada: color de la pieza que se mueve y color de la casilla."""
    p = board.piece_at(mv.from_square)
    color = "blanca" if p.color == chess.WHITE else "negra"
    oscura = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0      # a1 es oscura
    return f"{color}/{'oscura' if oscura else 'clara'}"


def evaluar(caps, kwargs, imagenes=None):
    vis = BoardVision(**kwargs)
    res = {"bien": 0, "aviso": 0, "mal": 0, "margen": [], "z": {g: [] for g in GRUPOS},
           "canales": {g: {} for g in GRUPOS}, "fallos": []}
    for c in caps:
        board = chess.Board(c["fen"])
        mv = c["jugada"]
        vis._banda = {}
        f_ref, f_new = vis.features(c["ref"]), vis.features(c["img"])
        can = vis.square_diffs_canales(f_new, f_ref)
        raw = can["total"]
        elegida, _, ranked = pick_move(raw, candidate_moves(board))
        real = [s for s, m in ranked if (m.from_square, m.to_square) == (mv.from_square, mv.to_square)]
        otras = [s for s, m in ranked if (m.from_square, m.to_square) != (mv.from_square, mv.to_square)]
        if real:
            res["margen"].append(real[0] - (otras[0] if otras else 0.0))
        if elegida is not None and (elegida.from_square, elegida.to_square) == (mv.from_square, mv.to_square):
            res["bien"] += 1
        else:
            res["aviso" if elegida is None else "mal"] += 1
            res["fallos"].append((c, elegida, ranked[:3], raw))
        z = normalize_scores(raw)
        zc = {k: normalize_scores(v) for k, v in can.items() if k != "total"}
        for sq in move_squares(board, mv):
            g = grupo(board, mv, sq)
            res["z"][g].append(z[sq])
            for k, v in zc.items():
                res["canales"][g].setdefault(k, []).append(v[sq])
    if imagenes:
        guardar_fallos(res["fallos"], imagenes)
    return res


def guardar_fallos(fallos, carpeta):
    """Referencia | después | mapa de cambios, con las casillas de la jugada real marcadas en verde."""
    os.makedirs(carpeta, exist_ok=True)
    for c, elegida, top, raw in fallos:
        z = normalize_scores(raw)
        mapa = np.zeros_like(c["img"])
        for idx in range(64):
            f, r = idx % 8, idx // 8
            v = int(np.clip(z[idx] / 4.0, 0, 1) * 255)
            mapa[(7 - r) * SQ:(8 - r) * SQ, f * SQ:(f + 1) * SQ] = (0, v // 2, v)
        board = chess.Board(c["fen"])
        partes = [c["ref"].copy(), c["img"].copy(), mapa]
        for p in partes:
            for sq in move_squares(board, c["jugada"]):
                f, r = chess.square_file(sq), chess.square_rank(sq)
                cv2.rectangle(p, (f * SQ + 2, (7 - r) * SQ + 2), ((f + 1) * SQ - 3, (8 - r) * SQ - 3), (0, 220, 0), 2)
        lienzo = np.hstack(partes)
        txt = f"real {board.san(c['jugada'])} | elegida {board.san(elegida) if elegida else '-'} | " + \
              ", ".join(f"{board.san(m)}={s:.1f}" for s, m in top)
        cv2.putText(lienzo, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        nombre = os.path.splitext(os.path.basename(c["ruta"]))[0]
        cv2.imwrite(os.path.join(carpeta, nombre + ".png"), lienzo)


def leer_variante(texto):
    """'mia:gamma_sombras=0.5,peso_textura=1' -> ('mia', {...})"""
    nombre, _, pares = texto.partition(":")
    kw = {}
    for par in filter(None, pares.split(",")):
        k, v = par.split("=")
        kw[k.strip()] = float(v)
    return nombre, kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rutas", nargs="+", help="carpetas o archivos .npz de capturas")
    ap.add_argument("--variante", action="append", default=[], metavar="NOMBRE:PARAM=VALOR,...",
                    help="variante extra del filtro (se puede repetir)")
    ap.add_argument("--solo-extra", action="store_true", help="comparar solo 'actual' y las variantes extra")
    ap.add_argument("--imagenes", default=None, help="carpeta donde guardar imágenes de los fallos de 'actual'")
    args = ap.parse_args()

    caps = leer_capturas(args.rutas)
    if not caps:
        print("No hay capturas. Guardalas con: python chesscam/chesscam.py --capturas (o prueba_robustez.py "
              "--capturas CARPETA)")
        return
    n_negras = sum(chess.Board(c["fen"]).turn == chess.BLACK for c in caps)
    print(f"{len(caps)} capturas ({n_negras} jugadas de negras, "
          f"{sum(c['fuente'] == 'manual' for c in caps)} cargadas a mano porque la cámara no las vio)\n")

    variantes = {"actual": {}} if args.solo_extra else dict(VARIANTES)
    for v in args.variante:
        nombre, kw = leer_variante(v)
        variantes[nombre] = kw

    print(f"{'variante':22s} {'bien':>8s} {'avisos':>7s} {'mal':>4s} {'margen med.':>12s}   "
          + "  ".join(f"{g:>13s}" for g in GRUPOS))
    print(f"{'':22s} {'':>8s} {'':>7s} {'':>4s} {'':>12s}   " + "  ".join(f"{'z mediana':>13s}" for _ in GRUPOS))
    detalle = None
    for nombre, kw in variantes.items():
        r = evaluar(caps, kw, imagenes=args.imagenes if nombre == "actual" else None)
        if nombre == "actual":
            detalle = r
        zs = "  ".join(f"{np.median(r['z'][g]):>8.2f} ({len(r['z'][g]):>3d})" if r["z"][g] else f"{'-':>13s}"
                       for g in GRUPOS)
        margen = np.median(r["margen"]) if r["margen"] else float("nan")
        print(f"{nombre:22s} {r['bien']:>4d}/{len(caps):<3d} {r['aviso']:>7d} {r['mal']:>4d} {margen:>12.2f}   {zs}",
              flush=True)

    print("\nSeñal por canal con el filtro actual (z mediana de las casillas de la jugada real):")
    canales = sorted({k for g in GRUPOS for k in detalle["canales"][g]})
    print(f"{'grupo':15s} " + " ".join(f"{k:>7s}" for k in canales))
    for g in GRUPOS:
        fila = [np.median(detalle["canales"][g][k]) if detalle["canales"][g].get(k) else float("nan")
                for k in canales]
        print(f"{g:15s} " + " ".join(f"{v:>7.2f}" for v in fila))
    print("\nL = brillo (con CLAHE), C = color (a+b de LAB), G = gradiente, T = textura (si la variante la usa).")
    print("z: cambio de la casilla en 'desvíos' sobre el ruido del resto del tablero (por debajo de ~1 no se nota).")
    if args.imagenes:
        print(f"Imágenes de los fallos en {args.imagenes}")


if __name__ == "__main__":
    main()
