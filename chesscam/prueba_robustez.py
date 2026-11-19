"""
prueba_robustez.py - Mide la deteccion de jugadas de ChessCam con una camara simulada y los problemas de una
webcam real, sin camara ni ventana.

La imagen sale de render() de vision/prueba_simulada.py (tablero visto por una webcam, piezas altas, ruido y una
mano) y se le agregan, por separado o juntos:
  (a) exposicion: al entrar la mano la camara reajusta la exposicion (ganancia 0,85 a 1,15 y una curva no lineal
      que satura los blancos) y no vuelve exactamente al valor anterior;
  (b) franjas: franjas horizontales de brillo que se mueven despacio (parpadeo de luces LED a 50 Hz);
  (c) corrimiento: despues de cada jugada toda la imagen se corre 1 a 3 pixeles (el brazo de la camara);
  (d) quieto: 10 segundos sin que nadie juegue (en todos los escenarios, antes de la partida): no tiene que
      aparecer ninguna jugada ni aviso.
Ademas render() trae una variacion de brillo de +-3 % por cuadro: se mide con y sin ella.

Usa la logica real de la aplicacion (App.process y App.detect_move de chesscam.py) sobre un objeto sin ventana.
Si una jugada no se detecta en su ventana de tiempo, se carga "a mano" (como haria la persona con un clic) y
se sigue: asi cada jugada se mide por separado y un error no arrastra a las siguientes.

Uso, desde la carpeta del repo:
    python chesscam/prueba_robustez.py                    # vision nueva
    python chesscam/prueba_robustez.py --vision-original  # como era antes, para comparar
"""
import argparse
import contextlib
import io
import os
import sys
import types

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.append(RAIZ)

try:                                       # sin ventana: si no hay tkinter (algunos Linux), un reemplazo vacio
    import tkinter  # noqa: F401
except ImportError:
    tk = types.ModuleType("tkinter")
    for sub in ("ttk", "filedialog", "messagebox"):
        m = types.ModuleType("tkinter." + sub)
        setattr(tk, sub, m)
        sys.modules["tkinter." + sub] = m
    sys.modules["tkinter"] = tk
try:
    from PIL import ImageTk  # noqa: F401
except ImportError:
    import PIL
    PIL.ImageTk = types.ModuleType("PIL.ImageTk")
    sys.modules["PIL.ImageTk"] = PIL.ImageTk

import chess
import cv2
import numpy as np

import chesscam as cc
from board_vision import BoardVision
import vision.prueba_simulada as ps

PARTIDA = ("e4 e5 Nf3 Nc6 Bc4 Nf6 O-O Nxe4 d4 exd4 Re1 d5 Bxd5 Qxd5 Nc3 Qa5 Nxe4 Be6 Neg5 O-O-O Nxe6 fxe6 "
           "Rxe6").split()
FPS = 30
ROJO = "#f38ba8"


# =============================================================== camara simulada
class Ruido:
    """Reemplaza ps.rng para poder sacar la variacion de +-3 % de render() sin tocar el ruido de sensor."""

    def __init__(self, parpadeo, semilla):
        self.r = np.random.default_rng(semilla)
        self.parpadeo = parpadeo

    def uniform(self, lo, hi, *a):
        return self.r.uniform(lo, hi, *a) if self.parpadeo else 1.0

    def normal(self, *a, **k):
        return self.r.normal(*a, **k)


class Camara:
    def __init__(self, exposicion=False, franjas=False, corrimiento=False, parpadeo=False, semilla=0):
        self.exp, self.fr, self.corr = exposicion, franjas, corrimiento
        self.rng = np.random.default_rng(semilla + 100)
        ps.rng = Ruido(parpadeo, semilla)
        self.g, self.g_obj = 1.0, 1.0          # ganancia actual y la que busca la autoexposicion
        self.gam, self.gam_obj = 1.0, 1.0      # curva no lineal
        self.fase = 0.0
        self.dx = self.dy = 0.0
        self.mano_antes = False
        yy = np.arange(ps.H, dtype=np.float32)[:, None, None]
        self.yy = yy

    def cuadro(self, board, mano=None):
        hay_mano = mano is not None
        if self.exp:
            if hay_mano and not self.mano_antes:           # entra la mano: la camara reajusta
                self.g_obj = self.rng.uniform(0.85, 1.15)
                self.gam_obj = self.rng.uniform(0.8, 1.2)
            if not hay_mano and self.mano_antes:            # sale la mano: vuelve cerca, pero no igual
                self.g_obj = float(np.clip(self.g_obj * self.rng.uniform(0.92, 1.08), 0.85, 1.15))
            self.g += 0.25 * (self.g_obj - self.g)          # la autoexposicion cambia de a poco
            self.gam += 0.25 * (self.gam_obj - self.gam)
        if self.corr and not hay_mano and self.mano_antes:  # el brazo se corre al terminar la jugada
            ang, mag = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(1, 3)
            self.dx, self.dy = mag * np.cos(ang), mag * np.sin(ang)
        self.mano_antes = hay_mano
        img = ps.render(board, mano).astype(np.float32)
        if self.corr and (self.dx or self.dy):
            M = np.float32([[1, 0, self.dx], [0, 1, self.dy]])
            img = cv2.warpAffine(img, M, (ps.W, ps.H), borderMode=cv2.BORDER_REPLICATE)
        if self.fr:
            self.fase += 2 * np.pi / (3 * FPS)              # una franja pasa cada ~3 s
            img *= 1 + 0.06 * np.sin(2 * np.pi * self.yy / 90.0 + self.fase)
        if self.exp:
            img = 255.0 * self.g * (np.clip(img, 0, 255) / 255.0) ** self.gam   # >255 se satura (blancos)
        return np.clip(img, 0, 255).astype(np.uint8)


# =============================================================== aplicacion sin ventana
class AppSinVentana:
    """Los metodos de deteccion son los de cc.App; lo que dibuja no hace nada."""
    set_ref = cc.App.set_ref
    apply_move = cc.App.apply_move
    process = cc.App.process
    detect_move = cc.App.detect_move

    def __init__(self, vision_kwargs):
        self.partida = cc.Partida(con_camara=True)
        self.board = self.partida.board
        self.vision = BoardVision(**vision_kwargs)
        self.vision.set_corners(ps.CAM)
        self.quietudes = 0                                  # eventos "settled" (la imagen quedo quieta)
        actualizar = self.vision.update

        def contar(frame):
            warped, ev = actualizar(frame)
            self.quietudes += ev == "settled"
            return warped, ev
        self.vision.update = contar
        self.cap, self.warped, self.sync_pending = object(), None, True
        self.status = types.SimpleNamespace(configure=lambda **k: None)
        self.cartel, self.selected = True, None
        self.mensajes = []

    def set_status(self, txt, color="#9fd3ff"):
        self.mensajes.append((txt, color))

    def draw_board(self):
        pass

    status_fin = after_human = draw_board


def trayectoria(k, n):
    """Posicion de la mano en el cuadro k de n: entra por abajo, cruza el tablero y sale."""
    t = k / max(n - 1, 1)
    return int(450 + 450 * t), int(700 - 520 * np.sin(np.pi * t))


def escenario(nombre, vision_kwargs, semilla=0, mano_cuadros=15, espera_cuadros=50, **problemas):
    cam = Camara(semilla=semilla, **problemas)
    app = AppSinVentana(vision_kwargs)
    real = chess.Board()
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        for _ in range(3 * FPS):                            # arranque: tiene que tomar la referencia
            app.process(cam.cuadro(real))
        ref_ok = app.vision.reference is not None
        n_men = len(app.mensajes)
        jugadas_antes = len(app.board.move_stack)
        for _ in range(10 * FPS):                           # (d) 10 s quieto
            app.process(cam.cuadro(real))
        quieto_avisos = sum(c == ROJO for _, c in app.mensajes[n_men:])
        quieto_jugadas = len(app.board.move_stack) - jugadas_antes
        while len(app.board.move_stack) > jugadas_antes:   # si invento jugadas, se deshacen y se resincroniza
            app.board.pop()
        if quieto_jugadas:
            app.set_ref(app.warped)
        bien = sin_quietud = ignoradas = avisos = mal = 0
        latencias = []
        for san in PARTIDA:
            m = real.parse_san(san)
            n_men, q0 = len(app.mensajes), app.quietudes
            for k in range(mano_cuadros):
                if k == mano_cuadros // 2:
                    real.push(m)
                app.process(cam.cuadro(real, mano=trayectoria(k, mano_cuadros)))
            detectada, lat = None, None
            for k in range(espera_cuadros):
                antes = len(app.board.move_stack)
                app.process(cam.cuadro(real))
                if len(app.board.move_stack) > antes and detectada is None:
                    detectada, lat = app.board.peek(), k
            nuevos = app.mensajes[n_men:]
            avisos += sum(c == ROJO for _, c in nuevos)
            if not any("detectada" in t or c == ROJO for t, c in nuevos) and detectada is None:
                if app.quietudes == q0:
                    sin_quietud += 1                        # nunca dio la imagen por quieta
                else:
                    ignoradas += 1                          # quieta, pero "no cambio nada": sin jugada ni aviso
            if detectada == m:
                bien += 1
                latencias.append(lat)
                continue
            if detectada is not None:                       # detecto otra jugada: se deshace
                mal += 1
                app.board.pop()
            app.apply_move(m, manual=True)                  # la persona la carga a mano y se sigue
    lat_med = float(np.median(latencias)) / FPS if latencias else float("nan")
    return {"nombre": nombre, "ref": ref_ok, "bien": bien, "total": len(PARTIDA), "mal": mal, "avisos": avisos,
            "sin_quietud": sin_quietud, "ignoradas": ignoradas, "quieto_jugadas": quieto_jugadas, "quieto_avisos": quieto_avisos,
            "latencia": lat_med}


ESCENARIOS = [
    ("sin problemas (sin el +-3 %)", {}),
    ("solo el +-3 % de prueba_simulada", {"parpadeo": True}),
    ("(a) exposicion", {"exposicion": True}),
    ("(b) franjas 50 Hz", {"franjas": True}),
    ("(c) corrimiento 1-3 px", {"corrimiento": True}),
    ("todos juntos (a+b+c+-3 %)", {"exposicion": True, "franjas": True, "corrimiento": True, "parpadeo": True}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-original", action="store_true", help="medir el detector como era antes")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()
    if args.vision_original:
        kwargs = dict(getattr(cc, "VISION_ORIGINAL", {}))
    else:
        kwargs = {}
    version = "original" if args.vision_original or not hasattr(cc, "VISION_ORIGINAL") else "nueva"
    print(f"vision {version} | partida de {len(PARTIDA)} jugadas | {FPS} cuadros/s\n")
    print(f"{'escenario':34s} {'ref':>4s} {'bien':>7s} {'otra':>5s} {'avisos':>7s} {'ignoradas':>10s} "
          f"{'nunca quieta':>13s} {'10 s quieto':>12s} {'demora':>7s}")
    for nombre, prob in ESCENARIOS:
        r = escenario(nombre, kwargs, semilla=args.semilla, **prob)
        quieto = f"{r['quieto_jugadas']} j, {r['quieto_avisos']} a"
        print(f"{nombre:34s} {'si' if r['ref'] else 'NO':>4s} {r['bien']:>3d}/{r['total']:<3d} {r['mal']:>5d} "
              f"{r['avisos']:>7d} {r['ignoradas']:>10d} {r['sin_quietud']:>13d} {quieto:>12s} {r['latencia']:>6.2f}s",
              flush=True)
    print("\nbien: jugadas detectadas correctamente | otra: detecto una jugada equivocada | avisos: mensajes de"
          " 'no estoy seguro' | ignoradas: la imagen quedo quieta pero tomo la jugada por 'nada cambio' (sin aviso) |"
          " nunca quieta: jugadas en las que no dio la imagen por quieta |"
          " 10 s quieto: jugadas (j) y avisos (a) inventados sin que nadie juegue | demora: mediana desde que"
          " sale la mano")


if __name__ == "__main__":
    main()
