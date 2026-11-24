"""
chesscam.py - Seguimiento de una partida de ajedrez real con webcam, y partida contra el bot del proyecto.

Uso (desde la carpeta del repo):
    python chesscam/chesscam.py                 # cámara 0, dos personas
    python chesscam/chesscam.py --cam 1         # otra cámara
    python chesscam/chesscam.py --board 640     # tamaño del tablero en pantalla
    python chesscam/chesscam.py --ck runs/final/ck_150000.pt              # contra el bot (vos con blancas)
    python chesscam/chesscam.py --ck runs/final/ck_150000.pt --bot blancas
    python chesscam/chesscam.py --ck runs/final/ck_150000.pt --sin-camara # solo el simulador, sin tablero fisico
    python chesscam/chesscam.py --cam 1 --fijar-camara                    # fija exposición, foco y balance
    python chesscam/chesscam.py --cam 1 --vision-original                 # el detector de antes del 19/11
    python chesscam/chesscam.py --cam 1 --auto                            # detecta sola al quedar quieta
    python chesscam/chesscam.py --cam 1 --capturas                        # guarda cada detección (diagnóstico)
    python chesscam/chesscam.py --cam 1 --filtro-negras                   # filtro para piezas negras

Detección manual (20/11, por defecto): después de jugar se pulsa «Detectar jugada» (o la barra espaciadora).
Se juntan 15 cuadros quietos, se combinan con la mediana por píxel y se busca la jugada una sola vez. Una mano
que pasa o la cámara reajustándose ya no disparan nada. Con --auto vuelve el comportamiento anterior.

Robustez (19/11): la referencia y la imagen que se compara son el promedio de 8 cuadros quietos, se esperan
2 s antes de la primera referencia, el detector de movimiento ignora el brillo general y la diferencia
tolera corrimientos de 2 px. Con --fijar-camara, después de 30 cuadros en automático se apagan el autofoco,
el balance de blancos y la exposición automáticos (si el driver lo permite; en Mac no se puede).

Contra el bot:
  * Con cámara: el bot NO mueve solo. Su jugada aparece como flecha (y descrita en el panel) para que la
    hagas en el tablero real; se aplica cuando la cámara la detecta. Si la cámara no la ve, «Bot: hecha».
  * Sin cámara: jugás haciendo clic en el tablero de la pantalla; el bot responde y su flecha queda marcada.
  * Al elegir una pieza se marcan sus jugadas legales (puntos; anillos si come). «Pista» sugiere una jugada.

Flujo:
  1. Coloca las piezas en la posición inicial.
  2. Pulsa "Calibrar" y haz clic en la vista de cámara en las 4 esquinas
     exteriores de las casillas, en este orden: a1, h1, h8, a8
     (a1 = esquina izquierda del lado de las blancas).
     La calibración se guarda en calib.json y se reutiliza.
  3. Juega. Tras cada jugada, retira la mano y pulsa «Detectar jugada» (o espacio): con la imagen
     quieta se deduce la jugada legal que explica los cambios (con --auto, sin pulsar nada).
  4. Si algo falla: haz clic en el tablero de la pantalla (origen → destino)
     para meter la jugada a mano, o usa Deshacer / Sincronizar.
"""
import argparse
import json
import os
import sys
import time

# Evita que Media Foundation (Windows) tarde muchísimo o se cuelgue al abrir algunas webcams
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import chess
import chess.pgn
import cv2
import numpy as np
from PIL import Image, ImageTk

from board_vision import BoardVision, pick_move, normalize_scores, WARP, SQ, VISION_ORIGINAL, FILTRO_NEGRAS
from partida import Partida, Bot, Reloj, san_es, formato_tiempo, leer_control, resumen_porque

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")
CORNER_NAMES = ["a1", "h1", "h8", "a8"]

LIGHT_SQ, DARK_SQ = "#EED8B5", "#B38763"
HL_LAST, HL_SEL = "#F5E663", "#7FC97F"
FLECHA_BOT, FLECHA_PISTA = "#15781B", "#2F6FD6"     # verde (jugada del bot) y azul (pista), como en Lichess
PUNTO, ANILLO_COME = "#5C6E4A", "#B03A2E"
COLOR_FIN = {"gane": "#a6e3a1", "perdi": "#f38ba8", "tablas": "#f9e2af", "fin": "#9fd3ff"}
def tinte_porque(base, v, lim):
    """Mezcla el color de la casilla con azul (la pieza ayuda a su dueño) o rojo (lo perjudica), más fuerte
    cuanto más pesa. La raíz cuadrada hace visibles también los valores chicos."""
    t = min(1.0, (abs(v) / lim) ** .5) * .8
    meta = (47, 111, 214) if v >= 0 else (214, 64, 64)
    r, g, b = (int(base[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(c0 + (c1 - c0) * t) for c0, c1 in zip((r, g, b), meta))
    return f"#{r:02x}{g:02x}{b:02x}"


GLYPH = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
GLYPH_OUT = {"k": "♔", "q": "♕", "r": "♖", "b": "♗", "n": "♘", "p": "♙"}


BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avf": cv2.CAP_AVFOUNDATION,
    "any": cv2.CAP_ANY,
}


# Valor de CAP_PROP_AUTO_EXPOSURE que significa "manual" según el backend (no está estandarizado)
EXPOSICION_MANUAL = {"dshow": 0.25, "msmf": 0.25, "v4l2": 1, "any": 0.25}


def fijar_camara(cap, backend, exposicion=None, cuadros=30):
    """Deja la cámara en automático unos cuadros para que encuentre una buena exposición y después apaga
    autofoco, balance de blancos y exposición automáticos. Cada driver acepta cosas distintas: se imprime
    qué quedó (cap.get después de cada cap.set)."""
    if sys.platform == "darwin":
        print("[cámara] --fijar-camara: en Mac OpenCV (AVFoundation) no puede cambiar exposición, foco ni "
              "balance de blancos; la cámara queda en automático.", flush=True)
        return
    print(f"[cámara] Fijando la cámara: {cuadros} cuadros en automático para que ajuste la exposición...",
          flush=True)
    for _ in range(cuadros):
        cap.read()
    pedidos = [("autofoco", cv2.CAP_PROP_AUTOFOCUS, 0),
               ("balance de blancos automático", cv2.CAP_PROP_AUTO_WB, 0),
               ("exposición automática", cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSICION_MANUAL.get(backend, 0.25))]
    if exposicion is not None:
        pedidos.append(("exposición", cv2.CAP_PROP_EXPOSURE, float(exposicion)))
    for nombre, prop, valor in pedidos:
        antes = cap.get(prop)
        ok = cap.set(prop, valor)
        ahora = cap.get(prop)
        if ok and abs(ahora - valor) < 1e-3:
            estado = "aceptado"
        elif ok:
            estado = "el driver dice que sí pero informa otro valor"
        else:
            estado = "el driver no lo acepta"
        print(f"[cámara]   {nombre}: antes {antes:g}, pedido {valor:g}, ahora {ahora:g} ({estado})", flush=True)
    print(f"[cámara]   exposición que quedó: {cap.get(cv2.CAP_PROP_EXPOSURE):g}"
          + ("" if exposicion is not None else " (para elegir otra: --exposicion N)"), flush=True)
    if backend == "msmf":
        print("[cámara]   con MSMF el significado de la exposición automática varía; si no quedó fija, "
              "probá --backend dshow", flush=True)


def open_camera(index, backend="auto", width=0, height=0, mjpg=False, fijar=False, exposicion=None):
    """Abre la cámara probando varios backends. Imprime el progreso en consola.
    fijar: apaga exposición, foco y balance automáticos después de 30 cuadros (ver fijar_camara)."""
    if backend == "auto":
        order = ["dshow", "msmf", "any"] if sys.platform.startswith("win") else ["any"]
    else:
        order = [backend]
    for name in order:
        print(f"[cámara] Abriendo índice {index} con backend '{name}'...", flush=True)
        cap = cv2.VideoCapture(index, BACKENDS[name])
        if not cap.isOpened():
            print(f"[cámara]   no se pudo abrir con '{name}'", flush=True)
            cap.release()
            continue
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"[cámara]   OK: {frame.shape[1]}x{frame.shape[0]} con '{name}'", flush=True)
            if fijar or exposicion is not None:
                fijar_camara(cap, name, exposicion)
            return cap
        print(f"[cámara]   se abrió pero no entrega imágenes con '{name}'", flush=True)
        cap.release()
    return None


def move_squares(board, mv):
    """Casillas cuyo contenido cambia físicamente con la jugada."""
    s = {mv.from_square, mv.to_square}
    if board.is_castling(mv):
        rank = chess.square_rank(mv.from_square)
        if board.is_kingside_castling(mv):
            s |= {chess.square(7, rank), chess.square(5, rank)}
        else:
            s |= {chess.square(0, rank), chess.square(3, rank)}
    elif board.is_en_passant(mv):
        s.add(chess.square(chess.square_file(mv.to_square),
                           chess.square_rank(mv.from_square)))
    return s


def vacated_filled(board, mv):
    """Casillas que quedan vacías y casillas vacías que pasan a ocuparse."""
    vac, fil = {mv.from_square}, set()
    if board.is_castling(mv):
        rank = chess.square_rank(mv.from_square)
        if board.is_kingside_castling(mv):
            vac.add(chess.square(7, rank)); fil |= {mv.to_square, chess.square(5, rank)}
        else:
            vac.add(chess.square(0, rank)); fil |= {mv.to_square, chess.square(3, rank)}
    elif board.is_en_passant(mv):
        vac.add(chess.square(chess.square_file(mv.to_square),
                             chess.square_rank(mv.from_square)))
        fil.add(mv.to_square)
    elif board.piece_at(mv.to_square) is None:
        fil.add(mv.to_square)
    return vac, fil


def empty_mask(board):
    return np.array([board.piece_at(s) is None for s in chess.SQUARES])


def candidate_moves(board):
    out, seen = [], set()
    for mv in board.legal_moves:
        if mv.promotion and mv.promotion != chess.QUEEN:
            continue            # la pieza de promoción no se distingue: dama
        key = (mv.from_square, mv.to_square)
        if key in seen:
            continue
        seen.add(key)
        vac, fil = vacated_filled(board, mv)
        out.append((mv, move_squares(board, mv), vac, fil))
    return out


class CalibrationWindow:
    """Ventana grande para marcar/arrastrar las 4 esquinas, con lupa y ajuste automático."""
    MAG_R, MAG_SIZE = 22, 220      # radio (px de cámara) y tamaño de la lupa en pantalla

    def __init__(self, app, initial=None):
        self.app = app
        self.pts = [tuple(p) for p in initial] if initial else []
        self.sel = None            # punto seleccionado (para flechas)
        self.drag = None
        self.mouse = None
        self.used = []
        self.alive = True
        self.top = tk.Toplevel(app.root)
        self.top.title("Calibración del tablero")
        self.top.configure(bg="#2b2b2b")
        h, w = app.frame.shape[:2]
        maxw = self.top.winfo_screenwidth() * 0.85
        maxh = self.top.winfo_screenheight() * 0.85 - 110
        self.scale = min(maxw / w, maxh / h, 2.0)
        self.info = tk.Label(self.top, fg="#f9e2af", bg="#2b2b2b", anchor="w",
                             justify="left", font=("Helvetica", 11))
        self.info.pack(fill="x", padx=8, pady=4)
        self.label = tk.Label(self.top, bg="black", cursor="crosshair")
        self.label.pack(padx=8)
        bar = tk.Frame(self.top, bg="#2b2b2b")
        bar.pack(fill="x", padx=8, pady=6)
        for txt, cmd in (("Ajuste automático", self.auto), ("Borrar puntos", self.clear),
                         ("Aceptar", self.accept), ("Cancelar", self.cancel)):
            ttk.Button(bar, text=txt, command=cmd).pack(side="left", padx=3)
        self.label.bind("<ButtonPress-1>", self.on_press)
        self.label.bind("<B1-Motion>", self.on_drag)
        self.label.bind("<ButtonRelease-1>", lambda e: setattr(self, "drag", None))
        self.label.bind("<Motion>", self.on_motion)
        self.label.bind("<Leave>", lambda e: setattr(self, "mouse", None))
        self.label.bind("<Button-3>", lambda e: self.pts and self.pts.pop())
        for key, d in (("Left", (-0.5, 0)), ("Right", (0.5, 0)),
                       ("Up", (0, -0.5)), ("Down", (0, 0.5))):
            self.top.bind(f"<{key}>", lambda e, d=d: self.nudge(*d))
        self.top.bind("<Return>", lambda e: self.accept())
        self.top.bind("<Escape>", lambda e: self.cancel())
        self.top.protocol("WM_DELETE_WINDOW", self.cancel)
        self.top.focus_set()
        self.refresh()

    # --- interacción
    def to_cam(self, ev):
        return ev.x / self.scale, ev.y / self.scale

    def nearest(self, p):
        if not self.pts:
            return None
        d = [np.hypot(p[0] - q[0], p[1] - q[1]) for q in self.pts]
        i = int(np.argmin(d))
        return i if d[i] * self.scale < 18 else None

    def on_press(self, ev):
        p = self.to_cam(ev)
        i = self.nearest(p)
        if i is None and len(self.pts) < 4:
            self.pts.append(p)
            i = len(self.pts) - 1
        self.drag = self.sel = i
        self.mouse = p

    def on_drag(self, ev):
        self.mouse = self.to_cam(ev)
        if self.drag is not None:
            self.pts[self.drag] = self.mouse

    def on_motion(self, ev):
        self.mouse = self.to_cam(ev)

    def nudge(self, dx, dy):
        if self.sel is not None and self.sel < len(self.pts):
            x, y = self.pts[self.sel]
            self.pts[self.sel] = (x + dx, y + dy)
            self.mouse = self.pts[self.sel]

    def clear(self):
        self.pts, self.used, self.sel = [], [], None

    def auto(self):
        if len(self.pts) < 4:
            return
        pts, used = self.app.vision.refine_corners(self.app.frame, self.pts)
        if len(used) >= 8:
            self.pts = [tuple(p) for p in pts]
            self.used = used
            self.flash = f"Ajustado con {len(used)} intersecciones (puntos celestes)."
        else:
            self.flash = ("No encontré suficientes intersecciones visibles. Acerca más "
                          "los puntos a mano y vuelve a probar (hace falta la posición "
                          "inicial o el tablero vacío).")

    def accept(self):
        if len(self.pts) == 4:
            self.alive = False
            self.top.destroy()
            self.app.finish_calibration(self.pts)

    def cancel(self):
        self.alive = False
        self.top.destroy()
        self.app.calib_win = None

    # --- dibujo
    def refresh(self):
        if not self.alive:
            return
        frame = self.app.frame
        img = frame.copy()
        n = len(self.pts)
        if n == 4:
            ideal = np.float32([[0, WARP], [WARP, WARP], [WARP, 0], [0, 0]])
            M = cv2.getPerspectiveTransform(ideal, np.float32(self.pts))
            for i in range(9):
                for a, b in (((0, i * SQ), (WARP, i * SQ)), ((i * SQ, 0), (i * SQ, WARP))):
                    seg = cv2.perspectiveTransform(np.float32([[a], [b]]), M).reshape(2, 2)
                    cv2.line(img, tuple(map(int, seg[0])), tuple(map(int, seg[1])),
                             (0, 255, 0), 1, cv2.LINE_AA)
        for (x, y) in self.used:
            cv2.circle(img, (int(x), int(y)), 3, (255, 200, 0), -1)
        disp = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        for i, (x, y) in enumerate(self.pts):
            c = (0, 140, 255) if i == self.sel else (0, 255, 255)
            X, Y = int(x * self.scale), int(y * self.scale)
            cv2.circle(disp, (X, Y), 7, c, 2)
            cv2.putText(disp, CORNER_NAMES[i], (X + 9, Y - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, c, 2)
        if self.mouse is not None:
            self.draw_magnifier(disp, img, self.mouse)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        self.label.configure(image=photo)
        self.label.image = photo
        if n < 4:
            msg = (f"Haz clic en la esquina exterior de {CORNER_NAMES[n]} ({n + 1}/4). "
                   "Clic derecho: borrar el último.")
        else:
            msg = ("Arrastra los puntos o usa las flechas (½ px) para afinar. "
                   "«Ajuste automático» los encaja con las intersecciones. Enter = aceptar.")
        flash = getattr(self, "flash", None)
        self.info.configure(text=msg + ("\n" + flash if flash else ""))
        self.top.after(40, self.refresh)

    def draw_magnifier(self, disp, img, p):
        r, S = self.MAG_R, self.MAG_SIZE
        pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT)
        x, y = int(round(p[0])) + r, int(round(p[1])) + r
        crop = pad[y - r:y + r + 1, x - r:x + r + 1]
        if crop.shape[0] != 2 * r + 1 or crop.shape[1] != 2 * r + 1:
            return
        mag = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
        cv2.line(mag, (S // 2, 0), (S // 2, S), (0, 0, 255), 1)
        cv2.line(mag, (0, S // 2), (S, S // 2), (0, 0, 255), 1)
        cv2.rectangle(mag, (0, 0), (S - 1, S - 1), (255, 255, 255), 2)
        H, W = disp.shape[:2]
        if S + 20 > W or S + 20 > H:
            return
        # esquina opuesta al cursor para no taparlo
        mx = 10 if p[0] * self.scale > W / 2 else W - S - 10
        disp[10:10 + S, mx:mx + S] = mag


class App:
    def __init__(self, root, cap, board_px, reopen=None, partida=None, vision_kwargs=None,
                 manual=True, rafaga=15, capturas=None):
        """manual: la jugada se detecta solo al pulsar «Detectar jugada» (o la barra espaciadora), con la
        mediana de una ráfaga de `rafaga` cuadros quietos; con manual=False, sola al quedar quieta la imagen.
        capturas: carpeta donde guardar cada detección (imágenes, posición y jugada real) para
        diagnostico_negras.py; None = no guardar."""
        self.root = root
        self.manual = manual
        self.n_rafaga = rafaga
        self.rafaga_desde = None       # cuadro en que se pidió la ráfaga (para el tiempo máximo)
        self.capturas = capturas
        self.captura_pendiente = None  # la última detección, hasta saber cuál fue la jugada real
        self.ultima_captura = None     # archivo de la última captura guardada (se marca si se deshace)
        self.n_capturas = 0
        self.partida = partida or Partida(con_camara=cap is not None)
        board_px = min(board_px, int(root.winfo_screenheight() * 0.82))
        self.board_px = board_px
        self.sq_px = board_px // 8
        self.cam_w = 440
        root.title("ChessCam")
        root.configure(bg="#2b2b2b")

        self.cap = cap
        self.reopen = reopen
        self.fail_count = 0
        self.frame_count = 0

        self.vision = BoardVision(**(vision_kwargs or {}))
        self.board = self.partida.board            # la misma posicion que lleva la partida
        bot = self.partida.bot
        self.flipped = bot is not None and self.partida.color_bot == chess.WHITE   # la persona abajo
        self.selected = None
        self.lim_porque = 1.0
        self.porque = None             # (fen, {casilla: importancia}) del botón «¿Por qué?», o None
        self.cartel = True             # cartel de fin de partida (se oculta con un clic para ver la posición)
        self.calib_win = None          # ventana de calibración abierta
        self.corners = None
        self.sync_pending = False      # tomar referencia en la próxima quietud
        self.frame = None
        self.warped = None
        self.cam_scale = 1.0
        self.view_mode = tk.StringVar(value="camera")

        self._build_ui()
        if self.cap is not None:
            self._load_calibration()
        else:
            self.set_status("Modo simulador: jugá haciendo clic en el tablero.")
        self.draw_board()
        self.after_human()                         # si el bot juega con blancas, empieza el
        self.loop()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        main = tk.Frame(self.root, bg="#2b2b2b")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(main, width=self.board_px, height=self.board_px,
                                highlightthickness=0, bg="#2b2b2b")
        self.canvas.grid(row=0, column=0, sticky="n")
        self.canvas.bind("<Button-1>", self.on_board_click)

        side = tk.Frame(main, bg="#2b2b2b", width=self.cam_w, height=self.board_px)
        side.grid(row=0, column=1, sticky="ns", padx=(12, 0))
        side.pack_propagate(False)      # no crece más que el tablero
        main.rowconfigure(0, weight=1)

        # --- reloj (si se pidió con --reloj): arriba el del bando que está arriba en el tablero
        self.reloj_lbl = {}
        if self.partida.reloj is not None:
            rf = tk.Frame(side, bg="#2b2b2b")
            rf.pack(fill="x", pady=(0, 6))
            for fila in (0, 1):
                caja = tk.Frame(rf)
                caja.grid(row=fila, column=0, sticky="ew", pady=1)
                t = tk.Label(caja, text="", font=("Consolas", 22, "bold"), anchor="e", width=7, padx=8)
                n = tk.Label(caja, text="", font=("Helvetica", 12, "bold"), anchor="w", padx=4)
                t.pack(side="left")
                n.pack(side="left", fill="both", expand=True)
                self.reloj_lbl[fila] = (caja, t, n)
            self.pausa_btn = ttk.Button(rf, text="Pausa", command=self.pausar_reloj, takefocus=0)
            self.pausa_btn.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(6, 0), pady=1)
            rf.columnconfigure(0, weight=1)
            self.pausa_calib = False     # el reloj se pausó solo al abrir la calibración

        # --- jugadas (arriba)
        tk.Label(side, text="Jugadas", fg="white", bg="#2b2b2b",
                 font=("Helvetica", 13, "bold")).pack(anchor="w")
        mv_frame = tk.Frame(side)
        mv_frame.pack(fill="x")
        self.moves_txt = tk.Text(mv_frame, width=30, height=9 if self.partida.reloj is None else 6, font=("Consolas", 12),
                                 bg="#1e1e1e", fg="#e6e6e6", relief="flat",
                                 state="disabled", wrap="none")
        sb = ttk.Scrollbar(mv_frame, command=self.moves_txt.yview)
        self.moves_txt.configure(yscrollcommand=sb.set)
        self.moves_txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # --- el bot: que jugada quiere y como se siente
        self.bot_lbl = tk.Label(side, text="", fg="#a6e3a1", bg="#2b2b2b", anchor="w",
                                justify="left", wraplength=self.cam_w, font=("Helvetica", 13, "bold"))
        self.bot_lbl.pack(fill="x", pady=(6, 0))
        self.bot_txt = tk.Label(side, text="", fg="#e6e6e6", bg="#2b2b2b", anchor="w",
                                justify="left", wraplength=self.cam_w, font=("Helvetica", 12))
        self.bot_txt.pack(fill="x")

        self.status = tk.Label(side, text="", fg="#9fd3ff", bg="#2b2b2b",
                               anchor="w", justify="left", wraplength=self.cam_w,
                               font=("Helvetica", 11))
        self.status.pack(fill="x", pady=(6, 4))

        btns = tk.Frame(side, bg="#2b2b2b")
        btns.pack(fill="x", pady=4)
        botones = [("Detectar jugada", self.detectar)] if (self.cap is not None and self.manual) else []
        for i, (txt, cmd) in enumerate(botones + [
            ("Calibrar", self.start_calibration),
            ("Sincronizar", self.sync_now),
            ("Deshacer", self.undo),
            ("Nueva partida", self.new_game),
            ("Girar tablero", self.flip),
            ("Guardar PGN", self.save_pgn),
            ("Pista", self.hint),
            ("¿Por qué?", self.toggle_porque),
            ("Bot: hecha", self.bot_move_done),
        ]):
            # takefocus=0: así la barra espaciadora no "pulsa" además el último botón usado
            ttk.Button(btns, text=txt, command=cmd, takefocus=0).grid(row=i // 3, column=i % 3,
                                                                      sticky="ew", padx=2, pady=2)
        if self.cap is not None and self.manual:
            self.root.bind("<space>", lambda e: self.detectar())
        for c in range(3):
            btns.columnconfigure(c, weight=1)
        for w in btns.grid_slaves():                # sin camara no hay nada que calibrar ni sincronizar
            if self.cap is None and w.cget("text") in ("Calibrar", "Sincronizar", "Bot: hecha"):
                w.state(["disabled"])
            if self.partida.bot is None and w.cget("text") in ("Pista", "Bot: hecha", "¿Por qué?"):
                w.state(["disabled"])

        if self.cap is None:                      # modo simulador: en lugar de la camara, un aviso
            tk.Label(side, text="Sin cámara (modo simulador).\nPara usar el tablero físico, abrilo sin --sin-camara.",
                     fg="#888", bg="#2b2b2b", justify="left", anchor="w",
                     font=("Helvetica", 11)).pack(fill="x", pady=(10, 0))
            return
        vm = tk.Frame(side, bg="#2b2b2b")
        vm.pack(fill="x")
        for val, txt in (("camera", "Cámara"), ("warped", "Rectificada + cambios")):
            tk.Radiobutton(vm, text=txt, value=val, variable=self.view_mode,
                           fg="white", bg="#2b2b2b", selectcolor="#444",
                           activebackground="#2b2b2b").pack(side="left")

        # --- cámara (abajo)
        # caja de tamaño fijo que ocupa el resto de la columna; la imagen se ajusta a ella
        self.cam_box = tk.Frame(side, bg="black")
        self.cam_box.pack(fill="both", expand=True, pady=(4, 0))
        self.cam_box.pack_propagate(False)
        self.cam_label = tk.Label(self.cam_box, bg="black")
        self.cam_label.pack(expand=True)
        self.cam_label.bind("<Button-1>", self.on_cam_click)

    def set_status(self, txt, color="#9fd3ff"):
        self.status.configure(text=txt, fg=color)

    # ------------------------------------------------------------ tablero
    def sq_to_xy(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped:
            f, r = 7 - f, 7 - r
        return f * self.sq_px, (7 - r) * self.sq_px

    def xy_to_sq(self, x, y):
        f, r = int(x // self.sq_px), 7 - int(y // self.sq_px)
        if not (0 <= f < 8 and 0 <= r < 8):
            return None
        if self.flipped:
            f, r = 7 - f, 7 - r
        return chess.square(f, r)

    def draw_board(self):
        c, s = self.canvas, self.sq_px
        c.delete("all")
        last = self.board.peek() if self.board.move_stack else None
        for sq in chess.SQUARES:
            x, y = self.sq_to_xy(sq)
            dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
            col = DARK_SQ if dark else LIGHT_SQ
            if last and sq in (last.from_square, last.to_square):
                col = HL_LAST
            if sq == self.selected:
                col = HL_SEL
            imp = self.importancias()
            if imp is not None and sq in imp:
                col = tinte_porque(col, imp[sq], self.lim_porque)
            c.create_rectangle(x, y, x + s, y + s, fill=col, outline="")
        # coordenadas
        fnt = ("Helvetica", max(9, s // 7), "bold")
        for i in range(8):
            f = 7 - i if self.flipped else i
            r = i if self.flipped else 7 - i
            c.create_text(i * s + s - 4, self.board_px - 3, text="abcdefgh"[f],
                          anchor="se", font=fnt, fill="#5a4630")
            c.create_text(3, i * s + 3, text=str(r + 1), anchor="nw",
                          font=fnt, fill="#5a4630")
        # piezas
        pf = ("DejaVu Sans", int(s * 0.62))
        for sq, p in self.board.piece_map().items():
            x, y = self.sq_to_xy(sq)
            cx, cy = x + s / 2, y + s / 2 + s * 0.03
            k = p.symbol().lower()
            if p.color == chess.WHITE:
                c.create_text(cx, cy, text=GLYPH[k], font=pf, fill="white")
                c.create_text(cx, cy, text=GLYPH_OUT[k], font=pf, fill="black")
            else:
                c.create_text(cx, cy, text=GLYPH[k], font=pf, fill="#111")
        imp = self.importancias()
        if imp is not None:                       # número abajo de cada pieza: cuánto pierde su dueño sin ella
            fn = ("Helvetica", max(8, s // 6), "bold")
            for sq, v in imp.items():
                x, y = self.sq_to_xy(sq)
                txt = f"{v:+.1f}".replace(".", ",")
                c.create_rectangle(x + s * .2, y + s * .78, x + s * .8, y + s * .98, fill="#ffffff", outline="")
                c.create_text(x + s / 2, y + s * .88, text=txt, font=fn, fill="#111")
        if self.board.is_check():
            ksq = self.board.king(self.board.turn)
            x, y = self.sq_to_xy(ksq)
            c.create_rectangle(x + 2, y + 2, x + s - 2, y + s - 2, outline="red", width=4)
        self.draw_legal_dots()
        if self.partida.pista is not None:
            self.draw_move_arrow(self.partida.pista, FLECHA_PISTA, before=self.board)
        if self.partida.flecha_bot is not None:
            # pendiente: la posicion todavia no la tiene; si ya la jugo (simulador), se mira la de antes
            antes = self.board
            if self.partida.pendiente is None and self.board.move_stack and \
                    self.board.peek() == self.partida.flecha_bot:
                antes = self.board.copy(); antes.pop()
            self.draw_move_arrow(self.partida.flecha_bot, FLECHA_BOT, before=antes)
        self.update_moves_text()
        self.update_bot_panel()
        self.draw_end_banner()

    def draw_end_banner(self):
        """Partida terminada: cartel grande en el medio del tablero (un clic lo oculta)."""
        fin = self.partida.resultado()
        if fin is None or not self.cartel:
            return
        titulo, detalle, tipo = fin
        c, B = self.canvas, self.board_px
        x0, x1 = B * 0.12, B * 0.88
        y0, y1 = B * 0.36, B * 0.64
        c.create_rectangle(x0 + 6, y0 + 6, x1 + 6, y1 + 6, fill="#111", outline="")          # sombra
        c.create_rectangle(x0, y0, x1, y1, fill="#2b2b2b", outline=COLOR_FIN[tipo], width=4)
        c.create_text(B / 2, y0 + (y1 - y0) * 0.32, text=titulo, fill=COLOR_FIN[tipo],
                      font=("Helvetica", max(14, int(B * 0.055)), "bold"))
        c.create_text(B / 2, y0 + (y1 - y0) * 0.62, text=detalle, fill="white",
                      font=("Helvetica", max(12, int(B * 0.04)), "bold"))
        seguir = "Nueva partida" if self.partida.tiempo_agotado is not None else "Nueva partida o Deshacer"
        c.create_text(B / 2, y0 + (y1 - y0) * 0.86, fill="#aaaaaa",
                      text=f"Clic para ver la posición · {seguir} para seguir",
                      font=("Helvetica", max(9, int(B * 0.019))))

    def status_fin(self):
        """Mensaje de fin en la línea de estado. Devuelve True si la partida terminó."""
        fin = self.partida.resultado()
        if fin is None:
            return False
        titulo, detalle, tipo = fin
        sep = " " if titulo.endswith("!") else ". "
        seguir = "«Nueva partida»" if self.partida.tiempo_agotado is not None else "«Nueva partida» o «Deshacer»"
        self.set_status(f"{titulo}{sep}{detalle}. Usá {seguir} para seguir.", COLOR_FIN[tipo])
        return True

    def sq_center(self, sq):
        x, y = self.sq_to_xy(sq)
        return x + self.sq_px / 2, y + self.sq_px / 2

    def draw_legal_dots(self):
        """Pieza elegida: punto en cada casilla a la que puede ir, anillo si ahí come."""
        if self.selected is None:
            return
        c, s = self.canvas, self.sq_px
        for mv in self.board.legal_moves:
            if mv.from_square != self.selected or (mv.promotion and mv.promotion != chess.QUEEN):
                continue
            cx, cy = self.sq_center(mv.to_square)
            if self.board.is_capture(mv):
                r = s * 0.44
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=ANILLO_COME, width=max(3, s // 16))
            else:
                r = s * 0.14
                c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=PUNTO, outline="")

    def draw_move_arrow(self, mv, color, before):
        """Flecha de origen a destino, anillo en el origen y marca en el destino, para poder copiar la jugada
        en el tablero fisico. Enroque: tambien la torre. Al paso: cruz en el peon que se saca."""
        c, s = self.canvas, self.sq_px
        w = max(4, int(s * 0.16))
        forma = (s * 0.42, s * 0.48, s * 0.2)

        def flecha(a, b, ancho, dy=0.0):
            (x0, y0), (x1, y1) = self.sq_center(a), self.sq_center(b)
            y0, y1 = y0 + dy, y1 + dy
            c.create_line(x0, y0, x1, y1, fill=color, width=ancho, arrow=tk.LAST,
                          arrowshape=forma, capstyle=tk.ROUND)

        ox, oy = self.sq_center(mv.from_square)
        r = s * 0.42
        c.create_oval(ox - r, oy - r, ox + r, oy + r, outline=color, width=max(3, s // 18))
        if before.is_castling(mv):
            fila = chess.square_rank(mv.from_square)
            corto = chess.square_file(mv.to_square) == 6
            t_de = chess.square(7 if corto else 0, fila)
            t_a = chess.square(5 if corto else 3, fila)
            # la torre va por la misma fila que el rey: su flecha se corre hacia el borde para no taparse
            _, yc = self.sq_center(mv.from_square)
            borde = 1 if yc > self.board_px / 2 else -1
            flecha(t_de, t_a, max(3, w // 2), dy=borde * s * 0.3)
        elif before.is_en_passant(mv):
            comido = chess.square(chess.square_file(mv.to_square), chess.square_rank(mv.from_square))
            cx, cy = self.sq_center(comido)
            d = s * 0.3
            for a, b in (((cx - d, cy - d), (cx + d, cy + d)), ((cx - d, cy + d), (cx + d, cy - d))):
                c.create_line(*a, *b, fill=ANILLO_COME, width=max(3, s // 14))
        flecha(mv.from_square, mv.to_square, w)
        if before.is_capture(mv) and not before.is_en_passant(mv):
            dx, dy = self.sq_center(mv.to_square)
            c.create_oval(dx - r, dy - r, dx + r, dy + r, outline=ANILLO_COME, width=max(3, s // 18))

    def update_bot_panel(self):
        p = self.partida
        self.bot_lbl.configure(text=p.estado_bot())
        lineas, color = [], "#e6e6e6"
        if p.bot is not None and p.pendiente is not None:
            lineas.append(f"➜ Hacé en el tablero: {p.texto_bot} (flecha verde)")
            color = "#f9e2af"
        elif p.bot is not None and p.flecha_bot is not None:
            lineas.append(f"El bot jugó: {p.texto_bot} (flecha verde)")
        if p.bot is not None and p.pista is not None:
            lineas.append(f"Pista: {san_es(self.board, p.pista)} (flecha azul)")
        imp = self.importancias()
        if imp:
            lineas.append("¿Por qué? Lo que más pesa para el bot: " + resumen_porque(self.board, imp)
                          + ". Azul = la pieza ayuda a su dueño; rojo = lo perjudica.")
        self.bot_txt.configure(text="\n".join(lineas), fg=color)

    def update_moves_text(self):
        tmp = self.board.root()          # la posicion en la que empezo (no siempre la inicial)
        lines = []
        for i, mv in enumerate(self.board.move_stack):
            san = san_es(tmp, mv)
            tmp.push(mv)
            if i % 2 == 0:
                lines.append(f"{i // 2 + 1:>3}. {san:<9}")
            else:
                lines[-1] += f"{san}"
        if self.partida.terminada():
            lines.append(f"\n  {self.partida.resultado_pgn()}"
                         + ("  (por tiempo)" if self.partida.tiempo_agotado is not None else ""))
        t = self.moves_txt
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", "\n".join(lines))
        t.configure(state="disabled")
        t.see("end")

    def on_board_click(self, ev):
        sq = self.xy_to_sq(ev.x, ev.y)
        if sq is None:
            return
        if self.partida.terminada():               # terminada: el clic oculta el cartel y avisa
            self.cartel = False
            self.selected = None
            self.status_fin()
            self.draw_board()
            return
        p = self.board.piece_at(sq)
        if self.selected is not None and sq != self.selected:
            mv = chess.Move(self.selected, sq)
            q = self.board.piece_at(self.selected)
            if q and q.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                mv.promotion = chess.QUEEN
            if mv in self.board.legal_moves:
                self.selected = None
                self.apply_move(mv, manual=True)
                return
        # elegir (o cambiar) la pieza: solo del bando que mueve
        self.selected = sq if (p and p.color == self.board.turn and sq != self.selected) else None
        self.draw_board()

    def set_ref(self, warped):
        self.vision.set_reference(warped, empty_mask(self.board))

    def apply_move(self, mv, manual=False, conf=None):
        """Toda jugada (camara o clic) pasa por la partida, que controla los turnos del bot.
        Devuelve True si se aplico."""
        fen_antes = self.board.fen()
        ok, msg = self.partida.jugada(mv)
        if not ok:
            self.draw_board()
            self.set_status(msg, "#f38ba8")
            return False
        self.guardar_captura(fen_antes, mv, "manual" if manual else "detectada")
        if self.warped is not None and self.cap is not None:
            self.set_ref(self.warped)
        how = "manual" if manual else f"detectada, confianza {conf:.2f}"
        self.set_status(f"{msg.rstrip('.')} ({how}).", "#a6e3a1")
        self.cartel = True
        self.status_fin()
        self.draw_board()
        self.after_human()
        return True

    def after_human(self):
        """Si ahora le toca al bot, que piense (en otro hilo; la ventana sigue respondiendo)."""
        if self.partida.turno_del_bot() and self.partida.pensar():
            self.update_bot_panel()

    def poll_bot(self):
        ev = self.partida.revisar()
        if ev is None:
            return
        tipo, dato = ev
        if tipo == "error":
            self.set_status(f"El bot falló: {type(dato).__name__}: {dato}", "#f38ba8")
        elif tipo == "bot":
            if self.partida.con_camara:
                self.set_status("Hacé en el tablero la jugada de la flecha verde"
                                + (" y pulsá «Detectar jugada» (o espacio)." if self.manual else "."), "#f9e2af")
            else:
                self.set_status("Tu turno.", "#a6e3a1")
            self.cartel = True
            self.status_fin()
        elif tipo == "pista":
            self.set_status("La flecha azul es lo que jugaría el bot en tu lugar.", "#9fd3ff")
        self.draw_board()

    # ------------------------------------------------------------ ¿Por qué?
    def importancias(self):
        """El mapa del botón «¿Por qué?» si corresponde a la posición actual (se apaga solo al mover)."""
        if self.porque is None or self.porque[0] != self.board.fen():
            return None
        return self.porque[1]

    def toggle_porque(self):
        """Muestra u oculta, sobre el tablero, cuánto vale cada pieza para el bot en esta posición (oclusión:
        se saca cada pieza y se mira cuánto cambia la evaluación de la red; ver explicar.py)."""
        if self.importancias() is not None:
            self.porque = None
            self.draw_board()
            return
        bot = self.partida.bot
        if bot is None:
            return
        if self.partida.pensando:
            self.set_status("Esperá a que el bot termine de pensar.", "#f9e2af")
            return
        try:
            imp = bot.porque(self.board)
        except Exception as e:
            self.set_status(f"No pude calcular el «¿Por qué?»: {type(e).__name__}: {e}", "#f38ba8")
            return
        self.porque = (self.board.fen(), imp)
        self.lim_porque = max([1.0] + [abs(v) for v in imp.values()])
        self.set_status("Número = cuántos peones pierde el dueño de esa pieza si se la saca, según el bot. "
                        "Se apaga al mover o pulsando otra vez «¿Por qué?».", "#9fd3ff")
        self.draw_board()

    def hint(self):
        p = self.partida
        if p.bot is None or p.pendiente is not None or p.turno_del_bot() or p.terminada():
            return
        if p.pensar(pista=True):
            self.set_status("Pensando una pista…", "#888")

    def bot_move_done(self):
        """La jugada del bot se hizo en el tablero real pero la camara no la vio."""
        pend = self.partida.pendiente
        if pend is None:
            self.set_status("No hay ninguna jugada del bot pendiente.", "#f9e2af")
            return
        self.apply_move(pend, manual=True)

    def undo(self):
        if not self.board.move_stack and self.partida.pendiente is None:
            return
        if self.partida.tiempo_agotado is not None:       # perder por tiempo no se deshace
            self.set_status("Se terminó el tiempo: para seguir, empezá una «Nueva partida».", "#f9e2af")
            return
        self.partida.deshacer()
        self.marcar_captura_deshecha()
        self.selected = None
        self.cartel = True
        self.draw_board()
        if self.cap is not None:
            self.sync_pending = True
            self.set_status("Deshecho. Devuelve las piezas físicamente; "
                            "se sincronizará al quedarse quieta la imagen.", "#f9e2af")
        else:
            self.set_status("Deshecho.", "#f9e2af")
        self.after_human()

    def new_game(self):
        self.partida.nueva()                       # el reloj vuelve al tiempo inicial y arranca al estar listo
        self.selected = None
        self.cartel = True
        self.draw_board()
        if self.cap is not None:
            self.sync_pending = True
            self.set_status("Nueva partida. Coloca la posición inicial; se sincronizará "
                            "al quedarse quieta la imagen (o pulsa Sincronizar).", "#f9e2af")
        else:
            self.set_status("Nueva partida.", "#a6e3a1")
        self.after_human()

    def sync_now(self):
        if self.warped is not None:
            self.set_ref(self.warped)
            self.sync_pending = False
            self.set_status("Referencia tomada: la imagen actual corresponde "
                            "al tablero de la pantalla.")

    def flip(self):
        self.flipped = not self.flipped
        self.draw_board()

    def save_pgn(self):
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = "ChessCam"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["Result"] = self.partida.resultado_pgn()
        r = self.partida.reloj
        if r is not None:
            game.headers["TimeControl"] = f"{r.base:g}+{r.incremento:g}"
            if self.partida.tiempo_agotado is not None:
                game.headers["Termination"] = "time forfeit"
        path = filedialog.asksaveasfilename(defaultextension=".pgn",
                                            filetypes=[("PGN", "*.pgn")])
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                print(game, file=fh)
            self.set_status(f"Guardado en {path}")

    # --------------------------------------------------------- calibración
    def start_calibration(self):
        if self.frame is None:
            return
        if self.calib_win is not None:
            self.calib_win.top.lift()
            return
        self.calib_win = CalibrationWindow(self, self.corners)
        self.set_status("Calibrando… (la detección está en pausa)", "#f9e2af")

    def on_cam_click(self, ev):
        self.start_calibration()

    def finish_calibration(self, pts):
        self.calib_win = None
        self.corners = [list(map(float, p)) for p in pts]
        self.vision.set_corners(self.corners)
        with open(CALIB_FILE, "w") as fh:
            json.dump({"corners": self.corners}, fh)
        self.sync_pending = True
        self.set_status("Calibrado. Se tomará la referencia al quedarse quieta la imagen.")

    def _load_calibration(self):
        if os.path.exists(CALIB_FILE):
            try:
                with open(CALIB_FILE) as fh:
                    self.corners = json.load(fh)["corners"]
                self.vision.set_corners(self.corners)
                self.sync_pending = True
                self.set_status("Calibración cargada. Coloca la posición inicial; "
                                "si la cuadrícula no encaja, pulsa Calibrar.")
                return
            except Exception:
                pass
        self.set_status("Pulsa «Calibrar» y marca las esquinas a1, h1, h8, a8.",
                        "#f9e2af")

    # --------------------------------------------------------------- bucle
    def loop(self):
        try:
            self.poll_bot()
            self.tick_reloj()
            if self.cap is None:                   # modo simulador: no hay camara que leer
                return
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.fail_count = 0
                self.frame = frame
                self.frame_count += 1
                if self.vision.calibrated and self.calib_win is None:
                    self.process(frame)
                self.show_camera()
            else:
                self.fail_count += 1
                if self.fail_count == 15:
                    self.set_status("La cámara dejó de enviar imágenes. Reintentando… "
                                    "(¿otra aplicación la está usando?)", "#f38ba8")
                    print("[cámara] sin imágenes; reabriendo", flush=True)
                if self.fail_count % 60 == 15 and self.reopen is not None:
                    self.cap.release()
                    cap = self.reopen()
                    if cap is not None:
                        self.cap = cap
                        self.fail_count = 0
                        self.sync_pending = True
                        self.set_status("Cámara recuperada. Se resincronizará al quedarse "
                                        "quieta la imagen.", "#f9e2af")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.set_status(f"Error interno: {type(e).__name__}: {e}. Mira la terminal "
                            "y pásame el mensaje.", "#f38ba8")
        finally:
            self.root.after(30, self.loop)

    # --------------------------------------------------------------- reloj
    def tick_reloj(self):
        """Arranca el reloj cuando el tablero está listo, lo pausa durante la calibración, detecta la caída de
        bandera y actualiza los números."""
        p, r = self.partida, self.partida.reloj
        if r is None:
            return
        listo = self.cap is None or (self.vision.reference is not None and not self.sync_pending)
        if not r.iniciado and listo and not p.terminada():
            r.iniciar(self.board.turn)             # en el tablero físico, recién cuando la cámara tiene referencia
        if self.calib_win is not None and r.iniciado and not r.pausado:
            r.pausar()
            self.pausa_calib = True
        elif self.calib_win is None and self.pausa_calib:
            self.pausa_calib = False
            r.reanudar()
        if r.iniciado and not r.pausado and self.board.is_game_over():
            r.pausar()                             # mate, ahogado, etc.: los relojes se detienen
        if p.revisar_reloj() is not None:          # bandera: termina la partida
            if self.cap is not None and self.vision.en_rafaga:
                self.vision.cancelar_rafaga()
            self.selected = None
            self.cartel = True
            self.status_fin()
            self.draw_board()
        self.draw_reloj()

    def draw_reloj(self):
        p, r = self.partida, self.partida.reloj
        arriba = chess.WHITE if self.flipped else chess.BLACK
        for fila, color in ((0, arriba), (1, not arriba)):
            t = r.tiempo(color)
            activo = r.iniciado and r.corre == color and not p.terminada()
            nombre = "Blancas" if color == chess.WHITE else "Negras"
            if p.bot is not None:
                nombre += " (bot)" if color == p.color_bot else " (vos)"
            if t <= 0:
                fg, bg = "white", "#c0392b"
            elif activo:
                fg, bg = ("#f38ba8" if t < 10 else "#111111"), "#e6e6e6"
            else:
                fg, bg = ("#f38ba8" if t < 10 else "#9a9a9a"), "#1e1e1e"
            pausa = "\nen pausa" if activo and r.pausado else ""
            caja, lt, ln = self.reloj_lbl[fila]
            caja.configure(bg=bg)
            lt.configure(text=formato_tiempo(t), fg=fg, bg=bg)
            ln.configure(text=nombre + pausa, fg=fg, bg=bg)
        fin = p.terminada()
        self.pausa_btn.configure(text="Reanudar" if r.pausado and not self.pausa_calib and not fin else "Pausa")
        self.pausa_btn.state(["disabled"] if fin or not r.iniciado else ["!disabled"])

    def pausar_reloj(self):
        r = self.partida.reloj
        if r is None or not r.iniciado or self.partida.terminada():
            return
        if r.pausado:
            self.pausa_calib = False
            r.reanudar()
            self.set_status("Reloj en marcha.", "#a6e3a1")
        else:
            r.pausar()
            self.set_status("Reloj en pausa: «Reanudar» para seguir.", "#f9e2af")

    def process(self, frame):
        # los cambios por casilla se calculan en cada cuadro solo si se están mostrando
        warped, event = self.vision.update(frame, puntajes=self.ver_cambios())
        self.warped = warped
        if self.vision.reference is None:
            if self.vision.listo_para_referencia():          # quieta (y, con la visión nueva, ~2 s después de arrancar)
                self.set_ref(warped)
                self.sync_pending = False
                self.set_status("Listo. Juegan las blancas.")
            return
        if event == "rafaga":                      # se juntaron los cuadros quietos pedidos con «Detectar jugada»
            self.rafaga_desde = None
            if self.sync_pending:
                self.set_ref(warped)
                self.sync_pending = False
                self.set_status("Sincronizado con el tablero.")
                return
            self.detect_move()
            return
        if self.vision.en_rafaga and self.rafaga_desde is not None \
                and self.vision.cuadros - self.rafaga_desde > self.MAX_CUADROS_RAFAGA:
            self.vision.cancelar_rafaga()
            self.rafaga_desde = None
            self.set_status("La imagen no se quedó quieta. Retirá la mano y volvé a pulsar «Detectar jugada».",
                            "#f38ba8")
            return
        if event == "moving":
            if not self.vision.en_rafaga:
                self.status.configure(text="Movimiento…", fg="#888")
        elif event == "settled":
            if self.sync_pending:
                self.set_ref(warped)
                self.sync_pending = False
                self.set_status("Sincronizado con el tablero.")
                return
            if not self.manual:
                self.detect_move()
            elif not self.vision.en_rafaga:
                self.set_status("Cuando termines la jugada, pulsá «Detectar jugada» (o espacio).", "#9fd3ff")

    MAX_CUADROS_RAFAGA = 150           # ~5 s: si en ese tiempo no hay `rafaga` cuadros quietos, se avisa

    def ver_cambios(self):
        return self.view_mode.get() == "warped"

    def detectar(self):
        """Botón «Detectar jugada»: junta una ráfaga de cuadros quietos y, al completarla, busca la jugada."""
        if self.cap is None or not self.vision.calibrated or self.calib_win is not None:
            return
        if self.vision.reference is None:
            self.set_status("Todavía no hay referencia: esperá a que la imagen quede quieta.", "#f9e2af")
            return
        if self.partida.terminada():
            self.status_fin()
            return
        if self.partida.pensando:
            self.set_status("Esperá la jugada del bot.", "#f9e2af")
            return
        if self.vision.en_rafaga:
            return
        self.vision.iniciar_rafaga(self.n_rafaga)
        self.rafaga_desde = self.vision.cuadros
        self.set_status("Mirando el tablero… no muevas nada.", "#888")

    # ------------------------------------------------- capturas para diagnóstico
    def guardar_captura(self, fen_antes, mv, fuente):
        """Si --capturas está activo y la última detección era de esta posición, la guarda con la jugada real
        (la que terminó aplicándose: detectada o cargada a mano)."""
        cap, self.captura_pendiente = self.captura_pendiente, None
        if not self.capturas or cap is None or cap["fen"] != fen_antes:
            return
        os.makedirs(self.capturas, exist_ok=True)
        self.n_capturas += 1
        ruta = os.path.join(self.capturas, f"{self.n_capturas:03d}_{mv.uci()}.npz")
        np.savez_compressed(ruta, ref=cap["ref"], img=cap["img"], fen=fen_antes, jugada=mv.uci(),
                            fuente=fuente, detectada=cap["detectada"])
        self.ultima_captura = ruta
        print(f"[captura] {ruta} ({fuente})", flush=True)

    def marcar_captura_deshecha(self):
        """Si se deshace la jugada de la última captura, su 'jugada real' no es confiable: se renombra."""
        if self.ultima_captura and os.path.exists(self.ultima_captura):
            os.replace(self.ultima_captura, self.ultima_captura[:-4] + "_deshecha.npz")
        self.ultima_captura = None

    def detect_move(self):
        if self.partida.pensando:                  # el bot todavia no dijo su jugada: sin tocar la referencia,
            self.set_status("Esperá la jugada del bot.", "#f9e2af")   # asi lo movido se evalua despues
            return
        raw = self.vision.last_scores
        z = normalize_scores(raw)
        if z.max() < 1.5 and not self.manual:
            # nada relevante cambió (p. ej. una mano que pasó): refresca referencia
            self.set_ref(self.warped)
            return
        # con el botón la persona dice que jugó: se busca igual (pick_move exige un mínimo) y la referencia NO
        # se refresca, para no perder una jugada que cambió poco (pieza negra sobre casilla oscura)
        if self.partida.terminada():
            return
        ed = None
        if self.vision.last_feats is not None:
            ed = self.vision.empty_distance(self.vision.last_feats)
        mv, conf, ranked = pick_move(raw, candidate_moves(self.board), ed,
                                     self.vision.empty_valid, empty_mask(self.board))
        print("[jugada] " + ", ".join(f"{self.board.san(m)}={sc:.1f}"
                                      for sc, m in ranked[:4]), flush=True)
        if self.capturas and self.vision.reference_img is not None:
            self.captura_pendiente = {"ref": self.vision.reference_img,
                                      "img": self.vision.imagen_estable(self.warped).copy(),
                                      "fen": self.board.fen(), "detectada": mv.uci() if mv else ""}
        if mv is None and z.max() < 1.5:
            self.set_status("No veo cambios en el tablero. Si ya jugaste, ajustá la pieza y volvé a pulsar, "
                            "o marcá la jugada en la pantalla.", "#f38ba8")
            return
        if mv is not None:
            if not self.apply_move(mv, conf=conf):
                pass            # no era la jugada esperada: la referencia NO se actualiza (se corrige y se reevalua)
        else:
            alt = ", ".join(self.board.san(m) for _, m in ranked[:3])
            self.set_status("No estoy seguro de la jugada (candidatas: " + alt +
                            "). Ajusta la pieza o márcala en el tablero.", "#f38ba8")
            # la referencia NO se actualiza: al reajustar la pieza se vuelve a evaluar

    # -------------------------------------------------------- vista cámara
    def show_camera(self):
        if self.cap is None or self.frame is None:
            return
        if self.view_mode.get() == "warped" and self.warped is not None:
            img = self.warped.copy()
            z = normalize_scores(self.vision.last_scores)
            overlay = img.copy()
            for idx in range(64):
                f, r = idx % 8, idx // 8
                x0, y0 = f * SQ, (7 - r) * SQ
                v = min(1.0, z[idx] / 5.0)
                if v > 0.05:
                    cv2.rectangle(overlay, (x0, y0), (x0 + SQ, y0 + SQ),
                                  (0, 0, int(255 * v)), -1)
            img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
            for i in range(9):
                cv2.line(img, (0, i * SQ), (WARP, i * SQ), (0, 255, 0), 1)
                cv2.line(img, (i * SQ, 0), (i * SQ, WARP), (0, 255, 0), 1)
        else:
            img = self.frame.copy()
            if self.vision.calibrated:
                for a, b in self.vision.grid_lines_camera():
                    cv2.line(img, tuple(map(int, a)), tuple(map(int, b)),
                             (0, 255, 0), 1, cv2.LINE_AA)
                a1 = cv2.perspectiveTransform(
                    np.float32([[[SQ / 2, WARP - SQ / 2]]]), self.vision.Hinv)[0, 0]
                cv2.putText(img, "a1", tuple(map(int, a1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
        cv2.putText(img, f"#{self.frame_count}", (6, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        h, w = img.shape[:2]
        bw, bh = self.cam_box.winfo_width(), self.cam_box.winfo_height()
        if bw < 50 or bh < 50:               # todavía sin dibujar
            bw, bh = self.cam_w, self.cam_w
        scale = min(bw / w, bh / h)          # cabe entera, sin recortes
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        if self.view_mode.get() == "camera":
            self.cam_scale = scale
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.cam_label.configure(image=photo)
        self.cam_label.image = photo

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="índice de la webcam")
    ap.add_argument("--board", type=int, default=720, help="tamaño del tablero (px)")
    ap.add_argument("--backend", default="auto", choices=["auto"] + list(BACKENDS),
                    help="backend de captura (en Windows prueba dshow o msmf)")
    ap.add_argument("--width", type=int, default=0, help="ancho pedido a la cámara (0 = el suyo)")
    ap.add_argument("--height", type=int, default=0, help="alto pedido a la cámara (0 = el suyo)")
    ap.add_argument("--mjpg", action="store_true",
                    help="pide formato MJPG (ayuda con algunas webcams USB)")
    ap.add_argument("--ck", default=None, help="checkpoint del bot para jugar contra él (sin --ck: dos personas)")
    ap.add_argument("--bot", choices=["blancas", "negras"], default="negras", help="con qué juega el bot")
    ap.add_argument("--depth", type=int, default=2, help="profundidad de búsqueda del bot (1 = rápido y más débil)")
    ap.add_argument("--sin-camara", action="store_true", help="solo el simulador: jugar con clics en la pantalla")
    ap.add_argument("--fijar-camara", action="store_true",
                    help="apaga autofoco, balance de blancos y exposición automáticos después de 30 cuadros")
    ap.add_argument("--exposicion", type=float, default=None,
                    help="exposición fija (con DirectShow suele ser una potencia de 2 negativa: -5, -6, -7...)")
    ap.add_argument("--vision-original", action="store_true",
                    help="el detector de antes del 19/11 (sin promedio, sin espera, sin tolerancias), para comparar")
    ap.add_argument("--auto", action="store_true",
                    help="detecta la jugada sola al quedar quieta la imagen (sin el botón «Detectar jugada»)")
    ap.add_argument("--rafaga", type=int, default=15,
                    help="cuadros quietos que se combinan (mediana) al pulsar «Detectar jugada»")
    ap.add_argument("--capturas", nargs="?", const="", default=None, metavar="CARPETA",
                    help="guarda cada detección con la jugada real para chesscam/diagnostico_negras.py "
                         "(por defecto en chesscam/capturas/<fecha>)")
    ap.add_argument("--reloj", default=None, metavar="MIN+INC",
                    help="reloj de ajedrez: minutos por jugador + segundos de incremento por jugada (ej. 5+3, 10, "
                         "3+2). «Detectar jugada» (espacio) aprieta el reloj de quien jugó si la cámara confirma la "
                         "jugada; si no, sigue corriendo el suyo")
    ap.add_argument("--filtro-negras", action="store_true",
                    help="activa el filtro para piezas negras (gamma en las sombras y textura local)")
    ap.add_argument("--pantalla", default=None, metavar="PUERTO",
                    help="puerto serie del display LED de los ojos (por ejemplo COM5); ver pantalla_led/")
    args = ap.parse_args()
    vision_kwargs = dict(VISION_ORIGINAL) if args.vision_original else {}
    if args.vision_original:
        print("[visión] detector original (sin promedio, sin espera inicial, sin tolerancias)", flush=True)
    if args.filtro_negras:
        vision_kwargs.update(FILTRO_NEGRAS)
        print(f"[visión] filtro para piezas negras: {FILTRO_NEGRAS}", flush=True)
    capturas = args.capturas
    if capturas == "":
        capturas = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capturas",
                                time.strftime("%Y%m%d-%H%M%S"))
    if capturas:
        print(f"[capturas] se guardan en {capturas}", flush=True)

    cap = None
    if not args.sin_camara:
        cap = open_camera(args.cam, args.backend, args.width, args.height, args.mjpg,
                          args.fijar_camara, args.exposicion)
        if cap is None:
            print(f"No se pudo obtener imagen de la cámara {args.cam}. Prueba otro --cam o "
                  "--backend, y cierra otras apps que usen la cámara (Zoom, Teams, Cámara). "
                  "Para jugar solo en la pantalla, usa --sin-camara.")
            sys.exit(1)

    reloj = None
    if args.reloj:
        try:
            reloj = Reloj(*leer_control(args.reloj))
        except ValueError as e:
            ap.error(str(e))
        print(f"[reloj] {reloj.base / 60:g} min + {reloj.incremento:g} s por jugada", flush=True)

    bot = None
    if args.ck:
        print(f"[bot] cargando {args.ck} (profundidad {args.depth})...", flush=True)
        bot = Bot(args.ck, args.depth)
    pantalla = None
    if args.pantalla:
        from pantalla_led.pantalla_led import PantallaLED
        print(f"[pantalla] abriendo el display LED en {args.pantalla}...", flush=True)
        pantalla = PantallaLED(args.pantalla)
        pantalla.pensando(borrar_jugada=True)
    partida = Partida(bot, chess.WHITE if args.bot == "blancas" else chess.BLACK, con_camara=cap is not None,
                      reloj=reloj, pantalla=pantalla)

    root = tk.Tk()
    app = App(root, cap, args.board, partida=partida, vision_kwargs=vision_kwargs,
              manual=not args.auto, rafaga=args.rafaga, capturas=capturas,
              reopen=lambda: open_camera(args.cam, args.backend, args.width,
                                         args.height, args.mjpg, args.fijar_camara, args.exposicion))
    # Permite cerrar con Ctrl+C desde la terminal
    import signal
    signal.signal(signal.SIGINT, lambda *a: app.close())
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
