"""
Detecta las jugadas que se hacen en un tablero real mirando la webcam, en tiempo real.

Idea (la de los robots de ajedrez clasicos: Urting 2003, Sokic 2008, Matuszek 2011): el robot ya sabe cual es
la posicion, asi que no hace falta reconocer cada pieza. Alcanza con ver QUE CASILLAS CAMBIARON despues de que
la mano sale del tablero y elegir, entre las jugadas legales, la que explica esos cambios.

  1. Calibracion (una sola vez, con la camara fija): se marcan las 4 esquinas del tablero (a1, h1, h8, a8).
     Con eso se "endereza" la imagen (homografia) y cada casilla queda en un lugar fijo.
  2. Estabilidad: mientras hay movimiento (la mano) no se mira nada. Cuando la imagen queda quieta un rato,
     se compara con la ultima imagen quieta (la referencia).
  3. Mapa de cambios: diferencia de color (espacio LAB) pixel a pixel entre las dos imagenes quietas.
  4. Jugada: para cada jugada legal se calcula que zonas de la imagen deberian cambiar (2 casillas, 3 al paso,
     4 en el enroque, cada una con su "huella": desde una camara inclinada una pieza alta tapa parte de la
     casilla de atras) y se elige la que explica los cambios: todas sus casillas cambiaron y fuera de su zona no.

Uso:
    pip install opencv-python python-chess numpy
    python vision/detectar_jugadas.py --listar-camaras            # ver que numero tiene la webcam
    python vision/detectar_jugadas.py --calibrar --camara 1       # marcar las esquinas (guarda la camara y la calibracion)
    python vision/detectar_jugadas.py                             # seguir una partida entre dos personas
    python vision/detectar_jugadas.py --ck runs/final/ck_150000.pt --bot negras   # jugar contra el bot
    python vision/detectar_jugadas.py --video partida.mp4         # probar con un video grabado

Se abren dos ventanas: la camara (con la grilla) y "partida", un tablero digital estilo Lichess con la posicion
detectada, la jugada que propone el bot como flecha, su chance de ganar y la lista de jugadas.

Teclas: q salir | r tomar la imagen actual como referencia | u deshacer | espacio forzar la deteccion
        l abrir la posicion en el analisis de Lichess | --depurar muestra tambien el mapa de cambios
"""
import argparse
import json
import os
import sys
import time
import webbrowser

import chess
import chess.pgn
import cv2
import numpy as np

S = 80                      # lado de una casilla en la imagen enderezada (pixeles)
LADO = 8 * S
AQUI = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(AQUI, "calibracion.json")


# =============================================================== calibracion
class Calibracion:
    """Homografia entre la imagen de la camara y el tablero visto desde arriba (blancas abajo)."""

    def __init__(self, puntos):
        self.puntos = np.float32(puntos)                       # a1, h1, h8, a8 (esquinas exteriores)
        destino = np.float32([[0, LADO], [LADO, LADO], [LADO, 0], [0, 0]])
        self.H = cv2.getPerspectiveTransform(self.puntos, destino)
        self.Hinv = np.linalg.inv(self.H)
        self.cerca = (0.0, 0.0)

    def fijar_cercania(self, ancho, alto):
        """La parte de abajo de la imagen es la mas cercana a la camara. Se averigua hacia donde queda eso en el
        tablero enderezado, para medir cada casilla en la zona de la base de la pieza (lo alto de una pieza
        tapa la casilla de atras, la base no)."""
        p = cv2.perspectiveTransform(np.float32([[[ancho / 2, alto]]]), self.H)[0, 0]
        v = p - np.float32([LADO / 2, LADO / 2])
        n = float(np.linalg.norm(v))
        self.cerca = (float(v[0] / n), float(v[1] / n)) if n > 1e-6 else (0.0, 0.0)

    def enderezar(self, frame):
        return cv2.warpPerspective(frame, self.H, (LADO, LADO))

    def a_camara(self, x, y):
        return cv2.perspectiveTransform(np.float32([[[x, y]]]), self.Hinv)[0, 0]

    def guardar(self, path, ancho, alto, camara=None):
        with open(path, "w") as f:
            json.dump({"puntos": self.puntos.tolist(), "ancho": ancho, "alto": alto, "camara": camara}, f, indent=1)

    @staticmethod
    def cargar(path):
        with open(path) as f:
            d = json.load(f)
        c = Calibracion(d["puntos"])
        c.fijar_cercania(d["ancho"], d["alto"])
        c.camara = d.get("camara")
        return c, (d["ancho"], d["alto"])


def esquinas_automaticas(frame):
    """Con el tablero VACIO, OpenCV encuentra las 7x7 esquinas interiores; se extrapolan las 4 exteriores.
    No sabe cual es a1: despues se gira (g) o espeja (e) hasta que las etiquetas queden bien."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok, pts = cv2.findChessboardCornersSB(gris, (7, 7), flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return None
    pts = pts.reshape(7, 7, 2)
    ideal = np.float32([[c + 1, r + 1] for r in range(7) for c in range(7)])
    H, _ = cv2.findHomography(ideal, pts.reshape(-1, 2))
    ext = cv2.perspectiveTransform(np.float32([[[0, 0], [8, 0], [8, 8], [0, 8]]]), H)[0]
    return ext.tolist()


def calibrar(camara, salida):
    cap = abrir_camara(camara)
    puntos, giro = [], 0
    nombres = ["a1", "h1", "h8", "a8"]

    def click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append([float(x), float(y)])

    cv2.namedWindow("calibrar")
    cv2.setMouseCallback("calibrar", click)
    print("Hace clic en las esquinas EXTERIORES del tablero en este orden: a1, h1, h8, a8.\n"
          "  a = deteccion automatica (tablero vacio) | g = girar | e = espejar | b = borrar | s = guardar | q = salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vista = frame.copy()
        orden = puntos[giro:] + puntos[:giro] if len(puntos) == 4 else puntos
        for i, (x, y) in enumerate(orden):
            cv2.circle(vista, (int(x), int(y)), 7, (0, 0, 255), -1)
            cv2.putText(vista, nombres[i], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(puntos) == 4:
            c = Calibracion(orden)
            dibujar_grilla(vista, c)
            cv2.imshow("enderezado", c.enderezar(frame))
        else:
            cv2.putText(vista, f"clic en {nombres[len(puntos)]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("calibrar", vista)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("b"):
            puntos.clear(); giro = 0
        if k == ord("g") and len(puntos) == 4:
            giro = (giro + 1) % 4
        if k == ord("e") and len(puntos) == 4:                # la deteccion automatica puede dar el orden al reves
            puntos.reverse(); giro = 0
        if k == ord("a"):
            ext = esquinas_automaticas(frame)
            print("no encontre el tablero (tiene que estar vacio y entero en la imagen)" if ext is None
                  else "tablero encontrado: gira con g hasta que a1 quede en su lugar y guarda con s")
            if ext:
                puntos[:] = ext; giro = 0
        if k == ord("s") and len(puntos) == 4:
            Calibracion(orden).guardar(salida, frame.shape[1], frame.shape[0], str(camara))
            print("guardado en", salida)
            break
    cap.release()
    cv2.destroyAllWindows()


# =============================================================== medir cambios
R = 4                       # el mapa de cambios se trabaja a 1/4 de resolucion (casilla de 20 px): rapido
SR = S // R


def lab(warped):
    return cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2LAB), (5, 5), 0).astype(np.float32)


def mapa_cambio(ref_lab, cur_lab, umbral):
    """Cambio de color entre dos imagenes quietas, en [0, 1] por pixel. Se resta la mediana para que un cambio
    de luz general no cuente; por debajo de `umbral` es ruido, desde 2 x umbral es cambio seguro."""
    d = np.abs(cur_lab - ref_lab).sum(axis=2)
    d = cv2.resize(d, (8 * SR, 8 * SR), interpolation=cv2.INTER_AREA)
    d -= np.median(d)
    return np.clip((d - umbral) / umbral, 0, 1)


class Huellas:
    """Donde se ve cada casilla en la imagen enderezada. Una pieza alta no queda adentro de su casilla: desde una
    camara inclinada su cuerpo se proyecta hacia atras (lejos de la camara) y tapa parte de la casilla de atras.
    La huella de una casilla es la casilla mas esa franja. `alto` = cuantas casillas se estira (0 con la camara
    justo arriba del tablero; 0,8 a 1,2 con una camara a 45-55 grados)."""

    def __init__(self, cerca, alto=0.9):
        self.mascaras = np.zeros((64, 8 * SR, 8 * SR), bool)
        dx, dy = -cerca[0], -cerca[1]                         # hacia atras = lejos de la camara
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            for t in np.linspace(0, alto, 7):
                x0, y0 = int(round((f + dx * t) * SR)), int(round((7 - r + dy * t) * SR))
                self.mascaras[sq, max(y0, 0):max(y0 + SR, 0), max(x0, 0):max(x0 + SR, 0)] = True
        self.propia = np.zeros_like(self.mascaras)
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            self.propia[sq, (7 - r) * SR:(8 - r) * SR, f * SR:(f + 1) * SR] = True

        # casillas "de atras" de cada casilla: las que su huella pisa (ahi puede verse un cambio de la jugada)
        area = SR * SR
        self.detras = [{q for q in chess.SQUARES if q != sq and (self.mascaras[sq] & self.propia[q]).sum() > 0.15 * area}
                       for sq in chess.SQUARES]

    def por_casilla(self, c):
        """Cambio promedio de cada casilla (para mostrar)."""
        return np.array([c[self.propia[q]].mean() for q in chess.SQUARES], np.float32)


def casillas_de(board, m):
    """Casillas que cambian de aspecto con la jugada m."""
    cs = {m.from_square, m.to_square}
    if board.is_castling(m):
        rank = chess.square_rank(m.from_square)
        if chess.square_file(m.to_square) == 6:
            cs |= {chess.square(7, rank), chess.square(5, rank)}
        else:
            cs |= {chess.square(0, rank), chess.square(3, rank)}
    elif board.is_en_passant(m):
        cs.add(chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)))
    return cs


def inferir_jugada(board, c, huellas, minimo=0.08, margen=0.05):
    """Elige la jugada legal que mejor explica el mapa de cambios c.
    s[q] = cambio promedio dentro de la casilla q. Para una jugada M (sus 2 a 4 casillas):
        puntaje = min(s en M)  -  max(s fuera de M y de las casillas de atras de M)
    o sea: todas sus casillas tienen que haber cambiado, y fuera de su zona no tiene que haber cambiado nada.
    Las casillas de atras se perdonan porque ahi se ve la parte alta de las piezas (camara inclinada).
    Devuelve (jugada o None, puntaje, mensaje)."""
    s = huellas.por_casilla(c)
    puntajes, vistos = [], set()
    for m in board.legal_moves:
        if m.promotion and m.promotion != chess.QUEEN:      # la coronacion se asume a dama
            continue
        cs = frozenset(casillas_de(board, m))
        if cs in vistos:
            continue
        vistos.add(cs)
        permitidas = set(cs).union(*(huellas.detras[q] for q in cs))
        dentro = min(s[q] for q in cs)
        fuera = max((s[q] for q in chess.SQUARES if q not in permitidas), default=0.0)
        puntajes.append((dentro - fuera, dentro, m))
    if not puntajes:
        return None, 0.0, "no hay jugadas legales"
    puntajes.sort(key=lambda t: -t[0])
    puntaje, dentro, m = puntajes[0]
    segundo = puntajes[1][0] if len(puntajes) > 1 else -1.0
    cambiadas = ", ".join(chess.square_name(q) for q in np.argsort(-s)[:4] if s[q] > minimo) or "nada"
    if dentro < minimo or puntaje < margen:
        return None, puntaje, f"no coincide con ninguna jugada legal (cambiaron: {cambiadas})"
    if puntaje - segundo < 0.02:
        return None, puntaje, f"dudo entre {board.san(m)} y {board.san(puntajes[1][2])} (cambiaron: {cambiadas})"
    return m, puntaje, "ok"


# =============================================================== estabilidad
class Estabilidad:
    """Avisa cuando la imagen queda quieta despues de un movimiento (la mano salio del tablero)."""

    def __init__(self, umbral_mov=2.5, umbral_quieto=1.2, cuadros_quieto=8):
        self.um, self.uq, self.n = umbral_mov, umbral_quieto, cuadros_quieto
        self.prev, self.hubo_mov, self.quietos = None, False, 0
        self.actividad = 0.0

    def actualizar(self, warped):
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(warped, (160, 160)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        g = g.astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6) * 50.0          # que un cambio de brillo o contraste general no cuente
        if self.prev is None:
            self.prev = g
            return False
        self.actividad = float(np.abs(g - self.prev).mean())
        self.prev = g
        if self.actividad > self.um:
            self.hubo_mov, self.quietos = True, 0
        elif self.actividad < self.uq:
            self.quietos += 1
        if self.hubo_mov and self.quietos >= self.n:
            self.hubo_mov, self.quietos = False, 0
            return True
        return False


# =============================================================== el bot (opcional)
class Bot:
    def __init__(self, ck, depth):
        sys.path.insert(0, os.path.dirname(AQUI))
        from model import load_checkpoint, default_device
        from search import search, make_evaluator, pick_move
        dev = default_device()
        model, _ = load_checkpoint(ck, dev)
        self.ev, self.search, self.pick, self.depth = make_evaluator(model, dev), search, pick_move, depth
        self.valor = 0.0

    def jugar(self, board):
        try:
            res = self.search(board, self.ev, self.depth, 4, mates=True, poda=self.depth == 2)
        except TypeError:                                     # version vieja de search.py
            res = self.search(board, self.ev, self.depth, 4)
        sorpresa, self.valor = res.value - self.valor, res.value
        return self.pick(res.moves, 0.0), res.value, sorpresa


# =============================================================== dibujo
def dibujar_grilla(img, calib, color=(0, 255, 0)):
    for i in range(9):
        for a, b in (((i * S, 0), (i * S, LADO)), ((0, i * S), (LADO, i * S))):
            p, q = calib.a_camara(*a), calib.a_camara(*b)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1)
    p = calib.a_camara(0.5 * S, 7.5 * S)
    cv2.putText(img, "a1", (int(p[0]) - 10, int(p[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def vista_tablero(warped, s, ultima):
    v = warped.copy()
    if s is not None:
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            t = float(np.clip(s[sq], 0, 1))
            if t > 0.05:
                capa = v.copy()
                cv2.rectangle(capa, (f * S, (7 - r) * S), ((f + 1) * S, (8 - r) * S), (0, 0, 255), -1)
                v = cv2.addWeighted(capa, 0.6 * t, v, 1 - 0.6 * t, 0)
    if ultima is not None:
        for sq in (ultima.from_square, ultima.to_square):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            cv2.rectangle(v, (f * S + 2, (7 - r) * S + 2), ((f + 1) * S - 2, (8 - r) * S - 2), (0, 255, 0), 3)
    return v


def abrir_camara(fuente):
    if str(fuente).isdigit() and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(int(fuente), cv2.CAP_DSHOW)   # en Windows DirectShow abre mas rapido y mas estable
    else:
        cap = cv2.VideoCapture(int(fuente) if str(fuente).isdigit() else fuente)
    if not cap.isOpened():
        sys.exit(f"no pude abrir la camara o el video: {fuente}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def listar_camaras(maximo=6):
    """Prueba las camaras 0, 1, 2, ... y muestra una imagen de cada una para ver cual es la webcam."""
    encontradas = []
    for i in range(maximo):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            for _ in range(5):                                 # las primeras imagenes suelen salir oscuras
                ok2, f2 = cap.read()
                frame = f2 if ok2 else frame
            encontradas.append(i)
            print(f"camara {i}: {frame.shape[1]}x{frame.shape[0]}")
            cv2.putText(frame, f"camara {i}  (cualquier tecla: siguiente)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.imshow("camaras", frame)
            cv2.waitKey(0)
        cap.release()
    cv2.destroyAllWindows()
    if not encontradas:
        print("no encontre ninguna camara")
    else:
        print(f"\nUsa la que corresponda con --camara N (por ejemplo --camara {encontradas[-1]}). "
              "Se guarda en la calibracion, despues no hace falta repetirla.")


# =============================================================== bucle principal
class Seguidor:
    """Lleva la partida: recibe imagenes y devuelve eventos (jugadas detectadas, avisos)."""

    def __init__(self, calib, board=None, bot=None, color_bot=None, umbral=15.0, alto=0.9, estab=None):
        self.calib, self.board = calib, board or chess.Board()
        self.bot, self.color_bot = bot, color_bot
        self.umbral = umbral
        self.huellas = Huellas(calib.cerca, alto)
        self.estab = estab or Estabilidad()
        self.ref, self.s, self.ultima = None, None, None
        self.esperando = None                                  # jugada del bot que la persona tiene que hacer
        self.mensaje = "Toma la referencia con el tablero quieto"
        self.aviso = None                                      # error de deteccion a mostrar (en rojo)
        self.valor = self.sorpresa = None
        self.version = 0                                       # sube cada vez que cambia algo para mostrar

    def turno_bot(self):
        if self.bot is None or self.board.is_game_over() or self.board.turn != self.color_bot:
            return
        m, v, td = self.bot.jugar(self.board)
        self.esperando, self.valor, self.sorpresa = m, v, td
        self.version += 1
        self.mensaje = f"BOT: {self.board.san(m)}  (chance {round((v + 1) * 50)} %, sorpresa {td:+.2f}). Movela vos."
        print(self.mensaje)

    def procesar(self, frame, forzar=False):
        warped = self.calib.enderezar(frame)
        quieto = self.estab.actualizar(warped)
        if self.ref is None:
            if self.estab.prev is not None and self.estab.actividad < self.estab.uq:
                self.ref = lab(warped)
                self.mensaje = "Listo. Juegan las " + ("blancas" if self.board.turn else "negras")
                self.version += 1
                print(self.mensaje)
                self.turno_bot()
            return warped, None
        if not (quieto or forzar):
            return warped, None
        cur = lab(warped)
        c = mapa_cambio(self.ref, cur, self.umbral)
        self.s = self.huellas.por_casilla(c)
        if c.sum() < 0.3 * SR * SR:                            # no cambio nada (la mano paso de largo)
            self.ref = cur
            return warped, None
        m, puntaje, msg = inferir_jugada(self.board, c, self.huellas)
        if m is None:
            self.mensaje = "?? " + msg + ". Corregi el tablero."
            self.aviso = msg[0].upper() + msg[1:] + ". Corregi el tablero."
            self.version += 1
            print(self.mensaje)
            return warped, None
        san = self.board.san(m)
        if self.esperando is not None and m != self.esperando:
            self.mensaje = f"Esa no es la jugada del bot: hiciste {san}, era {self.board.san(self.esperando)}"
            self.aviso = self.mensaje + ". Volve la pieza y hace la del bot."
            self.version += 1
            print(self.mensaje)
            return warped, None
        self.board.push(m)
        self.ref, self.ultima, self.esperando, self.aviso = cur, m, None, None
        self.version += 1
        self.mensaje = f"{len(self.board.move_stack)}. {san}  (seguridad {puntaje:.2f})"
        print(self.mensaje)
        if self.board.is_game_over():
            self.mensaje = "Fin de la partida: " + self.board.result()
            print(self.mensaje)
        else:
            self.turno_bot()
        return warped, m

    def deshacer(self, frame):
        if self.board.move_stack:
            self.board.pop()
        self.ref, self.esperando, self.ultima = lab(self.calib.enderezar(frame)), None, None
        self.mensaje = "Deshice la ultima jugada; acomoda el tablero y apreta r"
        self.aviso = None
        self.version += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", default=None,
                    help="indice de la camara (0, 1, 2...). Por defecto la de la calibracion, o 0. Ver --listar-camaras")
    ap.add_argument("--listar-camaras", action="store_true", help="muestra una imagen de cada camara con su numero")
    ap.add_argument("--video", default=None, help="probar con un video grabado en vez de la camara")
    ap.add_argument("--calibrar", action="store_true")
    ap.add_argument("--calibracion", default=CALIB)
    ap.add_argument("--fen", default=None, help="posicion inicial (por defecto la de siempre)")
    ap.add_argument("--ck", default=None, help="checkpoint del bot para jugar contra el")
    ap.add_argument("--bot", choices=["blancas", "negras"], default="negras")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--umbral", type=float, default=15.0, help="cambio de color que se considera ruido (subir si hay falsos cambios)")
    ap.add_argument("--alto", type=float, default=0.9,
                    help="cuantas casillas 'hacia atras' tapa una pieza: 0 camara arriba del tablero, ~0,9 camara a 50 grados")
    ap.add_argument("--quieto", type=int, default=8, help="cuadros quietos para dar la jugada por terminada")
    ap.add_argument("--girar", action="store_true", help="mostrar el tablero digital con las negras abajo")
    ap.add_argument("--depurar", action="store_true", help="muestra tambien el tablero enderezado con los cambios")
    args = ap.parse_args()
    if args.listar_camaras:
        listar_camaras()
        return
    if args.calibrar:
        calibrar(args.video or args.camara or "0", args.calibracion)
        return
    if not os.path.exists(args.calibracion):
        sys.exit("falta la calibracion: corre primero con --calibrar")
    calib, (ancho_cal, alto_cal) = Calibracion.cargar(args.calibracion)
    fuente = args.video or args.camara or calib.camara or "0"
    print("usando la camara", fuente)
    bot = Bot(args.ck, args.depth) if args.ck else None
    seg = Seguidor(calib, chess.Board(args.fen) if args.fen else None, bot,
                   chess.WHITE if args.bot == "blancas" else chess.BLACK, args.umbral, args.alto,
                   Estabilidad(cuadros_quieto=args.quieto))
    cap = abrir_camara(fuente)
    ok, prueba = cap.read()
    if ok and (prueba.shape[1], prueba.shape[0]) != (ancho_cal, alto_cal):
        print(f"OJO: esta camara da {prueba.shape[1]}x{prueba.shape[0]} y la calibracion se hizo con "
              f"{ancho_cal}x{alto_cal}. Si es otra camara, recalibra con --calibrar --camara N")
    from vision.tablero_digital import TableroDigital, emocion
    digital = TableroDigital()
    girar = args.girar or (bot is not None and args.bot == "blancas")
    version_dibujada = -1
    t0, cuadros = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cuadros += 1
        k = cv2.waitKey(1) & 0xFF
        warped, _ = seg.procesar(frame, forzar=(k == ord(" ")))
        if k == ord("q"):
            break
        if k == ord("r"):
            seg.ref = lab(warped); seg.mensaje = "Referencia tomada"; seg.aviso = None; seg.version += 1
        if k == ord("u"):
            seg.deshacer(frame)
        if k == ord("l"):                                      # abrir la posicion en el analisis de Lichess
            webbrowser.open("https://lichess.org/analysis/standard/" + seg.board.fen().replace(" ", "_"))
        vista = frame.copy()
        dibujar_grilla(vista, calib)
        fps = cuadros / max(time.time() - t0, 1e-6)
        cv2.putText(vista, f"movimiento {seg.estab.actividad:4.1f} | {fps:4.1f} fps", (15, vista.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("camara", vista)
        if seg.version != version_dibujada:                    # el tablero digital se redibuja solo si cambio algo
            emo = emocion(seg.valor, seg.sorpresa) if seg.valor is not None else None
            cv2.imshow("partida", digital.dibujar(seg.board, seg.ultima, seg.esperando, girar,
                                                  estado=seg.mensaje, valor=seg.valor, emocion=emo,
                                                  sorpresa=seg.sorpresa, aviso=seg.aviso))
            version_dibujada = seg.version
        if args.depurar:
            cv2.imshow("tablero enderezado", vista_tablero(warped, seg.s, seg.ultima))
    cap.release()
    cv2.destroyAllWindows()
    juego = chess.pgn.Game.from_board(seg.board)
    with open(os.path.join(AQUI, "ultima_partida.pgn"), "w") as f:
        print(juego, file=f)
    print("\nPartida:", seg.board.fen())
    print(juego.mainline_moves())


if __name__ == "__main__":
    main()
