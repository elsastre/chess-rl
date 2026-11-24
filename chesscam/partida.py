"""
partida.py - La partida contra el bot, sin nada de interfaz grafica (se puede probar sola).

Dos modos:
  * Con camara (tablero fisico): el bot NO mueve en la pantalla. Su jugada queda "pendiente" y se muestra
    con una flecha para que la persona la haga en el tablero real. Recien cuando la camara (o un clic en la
    pantalla) confirma esa jugada, se aplica.
  * Solo simulador: el bot juega directamente en la pantalla y su flecha queda marcada hasta que la
    persona responde, para que se vea que hizo.

El bot es la red del proyecto con la busqueda de search.py (la misma que usa el robot).
"""
import os
import queue
import sys
import threading
import time

import chess

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.append(RAIZ)        # al final: asi chesscam/ tiene prioridad y no se confunde con la carpeta vision/

NOMBRES = {chess.PAWN: "Peón", chess.KNIGHT: "Caballo", chess.BISHOP: "Alfil", chess.ROOK: "Torre",
           chess.QUEEN: "Dama", chess.KING: "Rey"}
SAN_ES = str.maketrans("NBRQK", "CATDR")


def san_es(board, mv):
    """Notacion algebraica con las letras en español (C, A, T, D, R)."""
    return board.san(mv).translate(SAN_ES)


def describir(board, mv):
    """La jugada contada para hacerla a mano en el tablero fisico. board: la posicion ANTES de la jugada."""
    de, a = chess.square_name(mv.from_square), chess.square_name(mv.to_square)
    if board.is_castling(mv):
        fila = chess.square_rank(mv.from_square)
        corto = chess.square_file(mv.to_square) == 6
        t_de, t_a = (chess.square(7, fila), chess.square(5, fila)) if corto else (chess.square(0, fila), chess.square(3, fila))
        txt = (f"Enroque {'corto' if corto else 'largo'}: rey de {de} a {a} "
               f"y torre de {chess.square_name(t_de)} a {chess.square_name(t_a)}")
    elif board.is_en_passant(mv):
        comido = chess.square_name(chess.square(chess.square_file(mv.to_square), chess.square_rank(mv.from_square)))
        txt = f"Peón de {de} come al paso en {a} (sacá el peón de {comido})"
    else:
        pieza = NOMBRES[board.piece_type_at(mv.from_square)]
        txt = f"{pieza} de {de} " + (f"come en {a}" if board.is_capture(mv) else f"a {a}")
        if mv.promotion:
            txt += f" y corona {NOMBRES[mv.promotion].lower()}"
    board.push(mv)
    if board.is_checkmate():
        txt += ". ¡Jaque mate!"
    elif board.is_check():
        txt += ", jaque"
    board.pop()
    return txt


def emocion(valor, sorpresa):
    """Las mismas emociones que la interfaz y el robot (CLAUDE.md 3.6)."""
    if sorpresa is not None and abs(sorpresa) > 0.16:
        return "sorprendido"
    p = (valor + 1) / 2
    return "ganando" if p >= .75 else "confiado" if p >= .5 else "preocupado" if p >= .3 else "perdiendo"


class Reloj:
    """Reloj de ajedrez con incremento (Fischer): corre el del bando que tiene que mover; al completar una
    jugada se le suma el incremento y pasa a correr el del rival. `ahora` se puede reemplazar en las pruebas."""

    def __init__(self, base, incremento=0.0, ahora=time.monotonic):
        self.base, self.incremento, self.ahora = float(base), float(incremento), ahora
        self.reiniciar()

    def reiniciar(self):
        self.restante = {chess.WHITE: self.base, chess.BLACK: self.base}
        self.corre = None            # bando cuyo reloj corre (o correria si no estuviera en pausa)
        self.desde = None            # momento desde el que se descuenta (None = detenido)
        self.iniciado = False

    def _descontar(self):
        if self.corre is not None and self.desde is not None:
            t = self.ahora()
            self.restante[self.corre] -= t - self.desde
            self.desde = t

    @property
    def pausado(self):
        return self.iniciado and self.desde is None

    def tiempo(self, color):
        self._descontar()
        return max(0.0, self.restante[color])

    def iniciar(self, color):
        """Arranca (o retoma) el reloj de `color`."""
        self._descontar()
        self.corre, self.desde, self.iniciado = color, self.ahora(), True

    def pausar(self):
        self._descontar()
        self.desde = None

    def reanudar(self):
        if self.iniciado and self.desde is None and self.corre is not None:
            self.desde = self.ahora()

    def jugo(self, color):
        """`color` completo su jugada (se aprieta el reloj): incremento y pasa a correr el del rival."""
        if not self.iniciado:                # primera jugada antes de arrancar: arranca el rival
            self.iniciar(not color)
            return
        self._descontar()
        corriendo = self.desde is not None
        if self.restante[color] > 0:
            self.restante[color] += self.incremento
        self.corre = not color
        self.desde = self.ahora() if corriendo else None

    def agotado(self):
        """El bando que se quedo sin tiempo, o None."""
        self._descontar()
        if self.corre is not None and self.restante[self.corre] <= 0:
            return self.corre
        return None

    def pasar_a(self, color):
        """Despues de deshacer: corre el reloj de `color`, sin tocar los tiempos."""
        self._descontar()
        corriendo = self.desde is not None
        self.corre = color
        self.desde = self.ahora() if corriendo else None


def formato_tiempo(seg):
    """5:00, 0:42, 0:09.4 (decimas en los ultimos 10 s), 1:05:00."""
    if seg < 10:
        return f"0:{seg:04.1f}"
    seg = int(seg)
    h, m, s = seg // 3600, seg % 3600 // 60, seg % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def leer_control(txt):
    """'5+3' -> (300 s, 3 s); '10' -> (600 s, 0 s); '0.5+2' -> (30 s, 2 s). Minutos + segundos de incremento."""
    try:
        base, _, inc = txt.partition("+")
        base, inc = float(base) * 60, float(inc or 0)
    except ValueError:
        raise ValueError(f"control de tiempo invalido: {txt!r} (ejemplos: 5+3, 10, 3+2)")
    if base <= 0 or inc < 0:
        raise ValueError(f"control de tiempo invalido: {txt!r}")
    return base, inc


class Bot:
    """La red del proyecto + busqueda. jugar(board) -> (jugada, valor para el bot en [-1, 1])."""

    def __init__(self, ck, depth=2):
        from model import load_checkpoint, default_device
        from search import search, make_evaluator, pick_move
        dev = default_device()
        model, _ = load_checkpoint(ck, dev)
        self.ev, self.search, self.pick, self.depth = make_evaluator(model, dev), search, pick_move, depth
        self.nombre = os.path.basename(ck)

    def jugar(self, board):
        try:
            res = self.search(board, self.ev, self.depth, 4, mates=True, poda=self.depth == 2)
        except TypeError:                                    # search.py viejo, sin mates ni poda
            res = self.search(board, self.ev, self.depth, 4)
        return self.pick(res.moves, 0.0), res.value


class Partida:
    def __init__(self, bot=None, color_bot=chess.BLACK, con_camara=True, reloj=None, pantalla=None):
        self.board = chess.Board()
        self.pantalla = pantalla     # display LED de los ojos (pantalla_led.PantallaLED) o None
        self.reloj = reloj           # Reloj o None (sin reloj)
        self.tiempo_agotado = None   # bando que perdio por tiempo (o empato: ver resultado())
        self.bot, self.color_bot, self.con_camara = bot, color_bot, con_camara
        self.pendiente = None        # con camara: jugada del bot que la persona tiene que hacer en el tablero
        self.flecha_bot = None       # jugada del bot a dibujar (pendiente o la ultima que hizo)
        self.pista = None            # jugada sugerida a la persona (otra flecha)
        self.valor = self.sorpresa = None
        self.texto_bot = ""
        self.pensando = False
        self._cola = queue.Queue()
        self._version = 0            # para descartar respuestas viejas del bot (deshacer, nueva partida)

    # ---------------------------------------------------------------- turnos
    def terminada(self):
        return self.board.is_game_over() or self.tiempo_agotado is not None

    def revisar_reloj(self):
        """Llamar seguido. Si a alguien se le termino el tiempo, termina la partida y devuelve ese bando."""
        if self.reloj is None or self.tiempo_agotado is not None or self.board.is_game_over():
            return None
        c = self.reloj.agotado()
        if c is not None:
            self.tiempo_agotado = c
            self.reloj.pausar()
        return c

    def turno_del_bot(self):
        return (self.bot is not None and not self.terminada()
                and self.board.turn == self.color_bot and self.pendiente is None)

    def pensar(self, pista=False):
        """Lanza la busqueda en un hilo (la ventana no se congela). pista=True: sugerencia para la persona."""
        if self.bot is None or self.pensando or self.terminada():
            return False
        self.pensando = True
        if self.pantalla is not None and not pista:
            self.pantalla.pensando()
        b, v = self.board.copy(), self._version

        def trabajo():
            try:
                mv, valor = self.bot.jugar(b)
                self._cola.put((v, pista, mv, valor, None))
            except Exception as e:                       # que un error no deje "pensando" para siempre
                self._cola.put((v, pista, None, None, e))
        threading.Thread(target=trabajo, daemon=True).start()
        return True

    def revisar(self):
        """Llamar seguido desde la interfaz. Devuelve un evento o None:
        ('bot', jugada) | ('pista', jugada) | ('error', excepcion)."""
        try:
            v, pista, mv, valor, err = self._cola.get_nowait()
        except queue.Empty:
            return None
        self.pensando = False
        if v != self._version or self.terminada():       # respuesta de una posicion que ya no existe
            return None
        if err is not None:
            return ("error", err)
        if pista:
            self.pista = mv
            return ("pista", mv)
        self.sorpresa = None if self.valor is None else valor - self.valor
        self.valor = valor
        self.flecha_bot = mv
        self.texto_bot = describir(self.board, mv)
        if self.pantalla is not None:                    # con el tablero de ANTES de la jugada
            self.pantalla.mostrar(emocion(self.valor, self.sorpresa), self.board, mv)
        if self.con_camara:
            self.pendiente = mv                          # la hace la persona en el tablero real: el reloj del
        else:                                            # bot sigue corriendo hasta que la camara la confirma
            self.board.push(mv)                          # en el simulador juega solo
            self._reloj_jugo(not self.board.turn)
        return ("bot", mv)

    def _reloj_jugo(self, color):
        if self.reloj is not None:
            self.reloj.jugo(color)

    # ---------------------------------------------------------------- jugadas
    def jugada(self, mv):
        """Una jugada que llega de la camara o de un clic. Devuelve (aplicada, mensaje).
        Si se aplica, se aprieta el reloj de quien jugo (incremento y pasa a correr el del rival)."""
        if self.terminada():
            return False, "La partida terminó."
        if mv not in self.board.legal_moves:
            return False, "Jugada ilegal."
        if self.pendiente is not None:
            if mv != self.pendiente:
                return False, (f"Esa no es la jugada del bot: hiciste {san_es(self.board, mv)}, "
                               f"era {san_es(self.board, self.pendiente)}. Volvé la pieza y hacé la del bot.")
            self.board.push(mv)
            self._reloj_jugo(not self.board.turn)
            self.pendiente, self.pista = None, None
            return True, f"Hecha la jugada del bot: {self.texto_bot}."
        if self.bot is not None and self.board.turn == self.color_bot:
            return False, "Le toca al bot: esperá su jugada."
        san = san_es(self.board, mv)
        self.board.push(mv)
        self._reloj_jugo(not self.board.turn)
        self.flecha_bot, self.pista = None, None
        self._version += 1
        if self.pantalla is not None and self.bot is not None and self.board.is_checkmate():
            self.pantalla.poner_emocion("perdiendo")     # le dieron mate al bot: no llega a pensar
        return True, san

    def jugada_pendiente_hecha(self):
        """Botón: la persona hizo la jugada del bot en el tablero y la camara no la vio."""
        if self.pendiente is None:
            return False, "No hay ninguna jugada del bot pendiente."
        return self.jugada(self.pendiente)

    def deshacer(self):
        """Vuelve al ultimo turno de la persona (deshace tambien la respuesta del bot)."""
        self._version += 1
        self.pensando = False
        if self.pendiente is not None:                   # el bot propuso pero nadie la hizo: solo se descarta
            self.pendiente = None
        if self.board.move_stack:
            self.board.pop()
        if self.bot is not None:
            while self.board.move_stack and self.board.turn == self.color_bot:
                self.board.pop()
        self.flecha_bot, self.pista = None, None
        self.valor = self.sorpresa = None
        self.texto_bot = ""
        self.tiempo_agotado = None                       # se puede deshacer una derrota por tiempo
        if self.pantalla is not None:
            self.pantalla.pensando(borrar_jugada=True)
        if self.reloj is not None and self.reloj.iniciado:
            self.reloj.pasar_a(self.board.turn)          # los tiempos quedan como estaban

    def nueva(self):
        self.board.reset()
        self._version += 1
        self.pensando = False
        self.pendiente = self.flecha_bot = self.pista = None
        self.valor = self.sorpresa = None
        self.texto_bot = ""
        self.tiempo_agotado = None
        if self.pantalla is not None:
            self.pantalla.pensando(borrar_jugada=True)
        if self.reloj is not None:
            self.reloj.reiniciar()                      # arranca cuando la interfaz diga que esta lista

    # ---------------------------------------------------------------- para mostrar
    def resultado(self):
        """Si la partida termino: (titulo, detalle, tipo) con tipo 'gane' | 'perdi' | 'tablas' | 'fin'
        (desde la persona que juega contra el bot; 'fin' si son dos personas). Si sigue: None."""
        if self.tiempo_agotado is not None and not self.board.is_game_over():
            return self._resultado_tiempo()
        o = self.board.outcome()
        if o is None:
            return None
        T = chess.Termination
        motivo = {T.CHECKMATE: "¡Jaque mate!", T.STALEMATE: "Tablas por ahogado",
                  T.INSUFFICIENT_MATERIAL: "Tablas por material insuficiente",
                  T.SEVENTYFIVE_MOVES: "Tablas por la regla de las 75 jugadas",
                  T.FIVEFOLD_REPETITION: "Tablas por repetición"}.get(o.termination, "Fin de la partida")
        res = self.board.result()
        if o.winner is None:
            return motivo, "Empate (½-½)", "tablas"
        if self.bot is None:
            return motivo, f"Ganan las {'blancas' if o.winner == chess.WHITE else 'negras'} ({res})", "fin"
        if o.winner == self.color_bot:
            return motivo, f"Ganó el bot ({res})", "perdi"
        return motivo, f"¡Ganaste! ({res})", "gane"

    def _resultado_tiempo(self):
        """Reglas de la FIDE: pierde quien se quedo sin tiempo, salvo que al rival no le alcance el material para
        dar mate de ninguna manera: entonces son tablas."""
        c = self.tiempo_agotado
        quien = "las blancas" if c == chess.WHITE else "las negras"
        if self.board.has_insufficient_material(not c):
            return f"Se terminó el tiempo de {quien}", "Tablas: al rival no le alcanza el material para dar mate (½-½)", "tablas"
        res = "0-1" if c == chess.WHITE else "1-0"
        if self.bot is None:
            return f"Se terminó el tiempo de {quien}", f"Ganan las {'negras' if c == chess.WHITE else 'blancas'} ({res})", "fin"
        if c == self.color_bot:
            return "Se terminó el tiempo del bot", f"¡Ganaste! ({res})", "gane"
        return "Se terminó tu tiempo", f"Ganó el bot ({res})", "perdi"

    def resultado_pgn(self):
        """'1-0', '0-1', '1/2-1/2' o '*', incluida la derrota por tiempo."""
        if self.tiempo_agotado is not None and not self.board.is_game_over():
            if self.board.has_insufficient_material(not self.tiempo_agotado):
                return "1/2-1/2"
            return "0-1" if self.tiempo_agotado == chess.WHITE else "1-0"
        return self.board.result()

    def estado_bot(self):
        """Una linea para el panel: que hace el bot y como se siente."""
        if self.bot is None:
            return ""
        lado = "blancas" if self.color_bot == chess.WHITE else "negras"
        if self.pensando:
            return f"Bot ({lado}): pensando…"
        if self.valor is None:
            return f"Bot ({lado}): {self.bot.nombre}"
        chance = round((self.valor + 1) * 50)
        emo = emocion(self.valor, self.sorpresa)
        return f"Bot ({lado}): chance {chance} %, {emo}"
