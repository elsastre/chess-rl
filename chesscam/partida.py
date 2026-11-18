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
    def __init__(self, bot=None, color_bot=chess.BLACK, con_camara=True):
        self.board = chess.Board()
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
    def turno_del_bot(self):
        return (self.bot is not None and not self.board.is_game_over()
                and self.board.turn == self.color_bot and self.pendiente is None)

    def pensar(self, pista=False):
        """Lanza la busqueda en un hilo (la ventana no se congela). pista=True: sugerencia para la persona."""
        if self.bot is None or self.pensando or self.board.is_game_over():
            return False
        self.pensando = True
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
        if v != self._version:                           # respuesta de una posicion que ya no existe
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
        if self.con_camara:
            self.pendiente = mv                          # la hace la persona en el tablero real
        else:
            self.board.push(mv)                          # en el simulador juega solo
        return ("bot", mv)

    # ---------------------------------------------------------------- jugadas
    def jugada(self, mv):
        """Una jugada que llega de la camara o de un clic. Devuelve (aplicada, mensaje)."""
        if self.board.is_game_over():
            return False, "La partida terminó."
        if mv not in self.board.legal_moves:
            return False, "Jugada ilegal."
        if self.pendiente is not None:
            if mv != self.pendiente:
                return False, (f"Esa no es la jugada del bot: hiciste {san_es(self.board, mv)}, "
                               f"era {san_es(self.board, self.pendiente)}. Volvé la pieza y hacé la del bot.")
            self.board.push(mv)
            self.pendiente, self.pista = None, None
            return True, f"Hecha la jugada del bot: {self.texto_bot}."
        if self.bot is not None and self.board.turn == self.color_bot:
            return False, "Le toca al bot: esperá su jugada."
        san = san_es(self.board, mv)
        self.board.push(mv)
        self.flecha_bot, self.pista = None, None
        self._version += 1
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

    def nueva(self):
        self.board.reset()
        self._version += 1
        self.pensando = False
        self.pendiente = self.flecha_bot = self.pista = None
        self.valor = self.sorpresa = None
        self.texto_bot = ""

    # ---------------------------------------------------------------- para mostrar
    def resultado(self):
        """Si la partida termino: (titulo, detalle, tipo) con tipo 'gane' | 'perdi' | 'tablas' | 'fin'
        (desde la persona que juega contra el bot; 'fin' si son dos personas). Si sigue: None."""
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
