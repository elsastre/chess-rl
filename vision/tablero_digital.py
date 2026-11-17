"""
Tablero digital estilo Lichess para mostrar lo que detecta la camara (y la jugada que propone el bot).
Se dibuja con Pillow sobre una imagen y se muestra con OpenCV, asi todo corre en la misma ventana de cv2.
"""
import chess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CASILLA = 76
BORDE = 22
TABLERO = 8 * CASILLA
PANEL = 340
ANCHO, ALTO = BORDE + TABLERO + 16 + PANEL, BORDE * 2 + TABLERO

CLARA, OSCURA = (240, 217, 181), (181, 136, 99)              # colores de Lichess (tema "brown")
ULTIMA = (205, 210, 106)
JAQUE = (226, 75, 74)
FLECHA = (21, 120, 27)
FONDO, TINTA, TINTA2, ACENTO = (38, 36, 33), (235, 233, 228), (170, 166, 158), (98, 153, 36)
GLIFO = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜", chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
LETRA = {chess.KING: "R", chess.QUEEN: "D", chess.ROOK: "T", chess.BISHOP: "A", chess.KNIGHT: "C", chess.PAWN: "P"}
SAN_ES = str.maketrans("NBRQK", "CATDR")

_FUENTES_PIEZAS = ["seguisym.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Apple Symbols.ttf",
                   "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "FreeSerif.ttf"]
_FUENTES_TEXTO = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def _fuente(nombres, tam, probar=None):
    for n in nombres:
        try:
            f = ImageFont.truetype(n, tam)
            if probar is None or f.getbbox(probar)[2] > tam * 0.4:
                return f
        except OSError:
            continue
    return None


class TableroDigital:
    def __init__(self):
        self.f_pieza = _fuente(_FUENTES_PIEZAS, int(CASILLA * 0.82), probar="♞")
        self.f_texto = _fuente(_FUENTES_TEXTO, 17) or ImageFont.load_default()
        self.f_chica = _fuente(_FUENTES_TEXTO, 14) or ImageFont.load_default()
        self.f_grande = _fuente(_FUENTES_TEXTO, 24) or ImageFont.load_default()
        self.f_letra = _fuente(_FUENTES_TEXTO, int(CASILLA * 0.45)) or ImageFont.load_default()

    def _xy(self, sq, flip):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if flip:
            f, r = 7 - f, 7 - r
        return BORDE + f * CASILLA, BORDE + (7 - r) * CASILLA

    def _pieza(self, d, p, x, y):
        cx, cy = x + CASILLA / 2, y + CASILLA / 2
        relleno, borde = ((250, 250, 248), (20, 20, 20)) if p.color else ((20, 20, 20), (200, 200, 195))
        if self.f_pieza is not None:
            d.text((cx, cy + 2), GLIFO[p.piece_type], font=self.f_pieza, anchor="mm", fill=relleno,
                   stroke_width=2, stroke_fill=borde)
        else:                                                  # sin fuente de simbolos: disco con la letra
            r = CASILLA * 0.36
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=relleno, outline=borde, width=2)
            d.text((cx, cy), LETRA[p.piece_type], font=self.f_letra, anchor="mm", fill=borde)

    def _flecha(self, d, desde, hasta, flip):
        x0, y0 = self._xy(desde, flip)
        x1, y1 = self._xy(hasta, flip)
        a = np.array([x0 + CASILLA / 2, y0 + CASILLA / 2])
        b = np.array([x1 + CASILLA / 2, y1 + CASILLA / 2])
        v = b - a
        n = np.linalg.norm(v)
        if n < 1:
            return
        u, w = v / n, np.array([-v[1], v[0]]) / n
        punta = b - u * CASILLA * 0.15
        base = punta - u * CASILLA * 0.42
        d.line([tuple(a), tuple(base)], fill=FLECHA + (170,), width=int(CASILLA * 0.17))
        d.polygon([tuple(punta), tuple(base + w * CASILLA * 0.25), tuple(base - w * CASILLA * 0.25)],
                  fill=FLECHA + (170,))

    def dibujar(self, board, ultima=None, sugerida=None, flip=False, titulo="", estado="", valor=None,
                emocion=None, sorpresa=None, aviso=None):
        """Devuelve la imagen (BGR, para cv2.imshow) del tablero con el panel de la partida."""
        img = Image.new("RGB", (ANCHO, ALTO), FONDO)
        d = ImageDraw.Draw(img)
        for sq in chess.SQUARES:
            x, y = self._xy(sq, flip)
            clara = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            color = CLARA if clara else OSCURA
            if ultima is not None and sq in (ultima.from_square, ultima.to_square):
                color = tuple(int(0.45 * c + 0.55 * u) for c, u in zip(color, ULTIMA))
            d.rectangle((x, y, x + CASILLA - 1, y + CASILLA - 1), fill=color)
        if board.is_check():
            x, y = self._xy(board.king(board.turn), flip)
            d.ellipse((x + 4, y + 4, x + CASILLA - 4, y + CASILLA - 4), fill=JAQUE)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                self._pieza(d, p, *self._xy(sq, flip))
        for i in range(8):                                     # coordenadas como en Lichess
            col = "abcdefgh"[7 - i if flip else i]
            fila = str(i + 1 if flip else 8 - i)
            d.text((BORDE + i * CASILLA + CASILLA / 2, BORDE + TABLERO + 11), col, font=self.f_chica,
                   anchor="mm", fill=TINTA2)
            d.text((BORDE / 2, BORDE + i * CASILLA + CASILLA / 2), fila, font=self.f_chica, anchor="mm",
                   fill=TINTA2)
        if sugerida is not None:
            capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
            self._flecha(ImageDraw.Draw(capa), sugerida.from_square, sugerida.to_square, flip)
            img = Image.alpha_composite(img.convert("RGBA"), capa).convert("RGB")
            d = ImageDraw.Draw(img)

        # ---------------- panel
        px, y = BORDE + TABLERO + 16, BORDE
        d.text((px, y), titulo or "Robot de ajedrez RL", font=self.f_grande, fill=TINTA); y += 38
        turno = "Juegan las blancas" if board.turn else "Juegan las negras"
        if board.is_game_over():
            turno = "Fin: " + board.result()
        d.text((px, y), turno, font=self.f_texto, fill=ACENTO); y += 30
        for linea in _cortar(estado, 34)[:3]:
            d.text((px, y), linea, font=self.f_chica, fill=TINTA); y += 20
        if aviso:
            for linea in _cortar(aviso, 34)[:3]:
                d.text((px, y), linea, font=self.f_chica, fill=JAQUE); y += 20
        y += 8
        if valor is not None:
            p = (valor + 1) / 2
            d.text((px, y), f"Chance del bot: {round(p * 100)} %", font=self.f_texto, fill=TINTA); y += 26
            d.rectangle((px, y, px + PANEL - 30, y + 10), fill=(70, 66, 60))
            d.rectangle((px, y, px + int((PANEL - 30) * p), y + 10), fill=ACENTO); y += 22
            if emocion:
                extra = f"  (sorpresa {sorpresa:+.2f})" if sorpresa is not None else ""
                d.text((px, y), f"Emoción: {emocion}{extra}", font=self.f_chica, fill=TINTA2); y += 26
        d.text((px, y), "Jugadas", font=self.f_texto, fill=TINTA); y += 26
        copia = board.root()
        primera = copia.fullmove_number
        sans = ["..."] if copia.turn == chess.BLACK else []
        for m in board.move_stack:
            sans.append(copia.san(m).translate(SAN_ES))
            copia.push(m)
        pares = [(primera + i // 2, sans[i], sans[i + 1] if i + 1 < len(sans) else "") for i in range(0, len(sans), 2)]
        alto_lista = (ALTO - y - 60) // 22
        for n, b, nn in pares[-alto_lista:]:
            d.text((px, y), f"{n:>3}.", font=self.f_chica, fill=TINTA2)
            d.text((px + 42, y), b, font=self.f_chica, fill=TINTA)
            d.text((px + 140, y), nn, font=self.f_chica, fill=TINTA)
            y += 22
        d.text((px, ALTO - 44), "q salir  r referencia  u deshacer", font=self.f_chica, fill=TINTA2)
        d.text((px, ALTO - 24), "l abrir en Lichess  espacio detectar", font=self.f_chica, fill=TINTA2)
        return np.array(img)[:, :, ::-1].copy()


def _cortar(texto, n):
    palabras, lineas, actual = (texto or "").split(), [], ""
    for p in palabras:
        if len(actual) + len(p) + 1 > n and actual:
            lineas.append(actual); actual = p
        else:
            actual = (actual + " " + p).strip()
    if actual:
        lineas.append(actual)
    return lineas


def emocion(valor, sorpresa):
    if sorpresa is not None and abs(sorpresa) > 0.16:
        return "sorprendido"
    p = (valor + 1) / 2
    return "ganando" if p >= .75 else "confiado" if p >= .5 else "preocupado" if p >= .3 else "perdiendo"
