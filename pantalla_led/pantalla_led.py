"""Envia la emocion y la jugada del bot al display LED de 32x32 (ATmega328p con pantalla_led.ino).

Protocolo (una orden por linea, el micro responde OK o ERR):
    E:<emocion>   pensando | confiado | ganando | preocupado | sorprendido | perdiendo
    M:<texto>     jugada, hasta 5 caracteres (e2-e4, e4xd5, O-O, e7e8D); "M:" la borra

Uso desde el bot:
    from pantalla_led.pantalla_led import PantallaLED
    led = PantallaLED("COM3")            # una sola vez al arrancar (abrir el puerto reinicia el micro)
    led.pensando()                       # cuando el bot empieza a pensar
    led.mostrar(emocion, board, move)    # cuando eligio (board ANTES de hacer la jugada)

Si no hay display conectado (o falta pyserial) avisa una vez y sigue: el bot nunca se frena por esto.

Prueba sola:
    python pantalla_led/pantalla_led.py --puertos                 # lista los puertos serie
    python pantalla_led/pantalla_led.py --puerto COM3 --demo      # recorre todas las emociones
    python pantalla_led/pantalla_led.py --puerto COM3 E:ganando M:e2-e4
"""
import argparse
import time

EMOCIONES = ("pensando", "confiado", "ganando", "preocupado", "sorprendido", "perdiendo")
CORONACION = {5: "D", 4: "T", 3: "A", 2: "C"}      # chess.QUEEN, ROOK, BISHOP, KNIGHT en letras espanolas


def texto_jugada(board, move):
    """Jugada en el formato corto que entra en el display (hasta 5 caracteres).
    board es la posicion ANTES de la jugada (python-chess)."""
    if board.is_castling(move):
        return "O-O" if move.to_square > move.from_square else "O-O-O"
    import chess
    desde, hasta = chess.square_name(move.from_square), chess.square_name(move.to_square)
    if move.promotion:
        return desde + hasta + CORONACION.get(move.promotion, "D")
    return desde + ("x" if board.is_capture(move) else "-") + hasta


class PantallaLED:
    def __init__(self, puerto, baudios=9600, espera=2.0):
        self.ser = None
        try:
            import serial
            self.ser = serial.serial_for_url(puerto, baudios, timeout=0.2, write_timeout=0.5)
            time.sleep(espera)                 # al abrir el puerto el micro se reinicia (Arduino)
            self.ser.reset_input_buffer()
        except ImportError:
            print("Display LED: falta pyserial (pip install pyserial); sigo sin display.")
        except Exception as e:
            print(f"Display LED: no pude abrir {puerto} ({e}); sigo sin display.")
            self.ser = None
        self.emocion = self.jugada = None

    def enviar(self, orden, esperar=True):
        """Manda una linea. Con esperar=True devuelve la respuesta del micro ('OK', 'ERR') o None.
        Durante la partida se usa esperar=False: no frena la ventana esperando la respuesta."""
        if self.ser is None:
            return None
        try:
            if not esperar:
                self.ser.reset_input_buffer()          # descarta los OK que no se leyeron
            self.ser.write((orden + "\n").encode("ascii"))
            if not esperar:
                return None
            return self.ser.readline().decode("ascii", "replace").strip() or None
        except Exception as e:
            print(f"Display LED: error al enviar ({e}); sigo sin display.")
            self.ser = None
            return None

    def poner_emocion(self, emocion):
        if emocion in EMOCIONES and emocion != self.emocion:
            self.enviar("E:" + emocion, esperar=False)
            self.emocion = emocion

    def poner_jugada(self, texto):
        texto = (texto or "")[:5]
        if texto != self.jugada:
            self.enviar("M:" + texto, esperar=False)
            self.jugada = texto

    def pensando(self, borrar_jugada=False):
        self.poner_emocion("pensando")
        if borrar_jugada:
            self.poner_jugada("")

    def mostrar(self, emocion, board=None, move=None):
        self.poner_emocion(emocion)
        if board is not None and move is not None:
            self.poner_jugada(texto_jugada(board, move))

    def cerrar(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None


def main():
    ap = argparse.ArgumentParser(description="Prueba del display LED de emociones del bot")
    ap.add_argument("--puerto", help="puerto serie (Windows: COM3; Mac: /dev/cu.usbserial-...)")
    ap.add_argument("--baudios", type=int, default=9600)
    ap.add_argument("--puertos", action="store_true", help="lista los puertos serie disponibles")
    ap.add_argument("--demo", action="store_true", help="recorre todas las emociones con jugadas de ejemplo")
    ap.add_argument("ordenes", nargs="*", help="ordenes sueltas, por ejemplo E:ganando M:e2-e4")
    a = ap.parse_args()

    if a.puertos:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(p.device, "-", p.description)
        return
    if not a.puerto:
        ap.error("falta --puerto (usar --puertos para verlos)")

    led = PantallaLED(a.puerto, a.baudios)
    for o in a.ordenes:
        print(o, "->", led.enviar(o))
    if a.demo:
        ejemplos = ["", "e2-e4", "e4xd5", "O-O", "e7e8D", "g1-f3"]
        for emo, jug in zip(EMOCIONES, ejemplos):
            print(f"{emo:12s} {jug:6s}", led.enviar("E:" + emo), led.enviar("M:" + jug))
            time.sleep(4)
    led.cerrar()


if __name__ == "__main__":
    main()
