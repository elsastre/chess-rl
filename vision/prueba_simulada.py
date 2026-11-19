"""
Prueba de detectar_jugadas.py sin camara: dibuja un tablero con piezas "altas" (que tapan parte de la casilla de
atras, como se ve desde una camara inclinada), lo proyecta como lo veria una webcam, le agrega ruido y una mano,
y juega una partida con enroques, capturas y al paso. Uso: python vision/prueba_simulada.py
"""
import os, sys, numpy as np, cv2, chess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # raiz del repo
import vision.detectar_jugadas as dj
rng = np.random.default_rng(0)
S, L = dj.S, dj.LADO
CAM = [[300, 650], [980, 650], [850, 180], [430, 180]]       # a1, h1, h8, a8 en la imagen de la camara (vista inclinada)
W, H = 1280, 720
calib = dj.Calibracion(CAM); calib.fijar_cercania(W, H)
# (relleno, borde) en BGR de las piezas blancas y negras; chesscam/prueba_robustez.py baja el contraste de las negras
COLOR_BLANCAS = ((235, 235, 230), (40, 40, 40))
COLOR_NEGRAS = ((35, 30, 30), (200, 200, 200))

def render(board, mano=None):
    top = np.zeros((L, L, 3), np.uint8)
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        col = (181, 217, 238) if (f + r) % 2 else (99, 136, 181)
        cv2.rectangle(top, (f * S, (7 - r) * S), ((f + 1) * S, (8 - r) * S), col, -1)
    # piezas "altas": la base en la casilla y el cuerpo se extiende hacia atras (tapa parte de la casilla de atras)
    cx_, cy_ = calib.cerca
    ang = np.degrees(np.arctan2(-cy_, -cx_))                 # el cuerpo se proyecta lejos de la camara
    def lejania(q):
        f, r = chess.square_file(q), chess.square_rank(q)
        return -((f + .5) * cx_ + (7 - r + .5) * cy_)
    for sq in sorted(chess.SQUARES, key=lambda q: -lejania(q)):   # primero las lejanas, las cercanas tapan
        p = board.piece_at(sq)
        if not p: continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        bx, by = (f + .5 + .3 * cx_) * S, (7 - r + .5 + .3 * cy_) * S
        alto = {1: .9, 2: 1.1, 3: 1.2, 4: 1.0, 5: 1.4, 6: 1.6}[p.piece_type] * S
        centro = (int(bx - cx_ * alto / 2), int(by - cy_ * alto / 2))
        c, borde = COLOR_BLANCAS if p.color else COLOR_NEGRAS
        cv2.ellipse(top, centro, (int(alto / 2), int(.3 * S)), ang, 0, 360, c, -1)
        cv2.ellipse(top, centro, (int(alto / 2), int(.3 * S)), ang, 0, 360, borde, 2)
    img = np.full((H, W, 3), 60, np.uint8)
    Hm = cv2.getPerspectiveTransform(np.float32([[0, L], [L, L], [L, 0], [0, 0]]), np.float32(CAM))
    img = cv2.warpPerspective(top, Hm, (W, H), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
    if mano is not None:
        cv2.ellipse(img, mano, (110, 160), 20, 0, 360, (120, 160, 210), -1)
    img = img.astype(np.float32) * rng.uniform(0.97, 1.03) + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

def jugar(san_list, fen=None, desordenar=False):
    seg = dj.Seguidor(calib, chess.Board(fen) if fen else None)
    real = chess.Board(fen) if fen else chess.Board()
    for _ in range(12): seg.procesar(render(real))
    ok = 0
    for san in san_list:
        m = real.parse_san(san); real.push(m)
        for t in range(10):                                  # la mano entra, mueve y sale
            seg.procesar(render(real, mano=(500 + 30 * t, 300 + 10 * t)))
        det = None
        for _ in range(20):
            _, d = seg.procesar(render(real))
            det = det or d
        ok += det == m
        if det != m: print("   FALLO en", san, "->", det, "|", seg.mensaje)
    return ok, len(san_list), seg

if __name__ == "__main__":                                   # importable: render() la usa chesscam/prueba_robustez.py
    partida = "e4 e5 Nf3 Nc6 Bc4 Nf6 O-O Nxe4 d4 exd4 Re1 d5 Bxd5 Qxd5 Nc3 Qa5 Nxe4 Be6 Neg5 O-O-O Nxe6 fxe6 Rxe6".split()
    ok, n, seg = jugar(partida)
    print(f"partida con enroques y capturas: {ok}/{n} jugadas bien detectadas")
    ok, n, _ = jugar("e4 a6 e5 d5 exd6".split())
    print(f"al paso: {ok}/{n}")
    # una mano que pasa sin mover nada no debe generar jugada
    b = chess.Board(); seg = dj.Seguidor(calib)
    for _ in range(12): seg.procesar(render(b))
    for t in range(8): seg.procesar(render(b, mano=(600, 350)))
    evs = [seg.procesar(render(b))[1] for _ in range(14)]
    print("mano sin jugada -> jugadas detectadas:", sum(e is not None for e in evs), "| referencia ok:", seg.board.fen() == b.fen())
    # jugada ilegal: no se acepta
    b = chess.Board(); seg = dj.Seguidor(calib)
    for _ in range(12): seg.procesar(render(b))
    mal = chess.Board(); mal.set_piece_at(chess.E5, mal.remove_piece_at(chess.E2))   # peon e2 a e5: ilegal
    for t in range(8): seg.procesar(render(mal, mano=(600, 350)))
    evs = [seg.procesar(render(mal))[1] for _ in range(14)]
    print("jugada ilegal -> aceptada:", any(e is not None for e in evs), "|", seg.mensaje)
