"""
vision.py - Rectificación del tablero y detección de cambios por casilla.

Idea: no se reconocen piezas. Se guarda una imagen de referencia del tablero
(rectificado, vista cenital) y, cuando la escena vuelve a estar quieta tras un
movimiento, se mide cuánto cambió cada una de las 64 casillas. Luego se elige,
entre las jugadas legales, la que mejor explica ese patrón de cambios.

Para el problema "pieza negra sobre casilla negra" se combinan varias señales:
  * luminancia con CLAHE (realza contraste en zonas oscuras),
  * crominancia (a, b de LAB),
  * textura / bordes (gradiente de Sobel): una pieza tiene contornos y brillos
    especulares aunque su color medio sea parecido al de la casilla.
Y aunque una casilla cambie poco, la jugada se deduce igual porque la otra
casilla implicada sí cambia y el conjunto de jugadas legales es pequeño.
"""
import numpy as np
import cv2

WARP = 480            # lado de la imagen rectificada (60 px por casilla)
SQ = WARP // 8


def square_name(idx):
    """idx 0..63 con a1=0, b1=1, ..., h8=63 (misma convención que python-chess)."""
    return "abcdefgh"[idx % 8] + str(idx // 8 + 1)


class BoardVision:
    def __init__(self, inner=0.70, motion_thresh=4.0, stable_frames=12):
        self.H = None                 # homografía cámara -> rectificada
        self.Hinv = None
        self.inner = inner            # fracción central de cada casilla que se usa
        self.motion_thresh = motion_thresh
        self.stable_frames = stable_frames
        self.reference = None         # features de referencia
        self.prev_small = None
        self.still_count = 0
        self.moved_since_ref = False
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.last_scores = np.zeros(64)
        self.last_feats = None
        self.empty_feats = None
        self.empty_valid = np.zeros(64, bool)

    # ---------- calibración ----------
    def set_corners(self, pts):
        """pts: 4 puntos (x, y) en la imagen de cámara, en orden a1, h1, h8, a8
        (esquinas EXTERIORES de esas casillas, sin el marco del tablero)."""
        src = np.float32(pts)
        dst = np.float32([[0, WARP], [WARP, WARP], [WARP, 0], [0, 0]])
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.Hinv = np.linalg.inv(self.H)
        self.reference = None
        self.prev_small = None
        self.empty_feats = None
        self.empty_valid = np.zeros(64, bool)

    def refine_corners(self, frame, corners, iterations=3):
        """Afina las 4 esquinas usando las intersecciones interiores visibles de la
        cuadrícula (las que no tapan piezas). Devuelve (esquinas, puntos_usados)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ideal_outer = np.float32([[0, WARP], [WARP, WARP], [WARP, 0], [0, 0]])
        corners = np.float32(corners)
        used = []
        for _ in range(iterations):
            H = cv2.getPerspectiveTransform(corners, ideal_outer)
            Hinv = np.linalg.inv(H)
            wgray = cv2.warpPerspective(gray, H, (WARP, WARP)).astype(np.float32)
            q = SQ // 3
            ideal, cam_guess = [], []
            for j in range(1, 8):
                for i in range(1, 8):
                    x, y = i * SQ, j * SQ
                    tl = wgray[y - q - 2:y - 2, x - q - 2:x - 2]
                    tr = wgray[y - q - 2:y - 2, x + 2:x + q + 2]
                    bl = wgray[y + 2:y + q + 2, x - q - 2:x - 2]
                    br = wgray[y + 2:y + q + 2, x + 2:x + q + 2]
                    m = [p.mean() for p in (tl, tr, bl, br)]
                    s = max(p.std() for p in (tl, tr, bl, br))
                    contrast = min(abs(m[0] - m[1]), abs(m[0] - m[2]),
                                   abs(m[3] - m[1]), abs(m[3] - m[2]))
                    # patrón de ajedrez: diagonales iguales, vecinos distintos, sin piezas
                    if (contrast > 30 and abs(m[0] - m[3]) < contrast * 0.5
                            and abs(m[1] - m[2]) < contrast * 0.5 and s < contrast * 0.6):
                        ideal.append((x, y))
                        cam_guess.append((x, y))
            if len(ideal) < 8:
                break
            cam_pts = cv2.perspectiveTransform(
                np.float32(cam_guess).reshape(-1, 1, 2), Hinv)
            # tamaño local de casilla en la cámara, para la ventana de búsqueda
            side = np.sqrt(abs(cv2.contourArea(corners))) / 8
            win = max(3, int(side * 0.2))
            before = cam_pts.copy()
            cv2.cornerSubPix(gray, cam_pts, (win, win), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
            shift = np.linalg.norm((cam_pts - before).reshape(-1, 2), axis=1)
            ok = shift < side * 0.3
            if ok.sum() < 8:
                break
            Hn, inl = cv2.findHomography(cam_pts[ok], np.float32(ideal)[ok].reshape(-1, 1, 2),
                                         cv2.RANSAC, 2.0)
            if Hn is None:
                break
            corners = cv2.perspectiveTransform(ideal_outer.reshape(-1, 1, 2),
                                               np.linalg.inv(Hn)).reshape(-1, 2)
            used = cam_pts[ok][inl.ravel() > 0].reshape(-1, 2)
        return corners.tolist(), np.asarray(used).tolist()

    @property
    def calibrated(self):
        return self.H is not None

    def warp(self, frame):
        return cv2.warpPerspective(frame, self.H, (WARP, WARP))

    def grid_lines_camera(self):
        """Segmentos de la cuadrícula proyectados en la imagen de cámara."""
        segs = []
        for i in range(9):
            t = i * SQ
            segs.append(((0, t), (WARP, t)))
            segs.append(((t, 0), (t, WARP)))
        pts = np.float32([p for s in segs for p in s]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, self.Hinv).reshape(-1, 2, 2)
        return proj

    # ---------- features ----------
    def features(self, warped_bgr):
        img = cv2.GaussianBlur(warped_bgr, (5, 5), 0)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Compensación de iluminación: se divide por la luz "de fondo" (muy suavizada).
        # Sombras suaves y cambios de exposición se cancelan; una pieza (más pequeña
        # que una casilla) apenas mueve ese fondo y se conserva.
        Lr = lab[:, :, 0]
        illum = cv2.GaussianBlur(Lr, (0, 0), SQ * 1.2)
        Ln = np.clip(Lr / (illum + 8.0) * 110.0, 0, 255).astype(np.uint8)
        L = self.clahe.apply(Ln)
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        a, b = lab[:, :, 1], lab[:, :, 2]
        return {
            "L": L.astype(np.float32),
            "a": a - np.median(a),          # quita derivas del balance de blancos
            "b": b - np.median(b),
            "g": grad,
        }

    def square_diffs(self, f_new, f_ref):
        """Vector de 64 valores: cuánto cambió cada casilla."""
        dL = np.abs(f_new["L"] - f_ref["L"])
        dC = np.abs(f_new["a"] - f_ref["a"]) + np.abs(f_new["b"] - f_ref["b"])
        dG = np.abs(f_new["g"] - f_ref["g"])
        combo = dL + 1.0 * dC + 0.25 * dG
        m = int(SQ * (1 - self.inner) / 2)
        out = np.zeros(64)
        for idx in range(64):
            f, r = idx % 8, idx // 8
            x0, y0 = f * SQ, (7 - r) * SQ
            patch = combo[y0 + m:y0 + SQ - m, x0 + m:x0 + SQ - m]
            # percentil alto: una pieza ocupa solo parte de la casilla
            out[idx] = 0.5 * patch.mean() + 0.5 * np.percentile(patch, 80)
        return out

    def set_reference(self, warped_bgr, empty_mask=None):
        """empty_mask: 64 booleanos, True = casilla vacía según la partida.
        Las casillas vacías se guardan como "aspecto de casilla vacía"."""
        self.reference = self.features(warped_bgr)
        self.moved_since_ref = False
        if empty_mask is not None:
            if self.empty_feats is None:
                self.empty_feats = {k: v.copy() for k, v in self.reference.items()}
            for idx in range(64):
                if empty_mask[idx]:
                    f, r = idx % 8, idx // 8
                    x0, y0 = f * SQ, (7 - r) * SQ
                    for k in self.empty_feats:
                        self.empty_feats[k][y0:y0 + SQ, x0:x0 + SQ] = \
                            self.reference[k][y0:y0 + SQ, x0:x0 + SQ]
                    self.empty_valid[idx] = True

    def empty_distance(self, feats):
        """Cuánto se diferencia cada casilla de su aspecto vacío (alto = parece ocupada)."""
        if self.empty_feats is None:
            return None
        return self.square_diffs(feats, self.empty_feats)

    # ---------- bucle por frame ----------
    def update(self, frame):
        """Procesa un frame. Devuelve (warped, evento):
           evento = None | 'moving' | 'settled'
           'settled' = la escena volvió a estar quieta después de haber habido movimiento."""
        warped = self.warp(frame)
        small = cv2.cvtColor(cv2.resize(warped, (96, 96)), cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)
        event = None
        if self.prev_small is not None:
            motion = np.abs(small - self.prev_small).mean()
            if motion > self.motion_thresh:
                self.still_count = 0
                self.moved_since_ref = True
                event = "moving"
            else:
                self.still_count += 1
                if self.moved_since_ref and self.still_count == self.stable_frames:
                    event = "settled"
        self.prev_small = small
        if self.reference is not None:
            self.last_feats = self.features(warped)
            self.last_scores = self.square_diffs(self.last_feats, self.reference)
        return warped, event


# ---------- inferencia de la jugada ----------
def normalize_scores(raw):
    """Resta el nivel de ruido (mediana) y escala por la dispersión robusta."""
    med = np.median(raw)
    mad = np.median(np.abs(raw - med)) + 1e-6
    z = (raw - med) / (1.4826 * mad + 2.0)
    return np.clip(z - 1.0, 0, None)      # por debajo de 1 "sigma" se considera ruido


def pick_move(raw_scores, candidates, empty_dist=None, empty_valid=None,
              empty_now=None, min_score=2.0, min_margin=1.5):
    """candidates: lista de (move, casillas_afectadas[, quedan_vacías, quedan_ocupadas]).

    Dos pistas:
      1) cambio: las casillas de la jugada deben haber cambiado respecto a la referencia;
      2) ocupación: las que quedan vacías deben parecerse a su aspecto vacío conocido y
         las que se ocupan, no. Esto descarta jugadas que solo explican una sombra.
    Devuelve (mejor_move | None, confianza, ranking)."""
    z = normalize_scores(raw_scores)
    occ = None
    if empty_dist is not None and empty_valid is not None and empty_now is not None:
        base = [s for s in range(64) if empty_valid[s] and empty_now[s]]
        if len(base) >= 6:
            b = empty_dist[base]
            med = np.median(b)
            mad = np.median(np.abs(b - med))
            occ = (empty_dist - med) / (1.4826 * mad + 2.0)
    ranked = []
    for cand in candidates:
        mv, squares = cand[0], list(cand[1])
        score = z[squares].sum() + 0.5 * z[squares].min() - 0.1 * len(squares)
        if occ is not None and len(cand) >= 4:
            for s in cand[2]:            # deberían verse vacías
                if empty_valid[s]:
                    score += np.clip(2.0 - occ[s], -4.0, 2.0)
            for s in cand[3]:            # deberían verse ocupadas
                if empty_valid[s]:
                    score += np.clip(occ[s] - 2.0, -4.0, 3.0)
        ranked.append((float(score), mv))
    if not ranked:
        return None, 0.0, []
    ranked.sort(key=lambda t: t[0], reverse=True)
    best = ranked[0][0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = best - second
    conf = margin / max(abs(best), 1.0)
    if best < min_score or margin < min_margin:
        return None, conf, ranked
    return ranked[0][1], conf, ranked
