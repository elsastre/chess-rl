"""
Interfaz grafica sencilla para jugar contra el bot (pygame).

Uso:
    pip install pygame
    python interfaz.py --ck runs\\final\\ck_150000.pt
    python interfaz.py --ck runs\\final\\ck_150000.pt --depth 1      (responde al instante, mas debil)

Clic en una pieza tuya y despues en la casilla destino (las jugadas legales se marcan).
La coronacion es siempre a dama. A la derecha: lo que siente y piensa el bot.
"""
import argparse
import os
import threading

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import chess
import pygame

from model import load_checkpoint, default_device
from search import search, make_evaluator, pick_move

SQ, BX, BY = 72, 24, 40
W, H = 940, 660
LIGHT, DARK = (238, 217, 182), (181, 136, 99)
SEL, LAST = (246, 211, 101), (205, 209, 111)
BG, INK, INK2, ACC = (246, 244, 239), (44, 44, 42), (95, 94, 90), (83, 74, 183)
GLYPH = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜", chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
SAN_ES = str.maketrans("NBRQK", "CATDR")    # notacion en espanol: Caballo, Alfil, Torre, Dama, Rey
NAMES = {chess.KNIGHT: "Caballo", chess.BISHOP: "Alfil", chess.ROOK: "Torre", chess.QUEEN: "Dama",
         chess.KING: "Rey", chess.PAWN: "Peón"}


def emotion(value, td):
    if abs(td) > 0.16:
        return "sorprendido"
    p = (value + 1) / 2
    return "ganando" if p >= .75 else "confiado" if p >= .5 else "preocupado" if p >= .3 else "perdiendo"


def speak(board, move):
    if board.is_castling(move):
        return "Enroque corto" if chess.square_file(move.to_square) == 6 else "Enroque largo"
    txt = NAMES[board.piece_type_at(move.from_square)] + (" come en " if board.is_capture(move) else " a ") \
        + chess.square_name(move.to_square)
    board.push(move)
    txt += ". ¡Jaque mate!" if board.is_checkmate() else ", jaque" if board.is_check() else ""
    board.pop()
    return txt


class Game:
    def __init__(self, evaluate, depth, q):
        self.evaluate, self.depth, self.q = evaluate, depth, q
        self.new(chess.WHITE)

    def new(self, human):
        self.board, self.human = chess.Board(), human
        self.selected, self.legal, self.last = None, [], None
        self.value, self.td, self.emo = 0.0, 0.0, "pensando"
        self.cands, self.said, self.thinking, self.result = [], "", False, None
        self.status = "Te toca." if human == chess.WHITE else "Pensando..."

    # ---------- turno del bot en un hilo aparte (la ventana no se congela)
    def bot_turn(self):
        if self.board.is_game_over(claim_draw=True) or self.board.turn == self.human or self.thinking:
            return
        self.thinking, self.emo, self.status = True, "pensando", "El bot está pensando..."
        board = self.board.copy()

        def work():
            res = search(board, self.evaluate, self.depth, self.q)
            self.result = res
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        if not self.thinking or self.result is None:
            return
        res, self.result, self.thinking = self.result, None, False
        move = pick_move(res.moves, 0.0)
        self.td = res.value - self.value
        self.value = res.value
        self.emo = emotion(res.value, self.td)
        self.cands = [(self.board.san(m).translate(SAN_ES), (s + 1) / 2) for m, s in sorted(res.moves, key=lambda t: -t[1])[:4]]
        self.said = speak(self.board, move)
        self.board.push(move)
        self.last = move
        self.check_end() or setattr(self, "status", "Te toca.")

    def check_end(self):
        if not self.board.is_game_over(claim_draw=True):
            return False
        out = self.board.outcome(claim_draw=True)
        if out.winner is None:
            self.status, self.emo = "Tablas.", "confiado"
        elif out.winner == self.human:
            self.status, self.emo = "¡Ganaste!", "perdiendo"
        else:
            self.status, self.emo = "Ganó el bot.", "ganando"
        return True

    def click_square(self, sq):
        if self.thinking or self.board.turn != self.human or self.board.is_game_over(claim_draw=True):
            return
        target = [m for m in self.legal if m.to_square == sq]
        if target:
            move = target[0]
            if len(target) > 1:                            # coronacion: siempre dama
                move = next(m for m in target if m.promotion == chess.QUEEN)
            self.board.push(move)
            self.last, self.selected, self.legal = move, None, []
            if not self.check_end():
                self.bot_turn()
            return
        p = self.board.piece_at(sq)
        if p and p.color == self.human:
            self.selected = sq
            self.legal = [m for m in self.board.legal_moves if m.from_square == sq]
        else:
            self.selected, self.legal = None, []

    def undo(self):
        if self.thinking:
            return
        n = 2 if self.board.turn == self.human else 1
        for _ in range(min(n, len(self.board.move_stack))):
            self.board.pop()
        self.selected, self.legal, self.last, self.said = None, [], None, ""
        self.status = "Te toca."
        if self.board.turn != self.human:
            self.bot_turn()


# =============================================================== dibujo
def sq_rect(sq, flip):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if flip:
        f, r = 7 - f, 7 - r
    return pygame.Rect(BX + f * SQ, BY + (7 - r) * SQ, SQ, SQ)


def pos_to_sq(x, y, flip):
    f, r = (x - BX) // SQ, 7 - (y - BY) // SQ
    if not (0 <= f < 8 and 0 <= r < 8):
        return None
    if flip:
        f, r = 7 - f, 7 - r
    return chess.square(f, r)


def draw_piece(screen, font, p, rect):
    g = GLYPH[p.piece_type]
    fg, edge = ((250, 250, 248), (30, 30, 30)) if p.color == chess.WHITE else ((25, 25, 25), (235, 235, 230))
    base = font.render(g, True, edge)
    c = rect.center
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        screen.blit(base, base.get_rect(center=(c[0] + dx, c[1] + dy + 2)))
    top = font.render(g, True, fg)
    screen.blit(top, top.get_rect(center=(c[0], c[1] + 2)))


def draw_face(screen, emo, center, r=46):
    cx, cy = center
    pygame.draw.circle(screen, (16, 19, 28), center, r)
    col = {"pensando": (159, 225, 203), "confiado": (159, 225, 203), "preocupado": (250, 199, 117),
           "sorprendido": (245, 196, 179), "ganando": (192, 221, 151), "perdiendo": (133, 183, 235)}[emo]
    ex = (cx - 16, cx + 16)
    if emo == "sorprendido":
        for x in ex:
            pygame.draw.circle(screen, col, (x, cy - 8), 9, 3)
        pygame.draw.circle(screen, col, (cx, cy + 20), 7, 3)
    elif emo in ("confiado", "ganando"):
        for x in ex:
            pygame.draw.arc(screen, col, (x - 9, cy - 14, 18, 16), 0.3, 2.8, 3)
        pygame.draw.arc(screen, col, (cx - 18, cy + 2, 36, 22 if emo == "ganando" else 16), 3.5, 5.9, 3)
    elif emo == "preocupado":
        for i, x in enumerate(ex):
            pygame.draw.ellipse(screen, col, (x - 5, cy - 14, 10, 13))
            s = -1 if i else 1
            pygame.draw.line(screen, col, (x - 9, cy - 22 + s * 4), (x + 9, cy - 22 - s * 4), 3)
        pygame.draw.arc(screen, col, (cx - 16, cy + 12, 32, 18), 0.4, 2.7, 3)
    elif emo == "perdiendo":
        for x in ex:
            pygame.draw.line(screen, col, (x - 8, cy - 6), (x + 8, cy - 6), 3)
        pygame.draw.arc(screen, col, (cx - 16, cy + 12, 32, 18), 0.4, 2.7, 3)
    else:   # pensando
        for x in ex:
            pygame.draw.ellipse(screen, col, (x - 5 + 4, cy - 18, 10, 15))
        pygame.draw.line(screen, col, (cx - 10, cy + 18), (cx + 10, cy + 18), 3)


class Button:
    def __init__(self, text, rect, action):
        self.text, self.rect, self.action = text, pygame.Rect(rect), action

    def draw(self, screen, font, hover):
        pygame.draw.rect(screen, (235, 233, 255) if hover else (255, 255, 255), self.rect, border_radius=8)
        pygame.draw.rect(screen, ACC, self.rect, 1, border_radius=8)
        t = font.render(self.text, True, INK)
        screen.blit(t, t.get_rect(center=self.rect.center))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--q", type=int, default=4)
    args = ap.parse_args()

    device = default_device()
    model, extra = load_checkpoint(args.ck, device)
    game = Game(make_evaluator(model, device), args.depth, args.q)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Robot de ajedrez RL")
    piece_font = pygame.font.SysFont(["segoeuisymbol", "dejavusans", "arialunicodems", "freeserif"], 54)
    f_big = pygame.font.SysFont(["segoeui", "dejavusans", "arial"], 22, bold=True)
    f = pygame.font.SysFont(["segoeui", "dejavusans", "arial"], 17)
    f_small = pygame.font.SysFont(["segoeui", "dejavusans", "arial"], 14)
    px = BX + 8 * SQ + 32
    buttons = [Button("Nueva (blancas)", (px, H - 110, 135, 36), lambda: game.new(chess.WHITE)),
               Button("Nueva (negras)", (px + 145, H - 110, 135, 36), lambda: (game.new(chess.BLACK), game.bot_turn())),
               Button("Deshacer", (px, H - 64, 280, 36), game.undo)]
    clock = pygame.time.Clock()

    running = True
    while running:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for b in buttons:
                    if b.rect.collidepoint(ev.pos):
                        b.action()
                sq = pos_to_sq(*ev.pos, game.human == chess.BLACK)
                if sq is not None:
                    game.click_square(sq)
        game.poll()

        flip = game.human == chess.BLACK
        screen.fill(BG)
        for sq in chess.SQUARES:
            r = sq_rect(sq, flip)
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            pygame.draw.rect(screen, LIGHT if light else DARK, r)
            if game.last and sq in (game.last.from_square, game.last.to_square):
                s = pygame.Surface((SQ, SQ), pygame.SRCALPHA); s.fill((*LAST, 140)); screen.blit(s, r)
            if sq == game.selected:
                s = pygame.Surface((SQ, SQ), pygame.SRCALPHA); s.fill((*SEL, 170)); screen.blit(s, r)
        if game.board.is_check():
            k = game.board.king(game.board.turn)
            pygame.draw.rect(screen, (226, 75, 74), sq_rect(k, flip), 4)
        for sq in chess.SQUARES:
            p = game.board.piece_at(sq)
            if p:
                draw_piece(screen, piece_font, p, sq_rect(sq, flip))
        for m in game.legal:
            r = sq_rect(m.to_square, flip)
            if game.board.piece_at(m.to_square):
                pygame.draw.circle(screen, (83, 74, 183), r.center, SQ // 2 - 3, 4)
            else:
                pygame.draw.circle(screen, (83, 74, 183), r.center, 9)
        for i in range(8):
            fl = "abcdefgh"[7 - i if flip else i]
            rk = str(i + 1 if flip else 8 - i)
            screen.blit(f_small.render(fl, True, INK2), (BX + i * SQ + SQ // 2 - 4, BY + 8 * SQ + 4))
            screen.blit(f_small.render(rk, True, INK2), (BX - 16, BY + i * SQ + SQ // 2 - 8))

        # panel
        y = BY
        screen.blit(f_big.render("Robot de ajedrez RL", True, INK), (px, y)); y += 32
        screen.blit(f_small.render(f"Entrenado con {extra.get('games', '?')} partidas", True, INK2), (px, y)); y += 30
        screen.blit(f.render(game.status, True, ACC if game.thinking else INK), (px, y)); y += 34
        draw_face(screen, game.emo, (px + 52, y + 50))
        screen.blit(f.render(game.emo, True, INK), (px + 116, y + 26))
        screen.blit(f_small.render(f"sorpresa (error TD): {game.td:+.2f}", True, INK2), (px + 116, y + 52)); y += 112
        p = (game.value + 1) / 2
        screen.blit(f.render(f"Chance de ganar del bot: {round(p * 100)} %", True, INK), (px, y)); y += 26
        pygame.draw.rect(screen, (217, 214, 204), (px, y, 280, 10), border_radius=5)
        pygame.draw.rect(screen, ACC, (px, y, int(280 * p), 10), border_radius=5); y += 26
        if game.said:
            screen.blit(f.render("Bot: " + game.said, True, INK), (px, y)); y += 30
        if game.cands:
            screen.blit(f_small.render("Lo que consideró:", True, INK2), (px, y)); y += 22
            for san, pr in game.cands:
                screen.blit(f.render(san, True, INK), (px, y))
                pygame.draw.rect(screen, (217, 214, 204), (px + 80, y + 7, 150, 8), border_radius=4)
                pygame.draw.rect(screen, ACC, (px + 80, y + 7, int(150 * pr), 8), border_radius=4)
                screen.blit(f_small.render(f"{round(pr * 100)} %", True, INK2), (px + 238, y + 1))
                y += 24
        for b in buttons:
            b.draw(screen, f, b.rect.collidepoint(mouse))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
