"""
Explicabilidad (XAI): que aprendio la red con el RL, por OCLUSION de piezas.

Idea: la entrada de la red es "que pieza hay en cada casilla" (12 planos de 8x8). Para saber cuanto le importa
a la red una pieza en una posicion, se la saca del tablero y se mira cuanto cambia el valor:

    importancia(pieza) = valor para su dueno CON la pieza - valor para su dueno SIN la pieza

Se mide con la salida real de la red (no con una aproximacion como los gradientes), y como la entrada es binaria
(hay o no hay pieza) sacar la pieza es la perturbacion natural: "media pieza" no existe.

Unidades: el valor v en [-1, 1] se pasa a peones con la misma escala con la que se preentreno la red de material
(v = tanh(cp * ln 10 / 800), ver medir_valor.py): cp = 800 / ln 10 * atanh(v). Asi la red de material deberia dar
algo cerca de los valores clasicos 1 / 3 / 3 / 5 / 9, y lo que se aparte el RL es lo que aprendio.

Dos analisis:
  1. Global: importancia promedio de cada tipo de pieza (y del peon segun la fila a la que llego) sobre miles de
     posiciones tranquilas de partidas de Lichess que ninguna red vio. Compara checkpoints (ck_0, 150k, 210k).
  2. Local: mapas sobre el tablero para algunas posiciones: cuanto vale cada pieza para cada red, y la diferencia.

Limitaciones: los reyes no se sacan (la posicion dejaria de ser ajedrez); al sacar una pieza la posicion puede
quedar rara (piezas colgadas), asi que se usan posiciones tranquilas (sin jaque y sin capturas que la quiescencia
quiera hacer) y se descartan las que quedan ilegales. La oclusion mide el efecto de cada pieza por separado, no
sus interacciones.

Uso:
    python explicar.py --pgn data/lichess_db_standard_rated_2013-02.pgn.zst \\
        --ck sin_rl=runs/final/ck_000000.pt rl_150k=runs/final/ck_150000.pt mej_210k=runs/mejoras/ck_210000.pt \\
        --out runs/explicar
"""
import argparse
import csv
import math
import os
import random

import chess
import chess.pgn
import numpy as np
import torch

from dataset_lichess import open_pgn, RESULT
from encoding import encode
from model import load_checkpoint
from search import qsearch, INF

TIPOS = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
NOMBRE = {chess.PAWN: "peon", chess.KNIGHT: "caballo", chess.BISHOP: "alfil", chess.ROOK: "torre",
          chess.QUEEN: "dama"}
CLASICO = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
ESCALA = 800 / math.log(10)            # cp = ESCALA * atanh(v)

# mismos colores que graficar_mediciones.py: gris sin RL, azul RL, naranja continuacion, verde mejoras
COLORES = {"sin_rl": "#8a8a85", "rl": "#2a78d6", "sf": "#eb6834", "mej": "#1baa7d"}
INK, INK2 = "#1f1f1f", "#5f5f5f"
GLIFO = {(chess.KING, True): "♔", (chess.QUEEN, True): "♕", (chess.ROOK, True): "♖", (chess.BISHOP, True): "♗",
         (chess.KNIGHT, True): "♘", (chess.PAWN, True): "♙", (chess.KING, False): "♚", (chess.QUEEN, False): "♛",
         (chess.ROOK, False): "♜", (chess.BISHOP, False): "♝", (chess.KNIGHT, False): "♞", (chess.PAWN, False): "♟"}


def color_de(nombre):
    for k, c in COLORES.items():
        if nombre.startswith(k):
            return c
    return "#eb6834"


def a_peones(v):
    return ESCALA * np.arctanh(np.clip(v, -0.999, 0.999)) / 100


# ---------------------------------------------------------------- datos
def posiciones_tranquilas(pgn, n, por_partida, saltear, min_elo, semilla):
    """FENs de posiciones tranquilas: sin jaque, la quiescencia no quiere capturar nada, y al menos 6 piezas
    que no sean reyes (para que haya algo que sacar)."""
    rng = random.Random(semilla)
    fens = []
    with open_pgn(pgn) as f:
        while len(fens) < n:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            h = game.headers
            if h.get("Result") not in RESULT or h.get("Termination", "Normal") != "Normal":
                continue
            try:
                if min(int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))) < min_elo:
                    continue
            except ValueError:
                continue
            moves = list(game.mainline_moves())
            if len(moves) <= saltear + 2:
                continue
            elegidas = set(rng.sample(range(saltear, len(moves)), min(por_partida, len(moves) - saltear)))
            b = game.board()
            for ply, mv in enumerate(moves):
                if ply in elegidas and b.outcome() is None and not b.is_check():
                    _, pv = qsearch(b, -INF, INF, 4)
                    if not pv and len(b.piece_map()) - 2 >= 6:
                        fens.append(b.fen())
                b.push(mv)
    return fens[:n]


def oclusiones(fen):
    """La posicion y sus variantes con una pieza (no rey) menos. Devuelve (tableros, info) con
    info[i] = (casilla, tipo, color de la pieza) para cada variante legal."""
    b = chess.Board(fen)
    tableros, info = [b], []
    for sq, p in b.piece_map().items():
        if p.piece_type == chess.KING:
            continue
        c = b.copy(stack=False)
        c.remove_piece_at(sq)
        if not c.is_valid():            # p. ej. deja en jaque al rey del que no mueve
            continue
        tableros.append(c)
        info.append((sq, p.piece_type, p.color))
    return tableros, info


@torch.no_grad()
def evaluar(model, tableros, device):
    X = np.stack([encode(t) for t in tableros])
    out = []
    for i in range(0, len(X), 4096):
        out.append(model(torch.from_numpy(X[i:i + 4096]).float().to(device)).float().cpu().numpy())
    return np.concatenate(out)


def importancias(model, fen, device):
    """Lista de (casilla, tipo, color, importancia en peones para el dueno de la pieza)."""
    tableros, info = oclusiones(fen)
    v = evaluar(model, tableros, device)
    turno = tableros[0].turn
    base = a_peones(v[0])                       # para el que mueve
    res = []
    for (sq, pt, col), vi in zip(info, v[1:]):
        delta = base - a_peones(vi)              # cuanto pierde el que mueve al sacarla
        res.append((sq, pt, col, delta if col == turno else -delta))
    return res


def fila_relativa(sq, color):
    """Fila del peon vista desde su dueno: 2 = sin mover, 7 = a punto de coronar."""
    r = chess.square_rank(sq) + 1
    return r if color == chess.WHITE else 9 - r


def es_pasado(b, sq, color):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for ff in (f - 1, f, f + 1):
        if not 0 <= ff <= 7:
            continue
        for rr in (range(r + 1, 8) if color == chess.WHITE else range(0, r)):
            q = b.piece_at(chess.square(ff, rr))
            if q and q.piece_type == chess.PAWN and q.color != color:
                return False
    return True


# ---------------------------------------------------------------- analisis global
def global_(modelos, fens, device, semilla, n_boot):
    """Por red: importancia media de cada tipo de pieza, del peon por fila y del peon pasado / no pasado,
    con IC 95 % por bootstrap de POSICIONES (las piezas de una misma posicion no son independientes)."""
    filas = []
    for nombre, model in modelos.items():
        # por posicion: suma y cantidad por grupo (para el bootstrap por posicion)
        grupos = {}
        for i, fen in enumerate(fens):
            b = chess.Board(fen)
            for sq, pt, col, imp in importancias(model, fen, device):
                claves = [NOMBRE[pt]]
                if pt == chess.PAWN:
                    claves.append(f"peon fila {fila_relativa(sq, col)}")
                    claves.append("peon pasado" if es_pasado(b, sq, col) else "peon no pasado")
                for k in claves:
                    g = grupos.setdefault(k, {})
                    s, c = g.get(i, (0.0, 0))
                    g[i] = (s + imp, c + 1)
        rng = np.random.default_rng(semilla)
        for k, g in grupos.items():
            idx = np.array(list(g))
            s = np.array([g[j][0] for j in idx])
            c = np.array([g[j][1] for j in idx])
            media = s.sum() / c.sum()
            boots = []
            for _ in range(n_boot):
                r = rng.integers(0, len(idx), len(idx))
                boots.append(s[r].sum() / c[r].sum())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            filas.append({"modelo": nombre, "grupo": k, "media": media, "lo": lo, "hi": hi,
                          "piezas": int(c.sum()), "posiciones": len(idx)})
        print(f"  {nombre}: listo", flush=True)
    return filas


def graficar_global(filas, nombres, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False})
    d = {(f["modelo"], f["grupo"]): f for f in filas}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})

    # izquierda: valor de cada pieza
    grupos = [NOMBRE[t] for t in TIPOS]
    n = len(nombres)
    ancho = 0.8 / (n + 1)
    for j, nom in enumerate(nombres):
        xs, ys, e = [], [], [[], []]
        for i, g in enumerate(grupos):
            f = d.get((nom, g))
            if f is None:
                continue
            xs.append(i + (j - n / 2) * ancho)
            ys.append(f["media"])
            e[0].append(f["media"] - f["lo"])
            e[1].append(f["hi"] - f["media"])
        a1.bar(xs, ys, ancho * 0.92, color=color_de(nom), label=nom, yerr=e, capsize=2,
               error_kw={"lw": 1, "ecolor": INK})
    xs = [i + (n / 2) * ancho for i in range(len(grupos))]
    a1.bar(xs, [CLASICO[t] for t in TIPOS], ancho * 0.92, color="white", edgecolor=INK2, hatch="//",
           label="clásico (1/3/3/5/9)")
    a1.set_xticks(range(len(grupos)))
    a1.set_xticklabels(grupos)
    a1.set_ylabel("cuánto pierde su dueño si se la saca (peones)")
    a1.set_title("Valor de cada pieza para la red", color=INK, fontweight="bold")
    a1.legend(frameon=False, fontsize=9)
    a1.grid(axis="y", color="#e5e5e5")
    a1.set_axisbelow(True)

    # derecha: el peon segun la fila a la que llego
    for nom in nombres:
        xs, ys, lo, hi = [], [], [], []
        for r in range(2, 8):
            f = d.get((nom, f"peon fila {r}"))
            if f is None or f["piezas"] < 30:
                continue
            xs.append(r)
            ys.append(f["media"])
            lo.append(f["lo"])
            hi.append(f["hi"])
        a2.fill_between(xs, lo, hi, color=color_de(nom), alpha=.15, lw=0)
        a2.plot(xs, ys, "-o", color=color_de(nom), lw=2, ms=7, label=nom)
    a2.axhline(1, color=INK2, lw=1, ls="--")
    a2.text(2.05, 1.02, "clásico: 1 peón", color=INK2, fontsize=8.5, va="bottom")
    a2.set_xticks(range(2, 8))
    a2.set_xlabel("fila a la que llegó el peón (desde su dueño; 7 = a un paso de coronar)")
    a2.set_ylabel("valor del peón (peones)")
    a2.set_title("¿Aprendió que un peón avanzado vale más?", color=INK, fontweight="bold")
    a2.legend(frameon=False, fontsize=9)
    a2.grid(axis="y", color="#e5e5e5")
    a2.set_axisbelow(True)
    npos = max(f["posiciones"] for f in filas)
    fig.text(.5, .005, f"Oclusión: se saca cada pieza (no los reyes) y se mide cuánto cambia el valor de la red. "
                       f"{npos} posiciones tranquilas de Lichess (feb. 2013), IC 95 % por bootstrap de posiciones.",
             ha="center", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- analisis local (mapas)
def dibujar_tablero(ax, fen, imps, lim, titulo, diferencia=False):
    """Tablero desde las blancas; cada pieza (salvo reyes) con un recuadro coloreado por su importancia."""
    import matplotlib.colors as mcolors
    from matplotlib import cm
    b = chess.Board(fen)
    import matplotlib
    cmap = matplotlib.colormaps["PuOr" if diferencia else "RdBu"]
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for f in range(8):
        for r in range(8):
            ax.add_patch(matplotlib_rect(f, r, "#f0d9b5" if (f + r) % 2 else "#b58863"))
    valores = {sq: imp for sq, _, _, imp in imps}
    for sq, p in b.piece_map().items():
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if sq in valores:
            ax.add_patch(matplotlib_rect(f + .06, r + .06, cmap(norm(valores[sq])), w=.88, alpha=.85))
            ax.text(f + .5, r + .1, f"{valores[sq]:+.1f}", ha="center", va="bottom", fontsize=7.5,
                    color=INK, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": .8})
        ax.text(f + .5, r + .64, GLIFO[(p.piece_type, p.color)], ha="center", va="center", fontsize=18,
                color="black", fontfamily="DejaVu Sans")
    for i in range(8):
        ax.text(i + .5, -.25, "abcdefgh"[i], ha="center", va="center", fontsize=8, color=INK2)
        ax.text(-.25, i + .5, str(i + 1), ha="center", va="center", fontsize=8, color=INK2)
    ax.set_xlim(-.45, 8)
    ax.set_ylim(-.45, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo, fontsize=10.5, color=INK, fontweight="bold")
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def matplotlib_rect(x, y, color, w=1.0, alpha=1.0):
    import matplotlib.patches as mp
    return mp.Rectangle((x, y), w, w, facecolor=color, edgecolor="none", alpha=alpha)


def graficar_mapa(fen, modelos_par, device, out, nota):
    """Tres tableros: importancia para la red sin RL, para la red con RL, y la diferencia."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (n0, m0), (n1, m1) = modelos_par
    i0 = {sq: (sq, pt, c, v) for sq, pt, c, v in importancias(m0, fen, device)}
    i1 = {sq: (sq, pt, c, v) for sq, pt, c, v in importancias(m1, fen, device)}
    dif = [(sq, i1[sq][1], i1[sq][2], i1[sq][3] - i0[sq][3]) for sq in i1 if sq in i0]
    lim = max(1.0, max(abs(v) for *_, v in list(i0.values()) + list(i1.values())))
    limd = max(0.5, max(abs(v) for *_, v in dif))
    b = chess.Board(fen)
    v0 = a_peones(evaluar(m0, [b], device)[0])
    v1 = a_peones(evaluar(m1, [b], device)[0])
    mueve = "blancas" if b.turn == chess.WHITE else "negras"
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"wspace": .32})
    sm = dibujar_tablero(axs[0], fen, list(i0.values()), lim, f"{n0}: {v0:+.1f} para las {mueve}")
    dibujar_tablero(axs[1], fen, list(i1.values()), lim, f"{n1}: {v1:+.1f} para las {mueve}")
    smd = dibujar_tablero(axs[2], fen, dif, limd, f"diferencia ({n1} − {n0})", diferencia=True)
    fig.colorbar(sm, ax=axs[0], fraction=.046, pad=.03).ax.set_visible(False)    # mismo tamaño que los otros
    fig.colorbar(sm, ax=axs[1], fraction=.046, pad=.03, label="valor de la pieza para su dueño (peones)")
    fig.colorbar(smd, ax=axs[2], fraction=.046, pad=.02, label="cuánto más la valora tras el RL")
    fig.text(.5, .01, nota, ha="center", va="bottom", fontsize=8.5, color=INK2, wrap=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


POSICIONES_FIJAS = [
    # (nombre, fen, por que)
    ("peon_pasado", "8/5k2/8/1P6/8/8/5K2/8 w - - 0 1",
     "Final de rey y peón: el peón pasado de b5 está a tres pasos de coronar y el rey negro no lo alcanza."),
    ("medio_juego", "r1bq1rk1/ppp2ppp/2n5/3np3/2B5/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 1",
     "Medio juego abierto, material igual. Un número negativo = la red cree que su dueño estaría mejor sin esa "
     "pieza (p. ej. un peón que tapa la columna de su dama)."),
]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Explicabilidad por oclusion de piezas")
    ap.add_argument("--pgn", default="data/lichess_db_standard_rated_2013-02.pgn.zst")
    ap.add_argument("--ck", nargs="+", required=True, help="nombre=ruta; el primero es la referencia sin RL")
    ap.add_argument("--out", default="runs/explicar")
    ap.add_argument("--posiciones", type=int, default=3000)
    ap.add_argument("--por-partida", type=int, default=3)
    ap.add_argument("--saltear", type=int, default=10, help="primeras jugadas de cada partida que no se usan")
    ap.add_argument("--min-elo", type=int, default=1500)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--mapas", type=int, default=3, help="posiciones de Lichess con mapa (las de mas cambio)")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cpu"                      # la red es chica: en CPU alcanza y da lo mismo en cualquier maquina
    torch.set_grad_enabled(False)

    modelos = {}
    for par in args.ck:
        nombre, ruta = par.split("=", 1)
        modelos[nombre] = load_checkpoint(ruta, device)[0]
    nombres = list(modelos)

    print(f"Leyendo {args.posiciones} posiciones tranquilas de {args.pgn} ...", flush=True)
    fens = posiciones_tranquilas(args.pgn, args.posiciones, args.por_partida, args.saltear, args.min_elo,
                                 args.semilla)
    print(f"  {len(fens)} posiciones", flush=True)

    print("1. Valor de cada pieza (oclusion) ...", flush=True)
    filas = global_(modelos, fens, device, args.semilla, args.boot)
    with open(os.path.join(args.out, "valores_piezas.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    graficar_global(filas, nombres, os.path.join(args.out, "valores_piezas.png"))

    d = {(f["modelo"], f["grupo"]): f for f in filas}
    orden = [NOMBRE[t] for t in TIPOS] + ["peon pasado", "peon no pasado"] + [f"peon fila {r}" for r in range(2, 8)]
    lineas = ["Valor de cada pieza para la red, por oclusion (peones; IC 95 % por bootstrap de posiciones)",
              f"{len(fens)} posiciones tranquilas de {args.pgn}", ""]
    lineas.append(f"{'grupo':<16}" + "".join(f"{n:>24}" for n in nombres) + f"{'clasico':>9}")
    for g in orden:
        celdas = []
        for n in nombres:
            f = d.get((n, g))
            celdas.append(f"{f['media']:+.2f} [{f['lo']:+.2f},{f['hi']:+.2f}]" if f else "-")
        clas = next((str(CLASICO[t]) for t in TIPOS if NOMBRE[t] == g), "")
        lineas.append(f"{g:<16}" + "".join(f"{c:>24}" for c in celdas) + f"{clas:>9}")
    # la misma tabla relativa al peon (no pasado) de cada red: separa "cambio el orden de valores" de
    # "la red se volvio mas segura de todo" (si solo se reescalara, los cocientes no cambiarian)
    lineas += ["", "Relativo al peon no pasado de cada red (x veces ese peon):",
               f"{'grupo':<16}" + "".join(f"{n:>24}" for n in nombres) + f"{'clasico':>9}"]
    for g in [NOMBRE[t] for t in TIPOS] + ["peon pasado", "peon fila 6", "peon fila 7"]:
        celdas = []
        for n in nombres:
            f, p = d.get((n, g)), d.get((n, "peon no pasado"))
            celdas.append(f"{f['media'] / p['media']:.1f}" if f and p else "-")
        clas = next((str(CLASICO[t]) for t in TIPOS if NOMBRE[t] == g), "")
        lineas.append(f"{g:<16}" + "".join(f"{c:>24}" for c in celdas) + f"{clas:>9}")
    txt = "\n".join(lineas)
    print(txt)
    open(os.path.join(args.out, "valores_piezas.txt"), "w").write(txt + "\n")

    print("2. Mapas sobre el tablero ...", flush=True)
    ref, ult = nombres[0], nombres[-1]
    par = [(ref, modelos[ref]), (ult, modelos[ult])]
    for nombre, fen, porque in POSICIONES_FIJAS:
        graficar_mapa(fen, par, device, os.path.join(args.out, f"mapa_{nombre}.png"),
                      f"{porque} Posición armada a mano. Número = cuánto pierde su dueño si se saca esa pieza.")
    # de Lichess: las posiciones donde mas cambio la importancia de alguna pieza entre las dos redes
    cambio = []
    for fen in fens[:1000]:
        a = {sq: v for sq, *_, v in importancias(modelos[ref], fen, device)}
        b = {sq: v for sq, *_, v in importancias(modelos[ult], fen, device)}
        cambio.append((max(abs(b[s] - a[s]) for s in b if s in a), fen))
    cambio.sort(reverse=True)
    for k, (c, fen) in enumerate(cambio[:args.mapas], 1):
        graficar_mapa(fen, par, device, os.path.join(args.out, f"mapa_lichess_{k}.png"),
                      f"Posición de Lichess elegida entre 1000 por ser de las que más cambió la importancia de "
                      f"una pieza ({c:.1f} peones): muestra el cambio más grande, no el típico.")
    print(f"Listo: {args.out}")


if __name__ == "__main__":
    main()
