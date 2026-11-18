#!/bin/bash
# Mediciones para atribuir la mejora al RL (ver CLAUDE.md, seccion 4.4). Corre todo en orden; se puede dejar de noche.
#   1. Prediccion del resultado en partidas de Lichess de febrero de 2013 (no usadas para entrenar).
#   2. Enfrentamientos directos con aperturas pareadas.
# Uso, desde la carpeta del repo:   bash lanzar_mediciones.sh
# Si se corta, se puede relanzar: lo ya hecho se saltea. Todo queda en runs/mediciones/ (salida.txt = esta pantalla).
#
# Checkpoints que usa si existen (los que no estan, se saltean):
#   runs/final/ck_000000.pt   red ANTES del RL (pedirsela a quien tiene la laptop del entrenamiento)
#   runs/final/ck_150000.pt   despues del RL (bot final)
#   runs/final_sf/ck_*.pt     continuacion con Stockfish
#   runs/mejoras/ck_*.pt      corrida con las mejoras (profundidad 2)
# Si falta ck_000000.pt, se regenera una red de material con el mismo procedimiento (checkpoints/material_regen.pt).

cd "$(dirname "$0")" || exit 1
source .venv/bin/activate || { echo "ERROR: no pude activar .venv"; exit 1; }
OUT=runs/mediciones
mkdir -p "$OUT" data checkpoints
exec > >(tee -a "$OUT/salida.txt") 2>&1
WORKERS=${WORKERS:-6}
PARES=${PARES:-500}
echo "== Mediciones ($(date)) | $WORKERS procesos, $PARES aperturas por enfrentamiento =="

# ---------------------------------------------------------------- linea de base (sin RL)
if [ -f runs/final/ck_000000.pt ]; then
    BASE=runs/final/ck_000000.pt; NBASE=sin_rl_ck0
else
    BASE=checkpoints/material_regen.pt; NBASE=sin_rl_regen
    if [ ! -f "$BASE" ]; then
        echo; echo "== Falta runs/final/ck_000000.pt: regenero la red de material (pocos minutos) =="
        python preentrenar_material.py --out "$BASE" --fens data/fens.txt || { echo "ERROR al regenerar"; exit 1; }
    fi
fi
echo "Linea de base sin RL: $BASE"

# ---------------------------------------------------------------- checkpoints a medir
CKS=("$NBASE=$BASE")
[ -f runs/final/ck_000000.pt ] && [ -f checkpoints/material_regen.pt ] && CKS+=("sin_rl_regen=checkpoints/material_regen.pt")
[ -f runs/final/ck_150000.pt ] && CKS+=("rl_150k=runs/final/ck_150000.pt")
for k in 200000 250000 300000; do
    [ -f runs/final_sf/ck_$k.pt ] && CKS+=("sf_$((k / 1000))k=runs/final_sf/ck_$k.pt")
done
ULT_MEJ=$(ls runs/mejoras/ck_*.pt 2>/dev/null | sort | tail -1)
for f in $(ls runs/mejoras/ck_*.pt 2>/dev/null | sort | awk 'NR % 10 == 0'); do
    [ "$f" != "$ULT_MEJ" ] && CKS+=("mej_$(basename "$f" .pt | sed 's/ck_0*//; s/000$/k/')=$f")
done
[ -n "$ULT_MEJ" ] && CKS+=("mej_$(basename "$ULT_MEJ" .pt | sed 's/ck_0*//; s/000$/k/')=$ULT_MEJ")
echo "Checkpoints: ${CKS[*]}"

# ---------------------------------------------------------------- 1. prediccion del resultado
PGN=data/lichess_db_standard_rated_2013-02.pgn.zst
if [ ! -f "$PGN" ]; then
    echo; echo "== Bajando partidas de Lichess de febrero de 2013 =="
    curl -L --fail -o "$PGN.tmp" https://database.lichess.org/standard/lichess_db_standard_rated_2013-02.pgn.zst \
        && mv "$PGN.tmp" "$PGN" || echo "AVISO: no pude bajar el PGN; salteo la medicion 1"
fi
if [ -f "$PGN" ]; then
    echo; echo "== 1. Prediccion del resultado ($(date)) =="
    caffeinate -i python -u medir_valor.py --pgn "$PGN" --ck "${CKS[@]}" --out "$OUT" --partidas 6000
fi

# ---------------------------------------------------------------- 2. enfrentamientos directos
duelo() {   # duelo A nombreA B nombreB
    [ -f "$1" ] && [ -f "$3" ] || { echo "(salteo $2 contra $4: falta un checkpoint)"; return; }
    echo; echo "== 2. Enfrentamiento: $2 contra $4 ($(date)) =="
    caffeinate -i python -u duelo.py --a "$1" --nombre-a "$2" --b "$3" --nombre-b "$4" --pares "$PARES" \
        --poda --workers "$WORKERS" --out "$OUT/duelo_${2}_vs_${4}.csv"
}
duelo runs/final/ck_150000.pt rl_150k "$BASE" "$NBASE"                 # lo que aporto el RL
[ -n "$ULT_MEJ" ] && duelo "$ULT_MEJ" "mej_$(basename "$ULT_MEJ" .pt | sed 's/ck_0*//; s/000$/k/')" runs/final/ck_150000.pt rl_150k
duelo runs/final_sf/ck_300000.pt sf_300k runs/final/ck_150000.pt rl_150k   # la meseta

python graficar_mediciones.py --dir "$OUT"

echo; echo "== RESUMEN ($(date)) =="
[ -f "$OUT/valor_resumen.txt" ] && cat "$OUT/valor_resumen.txt"
for f in "$OUT"/duelo_*_resumen.txt; do [ -f "$f" ] && { echo; cat "$f"; }; done
echo; echo "== Terminado ($(date)) =="
