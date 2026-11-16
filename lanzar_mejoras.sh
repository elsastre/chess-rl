#!/bin/bash
# Corrida con las mejoras de la rama mejoras-meseta (ver el diagnostico de la meseta).
# Uso, desde la carpeta del repo:   bash lanzar_mejoras.sh [partidas]      (por defecto 30000; se puede cortar con Ctrl+C y reanudar)
# La salida queda tambien en runs/mejoras/salida.txt

cd "$(dirname "$0")" || exit 1
source .venv/bin/activate || { echo "ERROR: no pude activar .venv"; exit 1; }

SF=/opt/homebrew/bin/stockfish
OUT=runs/mejoras
PARTIDAS=${1:-30000}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/salida.txt") 2>&1

[ -x "$SF" ] || { echo "ERROR: falta Stockfish en $SF"; exit 1; }
[ -f runs/final/ck_150000.pt ] || { echo "ERROR: falta runs/final/ck_150000.pt"; exit 1; }
[ -f data/fens.txt ] || { echo "ERROR: falta data/fens.txt"; exit 1; }
[ -f "$OUT/ck_150000.pt" ] || cp runs/final/ck_150000.pt "$OUT/"     # rival viejo

echo "== Lanzando $PARTIDAS partidas en $OUT ($(date)) =="
#   --depth 2 --poda   el maestro ve una jugada mas: profundidad 2 con poda alfa-beta (~8 veces mas
#                      lento por jugada que profundidad 1, por eso son menos partidas)
#   --mates            la busqueda detecta mates en 1 en las hojas (a profundidad 2: ve su propio mate en 2)
#   --sf-azar          escalones faciles de Stockfish, para que el curriculo pueda subir
#   --temp-stockfish 0 contra Stockfish el agente juega sin exploracion (el curriculo mide su fuerza real)
#   --temp-jugadas 10  en el resto, explora solo en las primeras 10 jugadas de cada partida o tramo
#   --p-completas 0.2  1 de cada 5 partidas de autojuego se juega entera (resultado real, no material)
#   --buffer 2000000   memoria 5 veces mas larga (208 MB)
#   --lr-final 1e-4    la tasa de aprendizaje baja de 5e-4 a 1e-4
caffeinate -i python -u entrenar.py --out "$OUT" --init runs/final/ck_150000.pt --resume --algo mixto \
    --start lichess --fens data/fens.txt --stockfish "$SF" \
    --workers 6 --games "$PARTIDAS" --checkpoint-every 1000 \
    --depth 2 --poda --mates --sf-azar --temp-stockfish 0 --temp-jugadas 10 --p-completas 0.2 --buffer 2000000 --lr-final 1e-4
echo "== Entrenamiento terminado con codigo $? ($(date)) =="
