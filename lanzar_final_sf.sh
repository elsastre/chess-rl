#!/bin/bash
# Verifica el entorno y lanza la continuacion del bot final con Stockfish como rival (runs/final_sf).
# Uso, desde la carpeta del repo:   bash lanzar_final_sf.sh
# Toda la salida queda tambien en runs/final_sf/salida.txt

cd "$(dirname "$0")" || exit 1
source .venv/bin/activate || { echo "ERROR: no pude activar .venv"; exit 1; }

SF=/opt/homebrew/bin/stockfish
OUT=runs/final_sf
SALIDA=$OUT/salida.txt
mkdir -p "$OUT"
exec > >(tee -a "$SALIDA") 2>&1

fallo() { echo; echo "ERROR: $1"; echo "== VERIFICACION FALLIDA =="; exit 1; }

echo "== 1. Verificando entorno ($(date)) =="
python -c "import torch; print('torch', torch.__version__, '| MPS disponible:', torch.backends.mps.is_available())" \
    || fallo "torch no se pudo importar"
DEV=$(python -c "from model import default_device; print(default_device())")
echo "default_device(): $DEV"
[ -f runs/final/ck_150000.pt ] || fallo "falta runs/final/ck_150000.pt"
[ -f data/fens.txt ] || fallo "falta data/fens.txt"
echo "data/fens.txt: $(wc -l < data/fens.txt) posiciones"
[ -x "$SF" ] || fallo "no encuentro Stockfish en $SF. Instalalo con:  brew install stockfish"
echo "Stockfish: $SF"

# Prueba real de MPS: cargar el bot y hacer un paso de entrenamiento en la GPU de la Mac
if [ "$DEV" = "mps" ]; then
    if python - <<'EOF'
import torch
from model import load_checkpoint
model, extra = load_checkpoint("runs/final/ck_150000.pt", "mps")
model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-9)
x = torch.rand(64, 12, 8, 8, device="mps")
loss = torch.nn.functional.smooth_l1_loss(model(x), torch.zeros(64, device="mps"), beta=0.1)
opt.zero_grad(); loss.backward(); opt.step()
print("prueba MPS OK | partidas del checkpoint:", extra.get("games"))
EOF
    then :
    else
        echo "AVISO: la prueba en MPS fallo, se entrena en CPU (CHESS_DEVICE=cpu)"
        export CHESS_DEVICE=cpu
    fi
fi

echo; echo "== Prueba de Stockfish: arena.py con 4 partidas =="
python arena.py --games 4 --stockfish "$SF" || fallo "arena.py fallo (revisar Stockfish)"

echo; echo "== 2. Rival viejo desde el principio =="
[ -f "$OUT/ck_150000.pt" ] || cp runs/final/ck_150000.pt "$OUT/"
ls "$OUT"/ck_*.pt

lanzar() {
    caffeinate -i python -u entrenar.py --out "$OUT" --init runs/final/ck_150000.pt --resume --algo mixto \
        --start lichess --fens data/fens.txt --stockfish "$SF" \
        --workers 6 --games 150000 --checkpoint-every 2000
}

echo; echo "== 3. Lanzando entrenamiento ($(date)) | dispositivo: ${CHESS_DEVICE:-$DEV} =="
lanzar
STATUS=$?

# Si fallo por MPS y no se estaba ya en CPU, se relanza en CPU (el log del intento fallido se guarda aparte)
if [ $STATUS -ne 0 ] && [ -z "$CHESS_DEVICE" ] && grep -qi "mps" "$SALIDA"; then
    echo; echo "AVISO: el entrenamiento fallo con un error de MPS, se relanza en CPU ($(date))"
    [ -f "$OUT/log.csv" ] && mv "$OUT/log.csv" "$OUT/log_intento_mps.csv"
    export CHESS_DEVICE=cpu
    lanzar
    STATUS=$?
fi
echo "== Entrenamiento terminado con codigo $STATUS ($(date)) =="
