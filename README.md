# chess-rl: un agente que aprende a jugar al ajedrez con TD-Leaf(λ)

## Archivos

| Archivo | Qué hace |
|---|---|
| `encoding.py` | Tablero → 12 planos de 8x8, siempre desde el bando que mueve |
| `model.py` | Red de valor (ResNet chica, salida en [-1, 1]) y checkpoints |
| `agent.py` | Búsqueda negamax con la red + jugador para la arena |
| `arena.py` | Rivales fijos (aleatorio, codicioso, minimax2, Stockfish) y cálculo de Elo |
| `dataset_lichess.py` | Partidas de Lichess → dataset (posición, resultado) |
| `train_supervised.py` | Arranque supervisado (imitación) |
| `selfplay_td.py` | **Autojuego con TD-Leaf(λ): el núcleo de RL** |
| `evaluar.py` | Evalúa todos los checkpoints de una corrida → `curva.csv` |

## Instalación

```
pip install chess numpy zstandard
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(Para torch, usar el comando exacto que da pytorch.org según la versión de CUDA de cada laptop.)
Verificar la GPU: `python -c "import torch; print(torch.cuda.is_available())"`

## Orden de ejecución

**1. Comprobar la arena** (segundos):
```
python arena.py
```

**2. Corrida desde cero** (laptop 2, se deja corriendo):
```
python selfplay_td.py --out runs/desde_cero
```

**3. Arranque supervisado** (laptop 1):
```
python dataset_lichess.py lichess_db_standard_rated_2013-01.pgn.zst --out data/lichess_2013_01.npz
python train_supervised.py data/lichess_2013_01.npz --out checkpoints/supervisado.pt
python selfplay_td.py --out runs/con_imitacion --init checkpoints/supervisado.pt
```

**4. Curva de aprendizaje** (cada noche, sobre cada corrida):
```
python evaluar.py --run runs/desde_cero --games 40
python evaluar.py --run runs/con_imitacion --games 40
```

## Tiempos medidos (CPU, sin GPU, en la máquina de prueba)

- Autojuego con profundidad 1: alrededor de 1 segundo por partida.
- Autojuego con profundidad 2: unos 45 segundos por partida de 200 jugadas. Usar profundidad 1
  para entrenar y profundidad 2 para evaluar o jugar contra personas.
- Con GPU debería ir más rápido, porque las hojas se evalúan en batch. Medirlo en la primera corrida.

## Hiperparámetros que conviene explorar (para el informe)

- `--lam`: 0 (TD puro), 0.7 (por defecto), 1 (Monte Carlo). Ablación natural.
- `--temperature`: exploración. Si las partidas se repiten mucho, subirla.
- `--max-plies`: las partidas que llegan al tope cuentan como tablas.
- `--seed`: correr al menos 2 semillas de la corrida principal si el tiempo da.

## Pendientes conocidos

- Banco de oponentes viejos (para evitar ciclos y olvido): hoy es autojuego puro.
- Paralelizar el autojuego en varios procesos.
- Planos extra en la codificación (derechos de enroque, turno, regla de 50 jugadas).


## Versión 2 (13/11): búsqueda con capturas + TreeStrap + rivales

| Archivo nuevo | Qué hace |
|---|---|
| `search.py` | Búsqueda con capturas (quiescence): la red solo evalúa posiciones tranquilas |
| `entrenar.py` | Entrenador v2: `--algo treestrap / tdleaf / mixto`, rivales mezclados, Stockfish con currículo, `--workers` |
| `posiciones_lichess.py` | Extrae posiciones reales (FEN) para arrancar el autojuego desde ahí |
| `preentrenar_material.py` | Inicialización pura: la red aprende el valor del material (KnightCap, Giraffe) |

`selfplay_td.py` queda como la versión 1 (TD-Leaf sin capturas), para comparar.

### Preparación (una vez)
```
python posiciones_lichess.py lichess_db_standard_rated_2013-01.pgn.zst --out data/fens.txt
python preentrenar_material.py --out checkpoints/material.pt --fens data/fens.txt
```

### Corridas
`--workers` = núcleos físicos menos 1 (i3-9100F: 3; laptops: 5 o 6).
```
# Comparación de algoritmos: TreeStrap desde cero (contra la corrida v1 de TD-Leaf)
python entrenar.py --out runs/treestrap_cero --algo treestrap --workers 5

# Bot final: arranque + posiciones reales + rivales + Stockfish con currículo
python entrenar.py --out runs/final --init checkpoints/material.pt --algo mixto --start lichess --fens data/fens.txt --stockfish "C:\ruta\stockfish.exe" --workers 3
```
Si se corta: mismo comando con `--init runs/X/ck_XXXXXX.pt --resume`.

### Evaluar (siempre con la búsqueda nueva para comparar parejo)
```
python evaluar.py --run runs/final --every 4 --games 20 --depth 1 --q 4 --rivals codicioso,minimax2
python graficar.py runs/final/curva_d1q4.csv runs/treestrap_cero/curva_d1q4.csv
```

### Prueba de control (evaluador = material exacto, 10 partidas)
| Búsqueda | vs codicioso | vs minimax2 |
|---|---|---|
| 1 jugada, sin capturas (como v1) | 0,40 | 0,00 |
| 1 jugada + capturas | 1,00 | 0,75 |
| 2 jugadas + capturas | 0,95 | 0,90 |
