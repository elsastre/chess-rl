# CLAUDE.md: robot de ajedrez con aprendizaje por refuerzo

Contexto completo del proyecto para trabajar desde cualquier máquina (Windows con NVIDIA o Mac con Apple Silicon).
Leer entero antes de tocar código o lanzar corridas. Estado al 14/11/2026.

---

## 0. Qué es este proyecto

Trabajo final grupal de **Machine Learning** (Ingeniería en IA y Ciencia de Datos, UCU). Tema elegido:
**aprendizaje por refuerzo (RL)**. Un agente aprende a jugar al ajedrez jugando partidas, su aprendizaje se ve en
una curva de Elo, y se puede jugar contra él. Va dentro de un **robot**: cuerpo impreso en 3D, cámara que lee el
tablero, voz que dice la jugada y pantalla con una cara cuyas emociones salen de señales de RL
(valor = confianza, error TD = sorpresa).

- **Objetivo de fuerza:** unos 1000 de Elo. Hoy: ≈ 1230 estimado (escala Stockfish).
- **Entrega y presentación:** clase 31, **jueves 26/11/2026**. Informe + proyecto Python en este repo
  (**sin notebooks**). Tiene que tener marco teórico de los métodos y un caso de aplicación. Se evalúa la
  originalidad y se tiene en cuenta el tamaño del grupo. El tema (RL) se elige antes con la herramienta de la web
  del curso.
- **Personas:** Matihas (dueño del repo, entrenamiento principal en su laptop con RTX 4050), Valentino (laptop
  con RTX 5060: arranque supervisado y comparaciones), estudiantes de eléctrica (cuerpo, cámara, voz, pantalla).
- **Documentos:** `docs/Propuesta_robot_ajedrez_RL.pdf` (propuesta para el grupo) y
  `docs/Resumen_robot_ajedrez_RL.pdf` (explicación intuitiva + resultados, para alguien de afuera).

## 1. Reglas para trabajar en este repo

- **Idioma:** todo en español (código comentado en español, mensajes, documentos). Sin guion largo (raya) en
  textos en español: usar comas, paréntesis o reformular. Explicaciones simples e intuitivas cuando se piden.
- **Nunca subir a git:** `*.pt`, `*.pt.tmp`, `data/`, `checkpoints/`, `*.pgn`, `*.zst`, `*.npz`, `.venv/`.
  El `.gitignore` ya lo cubre. De `runs/` sí se suben `curva*.csv`, `curva*.png` y `log.csv`.
- **No tocar una carpeta `runs/<corrida>` mientras su entrenamiento está corriendo** (el entrenador escribe
  `latest.pt`, `log.csv` y `ck_*.pt` ahí, y los procesos que juegan leen `latest.pt`).
- **Dos máquinas nunca escriben en la misma carpeta de corrida.** Cada máquina usa su propio `--out`.
- Los resultados se reportan con **intervalo de confianza** (Wilson 95 %), no como número suelto: con 20 partidas
  por punto el ruido es grande.
- No hay notebooks: todo son scripts con `argparse`.

## 2. Mapa del repositorio

| Archivo | Qué hace | Estado |
|---|---|---|
| `encoding.py` | Tablero → 12 planos 8×8 desde el bando que mueve; `pack`/`unpack` (96 bytes); `terminal_value` | estable |
| `model.py` | `ValueNet` (ResNet chica, salida tanh en [−1, 1]), `save_checkpoint`, `load_checkpoint`, `default_device` | estable |
| `search.py` | **Búsqueda v2**: negamax de profundidad fija + quiescencia de capturas, evaluación de hojas en lote; recolecta muestras de TreeStrap | **en uso** |
| `entrenar.py` | **Entrenador v2 (núcleo de RL)**: TreeStrap / TD-Leaf / mixto, rivales, currículo de Stockfish, procesos en paralelo | **en uso** |
| `preentrenar_material.py` | Inicialización: la red aprende el valor del material (KnightCap, Giraffe) | usado para el bot final |
| `posiciones_lichess.py` | PGN de Lichess → archivo de FEN (`data/fens.txt`) para arrancar partidas desde posiciones reales | usado |
| `evaluar.py` | Evalúa checkpoints contra la escalera fija → `curva*.csv` | **en uso** |
| `graficar.py` | `curva*.csv` → `curva.png` (compara corridas) | en uso |
| `arena.py` | Rivales fijos (aleatorio, codicioso, minimax2, Stockfish limitado), `match`, `elo_diff` | estable |
| `interfaz.py` | Ventana (pygame) para jugar contra un checkpoint, con cara, chance, sorpresa y candidatas | **en uso** |
| `jugar.py` | Lo mismo en la terminal | en uso |
| `selfplay_td.py` | **v1**: autojuego con TD-Leaf(λ), búsqueda sin quiescencia (corrida `desde_cero`) | histórico, para comparar |
| `agent.py` | **v1**: `NetSearcher` / `NetPlayer` (búsqueda sin capturas) | histórico |
| `dataset_lichess.py` | PGN → dataset (posición, resultado) para supervisado | para Valentino |
| `train_supervised.py` | Arranque supervisado (predecir el resultado de partidas humanas), split por partida | para Valentino |
| `README.md` | Guía corta de la v1 + sección v2 | |
| `requirements.txt` | `chess numpy zstandard matplotlib pygame` (torch se instala aparte) | |

Carpetas no versionadas: `.venv/`, `data/` (`fens.txt`, PGN), `checkpoints/` (`material.pt`, `supervisado.pt`),
y los `*.pt` dentro de `runs/`.

Corridas en `runs/`:

| Carpeta | Qué es |
|---|---|
| `runs/desde_cero` | v1, TD-Leaf(λ) desde red al azar, 20.000 partidas (`selfplay_td.py`) |
| `runs/treestrap_cero` | v2, TreeStrap puro desde red al azar, 20.000 partidas: **colapsó** |
| `runs/final` | **Bot final**: init `checkpoints/material.pt`, algo `mixto`, posiciones de Lichess, sin Stockfish, 7 procesos, 150.000 partidas (≈ 12,3 h en la RTX 4050). Checkpoint del robot: `runs/final/ck_150000.pt` |

## 3. Arquitectura

### 3.1 Codificación (`encoding.py`)

- `encode(board)` → `uint8 (12, 8, 8)`. Si mueven las negras, el tablero se espeja (`board.mirror()`), así que
  **siempre** se ve desde el que mueve: planos 0 a 5 = peón, caballo, alfil, torre, dama, rey del que mueve;
  planos 6 a 11 = los del rival. Fila 0 del plano = fila 1 del tablero (desde el que mueve).
- No codifica enroque, al paso, regla de 50 jugadas ni repeticiones (limitación conocida, pendiente menor).
- `pack`/`unpack`: 768 bits ↔ 96 bytes, para que el buffer de 400.000 muestras entre en RAM.
- `terminal_value(board)`: valor exacto para el que mueve (+1, 0, −1) si la partida terminó (incluye mate,
  ahogado, material insuficiente, 75 jugadas, 5 repeticiones), si no `None`.

### 3.2 Red de valor (`model.py`)

- `ValueNet(channels=64, blocks=4)`: conv 3×3 + BN + ReLU → 4 bloques residuales (conv-BN-ReLU-conv-BN + atajo)
  → cabeza: conv 1×1 a 8 canales + BN + ReLU → lineal 512→128 → ReLU → lineal 128→1 → **tanh**.
  ≈ 369.000 parámetros. Salida: valor para el que mueve en [−1, 1]; chance de ganar = (v + 1) / 2.
- `self.last_features` guarda la última capa convolucional (pensado para Grad-CAM / explicabilidad).
- Checkpoint = `{"config", "state", "extra"}`. `extra["games"]` = partidas de entrenamiento al guardarlo.
- `default_device()`: `cuda` > `mps` (Mac) > `cpu`. Se fuerza con la variable de entorno
  `CHESS_DEVICE=cpu|cuda|mps`.

### 3.3 Búsqueda (`search.py`)

- `search(board, evaluate, depth=2, qdepth=4, collect=False, max_frontier_samples=32)` → `SearchResult`:
  - `value`: valor de la raíz para el que mueve.
  - `moves`: `[(jugada, puntaje)]` de cada hijo de la raíz.
  - `pv_leaf`: `(planos, signo)` de la hoja de la variante principal (la usa TD-Leaf).
  - `treestrap`: `[(planos, objetivo)]`, con **la raíz siempre en la posición 0**.
  - `n_leaves`.
- Cómo funciona:
  1. `build(d)` arma el árbol completo de profundidad fija (negamax, sin poda alfa-beta en el árbol principal,
     porque las hojas se evalúan todas juntas en un lote).
  2. En `d == 0` corre `qsearch` (solo **material**, en centipeones: P 100, C 320, A 330, T 500, D 900) para
     encontrar la secuencia de capturas que calma la posición, la juega y la **posición tranquila** resultante es
     la hoja que va a la red. `qsearch` usa material incremental, `generate_legal_captures`, orden por ganancia
     y **poda delta** (DELTA = 200).
  3. `evaluate(lista_de_planos)` evalúa todas las hojas en un solo lote (`make_evaluator`, en trozos de 4096).
  4. `backup` sube los valores con negamax; el signo de cada hoja depende de la paridad de la secuencia de capturas.
  5. Con `collect=True` guarda los planos de los nodos interiores (siempre) y de las hojas cuya secuencia de
     capturas cambió algo (hasta 32 al azar), con su valor de búsqueda como objetivo.
- `pick_move(moves, temperature)`: con 0 elige la mejor (desempate al azar); con T > 0, softmax de los puntajes.
- `SearchPlayer(model, device, depth, qdepth)`: jugador para la arena, el robot y la interfaz.
- Velocidad (CPU de la laptop): profundidad 1 + q4 ≈ 10 a 15 ms por jugada (entrenamiento);
  profundidad 2 + q4 ≈ 400 ms por jugada (robot, evaluación de fuerza, jugar contra personas).

### 3.4 Entrenamiento (`entrenar.py`)

**Una partida** (`play_game`):
- Arranque: con `--start lichess`, un FEN al azar de `--fens` + 1 jugada al azar (idea de Giraffe). Con
  `--start inicial`, la posición inicial + `--random-plies` jugadas al azar.
- Largo: contra sí mismo y desde Lichess, tramos de `--segment` (30) jugadas. Contra rivales, partidas completas
  hasta `--max-plies` (200), para que el currículo de Stockfish use resultados reales.
- Si se corta sin terminar, se **adjudica por material** (`material_value`: `qsearch` resuelve los cambios y se
  pasa a [−1, 1] con `tanh(cp · ln10 / 800)`). Opción `--adjudicar busqueda` usa el valor de la propia red
  (no recomendado: sin ancla externa puede colapsar).
- En cada jugada del agente: `search(..., collect=True)`, se guardan muestras de TreeStrap y el registro para TD.
- Elección de jugada: `pick_move` con `--temperature` 0,05 (exploración chica).

**Las dos señales de RL** (`--algo mixto`, el que se usa):
- **TreeStrap** (Veness, Silver, Uther, Blair 2009): cada muestra es `(planos, objetivo = valor de búsqueda)`.
  Por jugada entran como máximo `--ts-max` (8): **la raíz siempre** + 7 al azar entre las demás del árbol.
  (Hasta el 14/11 la raíz entraba solo por azar, ≈ 1 de cada 3 jugadas; se cambió antes de la corrida con
  Stockfish.)
- **TD(λ) / TD-Leaf(λ)** (Sutton 1988; Baxter, Tridgell, Weaver 2000): con los valores de la raíz de cada jugada
  `w_t` y el resultado `z`, se calculan los λ-retornos hacia atrás:
  `G_t = (1 − λ) · w_{t+1} + λ · G_{t+1}`, con `G_T = z`. El objetivo `G_t` se aplica a la **hoja de la
  variante principal** de esa jugada (TD-Leaf). λ = 0,7. Cada muestra de TD se repite `--td-peso` (4) veces.
- Valores en la partida se manejan desde blancas y se convierten al que mueve con un signo por muestra:
  muestra = `(planos_empaquetados, signo, objetivo)`; la pérdida compara `signo · red(x)` con el objetivo.
- **Por qué mixto:** TreeStrap solo colapsó a "todo vale 0" (punto fijo del bootstrapping: si la red dice 0,
  la búsqueda dice 0 y no hay error). TD trae el ancla externa (quién ganó). Monitor de colapso: el entrenador
  imprime `|valor|` promedio y avisa `OJO: valores cerca de 0, posible colapso` si baja de 0,02 después de 500
  partidas.

**Rivales** (`Opponents.pick`):
- Con probabilidad `--p-stockfish` (0,25): Stockfish con currículo. `SF_LEVELS` = lista de (Skill Level, nodos
  por jugada), de (0, 30) a (20, 100000). Cada 20 partidas contra Stockfish: si el agente sacó > 60 % sube un
  nivel, si sacó < 40 % baja uno. El nivel arranca en 0 en cada lanzamiento.
- Con `--p-viejos` (0,15): un checkpoint viejo al azar de la **misma carpeta** (`ck_*.pt`, sin el último; hace
  falta que haya al menos 2), con cache de 3 redes.
- El resto: contra sí mismo.

**Procesos** (`--workers N`):
- N procesos (`spawn`) juegan en CPU (`--worker-device cpu`, 1 hilo de torch cada uno) y mandan muestras por
  una cola. El proceso principal es el **aprendiz** (en `default_device()`: cuda, mps o cpu).
- El aprendiz guarda `latest.pt` cada `--sync-every` (20) partidas; los procesos lo recargan cuando cambia.
- Buffer de repetición `--buffer` 400.000 muestras; pasos de entrenamiento proporcionales a las muestras nuevas
  (`--replay` 2,0 usos promedio por muestra), lotes de `--batch` 512.
- Optimizador Adam, `--lr` 5e-4, weight decay 1e-5. Pérdida **Huber** (`smooth_l1`, beta 0,1).
- Checkpoints `ck_XXXXXX.pt` cada `--checkpoint-every`. `log.csv` con columnas:
  `partida, segundos, oponente, nivel_sf, plies, resultado, termino, muestras, loss, buffer, valor_abs`.
  (La columna `segundos` sale inflada; para velocidad usar las "partidas/h" que se imprimen.)
- `--init ck.pt --resume`: sigue contando desde `extra["games"]` de ese checkpoint y **no** vuelve a guardar el
  checkpoint inicial. `--games` son partidas **adicionales**. El buffer y el estado de Adam arrancan vacíos.

### 3.5 Evaluación (`evaluar.py`, `arena.py`, `graficar.py`)

- Escalera fija: `aleatorio`, `codicioso` (come lo que más gana, si no al azar), `minimax2` (2 jugadas, solo
  material), `stockfish1320` (`UCI_LimitStrength`, `UCI_Elo` 1320, 0,05 s por jugada).
- `evaluar.py --run R --games 20 --every K --depth D --q Q [--rivals ...] [--stockfish ruta]`:
  - con `--q` usa `SearchPlayer` (búsqueda v2) y escribe `curva_d{D}q{Q}.csv`; sin `--q` usa la v1 (`curva.csv`).
  - Solo juega lo que falta (checkpoint o rival nuevo) y reescribe el CSV ordenado: se puede cortar y reanudar.
- Puntaje: victoria 1, tablas 0,5, derrota 0. `elo_diff = 400 · log10(s / (1 − s))` (s recortado a
  [0,5/n, 1 − 0,5/n]). Elo del robot = 1320 + elo_diff contra Stockfish.
- Bandas: intervalo de Wilson al 95 %.
- `graficar.py runs/final runs/final/curva_d2q4.csv` → `curva.png`.

### 3.6 Interfaz y emociones (`interfaz.py`, `jugar.py`)

- `python interfaz.py --ck runs/final/ck_150000.pt [--depth 2] [--q 4]`: clic en pieza y clic en destino,
  jugadas legales marcadas, coronación siempre a dama, botones Nueva (blancas / negras) y Deshacer. El bot piensa
  en un hilo aparte.
- Panel: chance de ganar = (valor + 1) / 2, sorpresa = valor actual − valor anterior (error TD), jugadas
  candidatas en notación española (C, A, T, D, R), texto hablado ("Caballo come en e5, jaque").
- `emotion(valor, td)`: |td| > 0,16 → sorprendido; si no, con p = (valor + 1) / 2: p ≥ 0,75 ganando,
  ≥ 0,5 confiado, ≥ 0,3 preocupado, si no perdiendo. Son las mismas que usará la cara del robot.

## 4. Resultados hasta ahora (20 partidas por punto)

Desde cero (evaluado con profundidad 1 + q4):

| Corrida | vs codicioso | vs minimax2 |
|---|---|---|
| TD-Leaf v1 (`desde_cero`), 0 → 20.000 partidas | 0,05 → 0,85 | 0,0 → 0,175 |
| TreeStrap puro (`treestrap_cero`) | oscila, termina 0,225 | colapsa a 0,0 desde 12.000 |

Bot final (`runs/final`, init material + mixto):

| Medida | 0 partidas (sin RL) | 150.000 partidas |
|---|---|---|
| vs codicioso (d1q4) | 0,95 | 0,975 (saturado, ya no sirve) |
| vs minimax2 (d1q4) | 0,45 | 0,825 (≈ +300 Elo frente a ese rival) |
| vs Stockfish 1320 (d2q4, el robot) | 0,15 | 0,375 |
| Elo estimado del robot | ≈ 1020 (IC 817 a 1220) | ≈ 1230 (IC 1079 a 1384) |

Lectura honesta: la mejora contra minimax2 es clara; contra Stockfish los intervalos se solapan (hacen falta
≈ 100 partidas para confirmarlo). Es Elo en escala Stockfish, no rating humano. La curva no es monótona (caída a
0,55 en 120.000) por ruido y altibajos reales.

## 5. Decisiones de diseño (y por qué)

| Decisión | Motivo |
|---|---|
| RL basado en valor (no política) con búsqueda | Es lo que entrena en una laptop en horas (KnightCap, Meep); AlphaZero no es viable con este cómputo |
| Arranque con material, no desde cero | Desde cero aprende pero lento; KnightCap y Giraffe arrancan así. El RL cambia el objetivo de "material" a "quién gana", y se mide el antes y el después |
| Quiescencia de capturas | Sin ella la red evalúa posiciones en medio de un cambio (efecto horizonte) y aprende ruido |
| Profundidad 1 para entrenar, 2 para jugar | Profundidad 2 es ≈ 30 veces más lenta; con 1 se juegan muchas más partidas |
| Mixto (TreeStrap + TD) y adjudicación por material | TreeStrap solo colapsó; hace falta un ancla externa |
| Posiciones de Lichess + 1 jugada al azar | Variedad (si no, las partidas de autojuego se repiten) |
| Tramos de 30 jugadas | Más partidas y más resultados por hora |
| Versiones viejas como rivales | Evitar ciclos y olvido |
| Huber en vez de MSE | Los objetivos de TD tienen valores atípicos (Giraffe) |
| Workers en CPU + aprendiz en GPU | La búsqueda es Python puro (CPU); la GPU solo entrena |
| Intervalos de Wilson | Con 20 partidas por punto, sin intervalos se sobreinterpretan los saltos |

## 6. Puesta a punto en una Mac con Apple Silicon (M1)

```bash
# 1. Herramientas
xcode-select --install                 # si git no está
brew install python@3.12 stockfish     # Stockfish para ARM queda en /opt/homebrew/bin/stockfish

# 2. Repo y entorno
git clone <url-del-repo> chess-rl && cd chess-rl
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch                      # en Mac el paquete normal ya trae MPS (GPU de Apple)
pip install -r requirements.txt

# 3. Verificar
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"   # debe decir True
python -c "from model import default_device; print(default_device())"                  # debe decir mps
python arena.py --games 4 --stockfish /opt/homebrew/bin/stockfish
```

**Archivos que no están en git y hay que traer** (pendrive, Drive, etc.) desde la laptop de Matihas:
- `runs/final/ck_150000.pt` (el bot final, ≈ 1,5 MB). Para jugar o para seguir entrenando.
- `checkpoints/material.pt` (solo si se quiere entrenar desde el arranque con material).
- `data/fens.txt` (posiciones de Lichess). Si no se tiene, regenerarlo:
  ```bash
  mkdir -p data && cd data
  curl -LO https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
  cd .. && python posiciones_lichess.py data/lichess_db_standard_rated_2013-01.pgn.zst --out data/fens.txt
  ```

**Notas de la Mac:**
- Tiene 4 núcleos rápidos y 4 lentos: usar `--workers 6`. Esperar ≈ 1,5 a 2 veces más lento que la laptop con
  la 4050 (150.000 partidas con Stockfish: ≈ 22 a 30 h; medirlo con las "partidas/h" de las primeras 2.000).
- Si algo falla en la GPU de Apple (MPS), forzar CPU: `CHESS_DEVICE=cpu python entrenar.py ...`
  (el soporte MPS se agregó sin poder probarlo en una Mac; avisar si da error).
- Rutas con `/` en Mac (en Windows `\`). `interfaz.py` necesita `pygame` (está en requirements).
- Que la Mac no se duerma: `caffeinate -i python entrenar.py ...`.

**En Windows con NVIDIA** (laptops de Matihas y Valentino): torch desde el índice de CUDA que corresponda
(Matihas: `pip install torch --index-url https://download.pytorch.org/whl/cu132`), activar con
`.venv\Scripts\Activate.ps1` y usar rutas con `\`.

## 7. Recetas

```bash
# Jugar contra el bot final
python interfaz.py --ck runs/final/ck_150000.pt
python jugar.py --ck runs/final/ck_150000.pt --color negras

# Continuar el bot final con Stockfish como rival (cada máquina en su propia carpeta)
python entrenar.py --out runs/final_sf --init runs/final/ck_150000.pt --resume --algo mixto \
    --start lichess --fens data/fens.txt --stockfish /opt/homebrew/bin/stockfish \
    --workers 6 --games 150000 --checkpoint-every 2000
#   (en una carpeta nueva los "viejos" aparecen recién cuando hay 2 checkpoints; se puede copiar
#    ck_150000.pt adentro para tenerlo como rival desde el principio)

# Curvas de una corrida
python evaluar.py --run runs/final_sf --games 20 --depth 1 --q 4 --every 5
python evaluar.py --run runs/final_sf --games 20 --depth 2 --q 4 --rivals minimax2 --stockfish /opt/homebrew/bin/stockfish --every 10
python graficar.py runs/final_sf runs/final_sf/curva_d1q4.csv

# Más partidas contra Stockfish para cerrar los intervalos del robot (ej. 100 en ck_0 y ck_150000):
#   copiar esos dos .pt a una carpeta aparte y correr evaluar.py --games 100 --depth 2 --q 4 sobre ella

# Desde cero (experimentos de comparación)
python preentrenar_material.py --out checkpoints/material.pt --fens data/fens.txt
python entrenar.py --out runs/td_v2 --algo tdleaf --workers 6 --games 20000        # ablación TD-Leaf v2
python entrenar.py --out runs/lambda0 --init checkpoints/material.pt --algo mixto --lam 0.0 --start lichess --fens data/fens.txt --workers 6 --games 20000

# Supervisado (Valentino)
python dataset_lichess.py data/lichess_db_standard_rated_2013-01.pgn.zst --out data/lichess_2013_01.npz
python train_supervised.py data/lichess_2013_01.npz --out checkpoints/supervisado.pt
```

## 8. Pendientes (prioridad aproximada)

1. Elegir el tema RL en la web del curso (si no se hizo).
2. Correr la continuación con Stockfish (`runs/final_sf`) y evaluarla.
3. ≈ 100 partidas contra Stockfish en ck_0 y en el último checkpoint para cerrar los intervalos.
4. Análisis "qué aprendió el RL": comparar la red de ck_0 y la final en posiciones de prueba (peón pasado, rey
   expuesto, actividad de piezas), y Grad-CAM con `last_features`.
5. Comparaciones para el informe: λ (0 / 0,7 / 1), TD-Leaf v2 vs mixto, arranque supervisado vs material.
6. Robot: `robot.py` con `elegir_jugada(tablero) -> (jugada, valor, sorpresa, emocion)`; visión
   `leer_tablero()` / `detectar_jugada()`; cara y voz `mostrar(emocion, texto)`. Cámara a 45 a 50 cm, ≈ 52°.
7. Codificar enroque / al paso / turno (planos extra) si sobra tiempo.
8. Informe (marco teórico: MDP, valor, TD(λ), TD-Leaf, TreeStrap, búsqueda minimax y quiescencia; caso de
   aplicación: el robot) y presentación del 26/11.

## 9. Problemas conocidos y soluciones

| Síntoma | Causa / solución |
|---|---|
| `torch.cuda.is_available()` da False en Windows | Se instaló torch de PyPI (solo CPU). Reinstalar desde el índice de CUDA |
| `|valor|` del log cae a ≈ 0 | Colapso. Revisar que `--algo` sea `mixto` o `tdleaf` y `--adjudicar material` |
| "los procesos que juegan se cerraron" | Un worker tiró error (ruta de Stockfish o de `--fens` mal). Probar con `--workers 0` para ver el error |
| `PermissionError` al guardar `latest.pt` en Windows | `safe_save` reintenta; si persiste, antivirus o un proceso leyendo el archivo |
| Push rechazado | `git pull --rebase` y `git push` |
| La interfaz no abre ventana | `pip install pygame`; en Mac correrla desde la Terminal con el venv activado |
| `UCI_Elo` fuera de rango | Algunas versiones de Stockfish tienen otro mínimo; `StockfishPlayer` ya lo ajusta |

## 10. Referencias

- Sutton y Barto (2018). *Reinforcement Learning: An Introduction*, 2.ª ed.
- Sutton (1988). Learning to predict by the methods of temporal differences. *Machine Learning*.
- Tesauro (1995). Temporal difference learning and TD-Gammon. *CACM*.
- Baxter, Tridgell y Weaver (2000). Learning to play chess using temporal differences (KnightCap, TD-Leaf). *Machine Learning*.
- Veness, Silver, Uther y Blair (2009). Bootstrapping from game tree search (TreeStrap, Meep). *NeurIPS*.
- Lai (2015). Giraffe: using deep reinforcement learning to play chess. Imperial College London.
