# CLAUDE.md: robot de ajedrez con aprendizaje por refuerzo

Contexto completo del proyecto para trabajar desde cualquier máquina (Windows con NVIDIA o Mac con Apple Silicon).
Leer entero antes de tocar código o lanzar corridas. Estado al 19/11/2026 (sección 4.4): el RL mejora al agente
de forma significativa (+36 Elo sobre `ck_000000`), la continuación a profundidad 1 suma poco (+31) y se estanca,
y la corrida con las mejoras (profundidad 2) da **+114 Elo** sobre 150k y **sigue mejorando**. Mejor red medida:
`runs/mejoras/ck_180000.pt`. La corrida sigue: ya están 185k, 190k y 195k en la Mac (sin medir) y llega a 210k la
noche del 19/11. Qué falta y en qué orden: **pendiente 1** (sección 8).

---

## 0. Qué es este proyecto

Trabajo final grupal de **Machine Learning** (Ingeniería en IA y Ciencia de Datos, UCU). Tema elegido:
**aprendizaje por refuerzo (RL)**. Un agente aprende a jugar al ajedrez jugando partidas, su aprendizaje se ve en
una curva de Elo, y se puede jugar contra él. Va dentro de un **robot**: cuerpo impreso en 3D, cámara que lee el
tablero, voz que dice la jugada y pantalla con una cara cuyas emociones salen de señales de RL
(valor = confianza, error TD = sorpresa).

- **Objetivo de fuerza:** unos 1000 de Elo. Medido con `ck_150000`: ≈ 1240 (IC 95 %: 1200 a 1280, 300 partidas,
  escala Stockfish por tiempo). `runs/mejoras/ck_180000.pt` le gana a 150k por +114 Elo en enfrentamiento directo;
  su Elo absoluto contra Stockfish todavía no se midió.
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
| `search.py` | **Búsqueda v2**: negamax de profundidad fija + quiescencia de capturas, evaluación de hojas en lote; recolecta muestras de TreeStrap. Opcionales (15/11): detección de mate en 1 (`mates`) y profundidad 2 con poda alfa-beta (`poda`) | **en uso** |
| `entrenar.py` | **Entrenador v2 (núcleo de RL)**: TreeStrap / TD-Leaf / mixto, rivales, currículo de Stockfish, procesos en paralelo. Opciones nuevas (15/11): `--poda --mates --sf-azar --temp-stockfish --p-completas --lr-final` y buffer en numpy | **en uso** |
| `preentrenar_material.py` | Inicialización: la red aprende el valor del material (KnightCap, Giraffe) | usado para el bot final |
| `posiciones_lichess.py` | PGN de Lichess → archivo de FEN (`data/fens.txt`) para arrancar partidas desde posiciones reales | usado |
| `evaluar.py` | Evalúa checkpoints contra la escalera fija → `curva*.csv`. Opciones nuevas: `--poda`, `--mates`, `--sf-nodes` | **en uso** |
| `lanzar_final_sf.sh`, `lanzar_mejoras.sh` | Scripts de la Mac: verifican el entorno y lanzan una corrida con `caffeinate` (rutas de Homebrew) | solo Mac |
| `medir_valor.py` | **Medición 1 (17/11)**: qué tan bien predice cada red el resultado de partidas que no vio (Brier, LogLoss, AUC, calibración), con IC por bootstrap de partidas (sección 4.4) | **en uso** |
| `duelo.py` | **Medición 2 (17/11)**: enfrentamiento directo entre dos redes con la misma búsqueda, aperturas pareadas (cada una con los dos colores), Elo con IC; reanudable | **en uso** |
| `graficar_mediciones.py` | Gráfico resumen de 4.4 desde los CSV (`runs/mediciones/resumen_atribucion.png`); lo llama el lanzador | en uso |
| `lanzar_mediciones.sh` | Corre las dos mediciones en orden en la Mac (regenera la red de material si falta `ck_000000.pt`) | solo Mac |
| `graficar.py` | `curva*.csv` → `curva.png` (compara corridas) | en uso |
| `arena.py` | Rivales fijos (aleatorio, codicioso, minimax2, Stockfish limitado), `match`, `elo_diff` | estable |
| `interfaz.py` | Ventana (pygame) para jugar contra un checkpoint, con cara, chance, sorpresa y candidatas | **en uso** |
| `jugar.py` | Lo mismo en la terminal | en uso |
| `vision/detectar_jugadas.py` | Detecta las jugadas de un tablero real con la webcam (casillas que cambian + jugadas legales), ventanas de OpenCV | en uso |
| `chesscam/chesscam.py` | **ChessCam** (aporte de un compañero del grupo, integrado el 17/11): ventana tkinter con tablero digital, calibración con lupa y ajuste automático, detección con color + textura (`board_vision.py`). Contra el bot: flecha con la jugada que hay que hacer en el tablero real, puntos de jugadas legales, pista; también sin cámara (`--sin-camara`) | **en uso** |
| `chesscam/partida.py` | Turnos contra el bot para ChessCam, sin interfaz (se prueba sola): jugada pendiente, descripción en español, deshacer | en uso |
| `chesscam/camtest.py` | Diagnóstico: qué índice y backend de cámara entrega imagen (Windows, Mac y Linux) | en uso |
| `chesscam/prueba_robustez.py` | Prueba sin cámara de la detección de ChessCam con exposición, franjas de luz, corrimientos, escena quieta y piezas negras con poco contraste; detección manual o `--auto`, filtros con `--filtro` (3.7) | en uso |
| `chesscam/diagnostico_negras.py` | Sobre capturas guardadas con `--capturas`: aciertos y señal por pieza negra/blanca y casilla oscura/clara, comparando variantes del filtro (3.7) | en uso |
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
| `runs/final_sf` | Continuación de `final` con Stockfish como rival, 150.000 a 300.000 partidas, corrida en la Mac (M1, ≈ 19.000 partidas/h, 7,8 h). **Mejoró +31 Elo** en enfrentamiento directo contra 150k; contra Stockfish con 100 partidas no se veía (4.1 y 4.4) |
| `runs/confirmar` | `ck_150000`, `ck_230000` y `ck_300000` de `final_sf` evaluados con 100 partidas por rival |
| `runs/mejoras` | Corrida con las mejoras de 4.3, a profundidad 2 en la laptop con GPU: 150.000 a 180.000 (15 a 17/11), sigue hasta 195.000 (19/11) y 210.000 (noche del 19/11; pendiente 1). **La mejor**: +114 Elo contra 150k y +95 contra `final_sf` 300k, y la predicción sigue subiendo en cada checkpoint (4.4). **Checkpoint para el robot: `runs/mejoras/ck_180000.pt`** |
| `runs/mediciones` | Mediciones para atribuir al RL (4.4): `valor.csv`, `valor_calibracion.png`, `duelo_*.csv` y resúmenes, `resumen_atribucion.png` |
| `runs/final` | **Bot final**: init `checkpoints/material.pt`, algo `mixto`, posiciones de Lichess, sin Stockfish, 7 procesos, 150.000 partidas (≈ 12,3 h en la RTX 4050). Fue el checkpoint del robot (`ck_150000.pt`) hasta el 18/11 |

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
- **`mates=True`** (15/11): en cada hoja se mira si el que mueve tiene mate en 1 (`mate_en_1`, valor exacto +1).
  Sin esto, a profundidad 1 la búsqueda no ve un mate del rival que no sea captura. Costo: ≈ 15 % más de tiempo.
- **`poda=True`** (15/11, solo profundidad 2): poda alfa-beta en tres pasos para seguir evaluando en lote.
  (1) La mejor candidata según una búsqueda de profundidad 1 se mira entera. (2) De las demás se miran las
  primeras 4 respuestas, todas en un lote. (3) Las que no quedaron refutadas se siguen de a tandas.
  Da **el mismo valor de la raíz** que el árbol completo (verificado en 150 posiciones de Lichess, diferencia
  máxima 2e-7). Los puntajes de las jugadas refutadas son cotas superiores, no valores exactos. Para TreeStrap
  solo se usan la raíz, los hijos con valor exacto y las hojas con cambios.
- `pick_move(moves, temperature)`: con 0 elige la mejor (desempate al azar); con T > 0, softmax de los puntajes.
- `SearchPlayer(model, device, depth, qdepth)`: jugador para la arena, el robot y la interfaz.
- Velocidad (CPU de la laptop): profundidad 1 + q4 ≈ 10 a 15 ms por jugada (entrenamiento);
  profundidad 2 + q4 ≈ 400 ms por jugada (robot, evaluación de fuerza, jugar contra personas).
  Con `poda`, la profundidad 2 evalúa ≈ 250 posiciones en vez de ≈ 1070: unas 4 veces más rápida que la
  completa y unas 8 veces más lenta que la profundidad 1.

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
  Con `--sf-azar` se agregan 6 escalones más fáciles al principio (`SF_AZAR`): el Stockfish más débil juega una
  jugada al azar con probabilidad 0,8 / 0,6 / 0,45 / 0,3 / 0,2 / 0,1. **Ojo:** con esa opción los niveles se
  corren 6 lugares (el viejo nivel 0 pasa a ser el nivel 6). Con `--temp-stockfish 0` el agente juega sin
  exploración contra Stockfish, así el currículo mide su fuerza real.
- Con `--p-viejos` (0,15): un checkpoint viejo al azar de la **misma carpeta** (`ck_*.pt`, sin el último; hace
  falta que haya al menos 2), con cache de 3 redes.
- El resto: contra sí mismo.

**Procesos** (`--workers N`):
- N procesos (`spawn`) juegan en CPU (`--worker-device cpu`, 1 hilo de torch cada uno) y mandan muestras por
  una cola. El proceso principal es el **aprendiz** (en `default_device()`: cuda, mps o cpu).
- El aprendiz guarda `latest.pt` cada `--sync-every` (20) partidas; los procesos lo recargan cuando cambia.
- Buffer de repetición `--buffer` 400.000 muestras (desde el 15/11 es un buffer circular en numpy, 104 bytes
  por muestra: 2 millones ocupan 208 MB); pasos de entrenamiento proporcionales a las muestras nuevas
  (`--replay` 2,0 usos promedio por muestra), lotes de `--batch` 512.
- Optimizador Adam, `--lr` 5e-4, weight decay 1e-5. Con `--lr-final` la tasa baja en línea recta hasta ese
  valor a lo largo de la corrida. Pérdida **Huber** (`smooth_l1`, beta 0,1).
- Checkpoints `ck_XXXXXX.pt` cada `--checkpoint-every`. `log.csv` con columnas:
  `partida, segundos, oponente, nivel_sf, plies, resultado, termino, muestras, loss, buffer, valor_abs`.
  (La columna `segundos` sale inflada; para velocidad usar las "partidas/h" que se imprimen.)
- `--p-completas P`: esa fracción de las partidas de autojuego se juega entera (hasta `--max-plies`) en vez de
  un tramo de 30 jugadas, para tener más resultados reales y menos adjudicaciones por material.
- `--depth 2 --poda --mates`: entrenar a profundidad 2 con poda y detección de mates (sección 3.3).
- `--temp-jugadas N` (15/11): la exploración (`--temperature`) solo se usa en las primeras N jugadas de cada
  partida o tramo; después el agente juega su mejor jugada. Por defecto, exploración en todas las jugadas.
- Al reanudar (`--resume`) se recuperan del checkpoint el **nivel de Stockfish** (si `--sf-azar` coincide) y el
  **plan de `--lr-final`** (`extra["lr_plan"]` = partida de origen, largo, lr, lr_final), así cortar y seguir no
  reinicia el currículo ni la tasa. El buffer de repetición sí arranca vacío.
- `--init ck.pt --resume`: sigue contando desde `extra["games"]` de ese checkpoint y **no** vuelve a guardar el
  checkpoint inicial. `--games` son partidas **adicionales**. El buffer y el estado de Adam arrancan vacíos.

### 3.5 Evaluación (`evaluar.py`, `arena.py`, `graficar.py`)

- Escalera fija: `aleatorio`, `codicioso` (come lo que más gana, si no al azar), `minimax2` (2 jugadas, solo
  material), `stockfish1320` (`UCI_LimitStrength`, `UCI_Elo` 1320, 0,05 s por jugada).
- `evaluar.py --run R --games 20 --every K --depth D --q Q [--rivals ...] [--stockfish ruta]`:
  - con `--q` usa `SearchPlayer` (búsqueda v2) y escribe `curva_d{D}q{Q}.csv`; sin `--q` usa la v1 (`curva.csv`).
  - Solo juega lo que falta (checkpoint o rival nuevo) y reescribe el CSV ordenado: se puede cortar y reanudar.
- `--poda`: profundidad 2 con poda (mismas jugadas salvo empates, 4 veces más rápido). `--mates`: el agente
  detecta mates en 1 (escribe `curva_d{D}q{Q}m.csv`, archivo aparte). `--sf-nodes N`: Stockfish piensa N nodos
  por jugada en vez de 0,05 s (columna `stockfish1320nN`). Con tiempo fijo la fuerza de Stockfish depende de
  la máquina: `ck_150000` sacó 0,375 en la laptop y 0,15 en la Mac con 20 partidas.
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

### 3.7 ChessCam: tablero físico o simulador contra el bot (`chesscam/`)

Se corre desde la carpeta del repo como script (`python chesscam/chesscam.py ...`), no con `-m`: así
`board_vision.py` se encuentra en `chesscam/` y no se confunde con la carpeta `vision/`. La calibración se
guarda en `chesscam/calib.json` (propia de cada máquina, no se sube).

- **Sin `--ck`**: lo de siempre, seguimiento de una partida entre dos personas.
- **Con `--ck` y cámara**: el bot **no mueve en la pantalla**. Su jugada queda pendiente y se muestra con una
  flecha verde (anillo en el origen, anillo rojo si come, segunda flecha para la torre en el enroque, cruz en el
  peón que se saca al paso) y descrita en el panel ("Caballo de g1 a f3"). Se aplica cuando la cámara la
  detecta. Si se detecta otra jugada, se avisa y no se aplica (la referencia no cambia: se corrige la pieza y
  se reevalúa). Botón «Bot: hecha» por si la cámara no la ve.
- **Con `--ck --sin-camara`**: el simulador solo. Clic en una pieza (puntos en sus destinos legales, anillo si
  come) y clic en el destino; el bot responde en la pantalla y su flecha queda marcada hasta tu jugada.
- **Pista**: flecha azul con lo que jugaría el bot en tu lugar. **Deshacer** vuelve a tu turno (deshace
  también la respuesta del bot).
- El bot usa `search` con `mates=True` y `poda` a profundidad 2, en un hilo aparte; el panel muestra chance de
  ganar y emoción (las mismas reglas que 3.6).
- Probado el 17/11 con la cámara simulada de `vision/prueba_simulada.py`: la jugada equivocada se rechaza, la
  del bot se acepta y la partida queda igual en pantalla y en el tablero.

**Robustez para una webcam real (19/11, rama `vision-robusta`).** Con la cámara real aparecían "cambios" en
casillas que nadie tocó: la webcam reajusta sola exposición, foco y balance de blancos, las luces LED
parpadean (franjas a 50 Hz) y el brazo puede correrse 1 o 2 píxeles. Cambios (todos configurables en el
constructor de `BoardVision`; `BoardVision(**VISION_ORIGINAL)` es el detector de antes):
- `promedio=8`: la referencia y la imagen que se compara al quedar quieta son el promedio de los últimos 8
  cuadros quietos (la vista en pantalla sigue cuadro a cuadro).
- `espera_inicial=60`: al arrancar o recalibrar se esperan ~2 s antes de la primera referencia.
- `normalizar_movimiento=True` y `motion_thresh=2.5`: el detector de movimiento resta el brillo medio de cada
  cuadro y divide por su desvío. En la simulación con todos los problemas, el ruido quieto da < 0,8 y la mano
  > 3 (antes el ruido llegaba a 8,9 y la mano bajaba a 2,2: se mezclaban).
- `radio_tolerancia=2`: la diferencia con la referencia es `max(0, nuevo - dilatar(ref)) + max(0,
  erosionar(ref) - nuevo)` con un núcleo de 5×5 (sobre 60 px por casilla), en L, a, b y el gradiente: un
  corrimiento de unos píxeles da 0, una pieza que aparece o se va no.
- Opciones de `chesscam.py`: `--fijar-camara` (30 cuadros en automático y después apaga autofoco, balance de
  blancos y exposición automática; imprime qué aceptó el driver), `--exposicion N` (valor fijo; con
  DirectShow suele ser una potencia de 2 negativa: -5, -6, -7) y `--vision-original` (para comparar).

Prueba sin cámara: `python chesscam/prueba_robustez.py [--vision-original]`. Usa la lógica real de la
aplicación (`App.process` y `App.detect_move`) con la cámara simulada de `vision/prueba_simulada.py`, una
partida de 23 jugadas (enroques, capturas) y 10 s quieto antes de empezar. Si una jugada no se detecta, se
carga a mano y se sigue (cada jugada se mide por separado). Resultados (19/11):

| Escenario | Original: bien | Original: avisos / ignoradas / nunca quieta | Nueva: bien | Nueva: avisos |
|---|---|---|---|---|
| Sin problemas (sin el ±3 %) | 23/23 | 0 / 0 / 0 | 23/23 | 0 |
| Solo el ±3 % por cuadro de `prueba_simulada` | **1/23** (nunca toma la referencia) | 0 / 0 / 22 | 23/23 | 0 |
| (a) Exposición al entrar la mano (ganancia 0,85 a 1,15 y curva que satura) | 22/23 | 1 / 0 / 0 | 23/23 | 0 |
| (b) Franjas de brillo que se mueven (50 Hz) | 23/23 | 0 / 0 / 0 | **21/23** | 2 |
| (c) Corrimiento de 1 a 3 px después de cada jugada | **12/23** | 2 / 9 / 0 | 23/23 | 0 |
| Todos juntos (a + b + c + ±3 %) | **3/23** (nunca toma la referencia) | 1 / 0 / 19 | **21/23** | 2 |

En ningún escenario, en ninguna de las dos versiones, aparecieron jugadas ni avisos en los 10 s quieto, ni se
detectó una jugada equivocada. "Ignoradas" = la imagen quedó quieta pero el detector tomó el cambio por ruido y
refrescó la referencia sin avisar (con el corrimiento cambian todas las casillas y la jugada no se destaca).
La demora desde que sale la mano pasa de ≈ 0,3 s a ≈ 0,4 s. **Punto débil de la versión nueva:** con franjas
que se mueven hay 2 avisos de "no estoy seguro" que el original no tenía (probablemente porque el promedio de 8
cuadros mezcla franjas en otra posición que la de la referencia); son avisos, no jugadas equivocadas, y se
resuelven con un clic. No se probó con una webcam real (el entorno donde se hizo no tiene cámara).

**Detección manual por ráfaga (19/11, por defecto).** Ya no se detecta sola al quedar quieta la imagen: después
de jugar se pulsa «Detectar jugada» o la barra espaciadora. `BoardVision.iniciar_rafaga(15)` junta 15 cuadros
quietos (con 3 de margen después del último movimiento; si hay movimiento vuelve a empezar), los combina con la
**mediana por píxel** y evalúa una vez; esa mediana también queda como referencia siguiente. Si no ve cambios
avisa "No veo cambios" y **no** refresca la referencia (antes la refrescaba en silencio y una jugada que cambió
poco se perdía). Si en ≈ 5 s no hay quietud, avisa. `--auto` vuelve al comportamiento anterior. Los cambios por
casilla se calculan solo en la ráfaga (o en cada cuadro si se mira la vista "Rectificada + cambios").
Probado con la webcam real (`--cam 1`, 19/11): funciona; con `calib.json` viejo hubo dudas entre e4 y h4
(conviene recalibrar al cambiar de cámara o de posición).

**Piezas negras (19/11).** Filtros opcionales en `BoardVision` (apagados por defecto): `gamma_sombras` (gamma
< 1 antes del CLAHE), `peso_textura` (diferencia del desvío local del brillo en 5×5), `peso_grad` y
`refuerzo_oscuras`. `--filtro-negras` activa `FILTRO_NEGRAS` (gamma 0,6 y textura 0,5, **provisorio**).
Para elegir con datos reales: `chesscam.py --capturas` guarda cada detección (referencia, imagen, posición y la
jugada que terminó aplicándose, detectada o cargada a mano; al deshacer se renombra `_deshecha`) en
`chesscam/capturas/<fecha>/` (no se sube), y `python chesscam/diagnostico_negras.py chesscam/capturas
[--imagenes DIR]` compara variantes: aciertos, margen y señal (z) de las casillas de la jugada por grupo (pieza
negra o blanca × casilla oscura o clara) y por canal. Solo usa la pista de cambio (no la de casilla vacía).

Prueba simulada con la detección manual (semilla 0; las filas de (e) y "todos" en manual no terminaron):

| Escenario | Automática: bien | Manual: bien | Manual + `--filtro-negras`: bien |
|---|---|---|---|
| Sin problemas, ±3 %, (a) exposición, (c) corrimiento | 23/23 | 23/23 | 23/23 |
| (b) Franjas 50 Hz | 21/23 | 22/23 | **23/23** |
| Todos juntos | 21/23 | 21/23 | 21/23 |
| (e) Negras con poco contraste | 23/23 | sin terminar | 23/23 |

Ninguna jugada equivocada ni inventada en los 10 s quieto (en manual se pulsa una vez sin jugar). La demora
desde que sale la mano pasa a ≈ 0,53 s. El escenario (e) resultó demasiado fácil (lo detecta hasta la
automática): todavía no muestra si el filtro ayuda; hace falta uno más difícil o las capturas reales.
`prueba_robustez.py` acepta `--auto`, `--filtro PARAM=VALOR ...`, `--solo TEXTO` y `--capturas DIR`, y genera el
ruido con OpenCV en float32 (≈ 1,5 min por escenario en vez de 10).

**Reloj de ajedrez (19/11, `--reloj MIN+INC`, opcional).** Ejemplo: `--reloj 5+3` = 5 minutos por jugador y
3 s de incremento por jugada (Fischer); también `--reloj 10` o `--reloj 0.5+2`. Corre el reloj del bando que
tiene que mover. «Detectar jugada» (espacio) hace de botón del reloj: si la cámara confirma una jugada legal, se
suma el incremento a quien jugó y pasa a correr el del rival; si no ve cambios, no está segura o la jugada es
ilegal, sigue corriendo el de quien tenía que jugar (como en un reloj real, no se aprieta hasta terminar). Una
jugada cargada con clic o con «Bot: hecha» también aprieta el reloj. Contra el bot: su reloj corre mientras
piensa y, con cámara, hasta que la persona hace su jugada en el tablero y la confirma (en el simulador, al
jugar). Al llegar a 0 pierde ese bando, salvo que al rival no le alcance el material para dar mate
(`board.has_insufficient_material`): tablas, como en el reglamento FIDE. Con cámara arranca cuando hay
referencia (y después de sincronizar en una partida nueva); se pausa solo mientras está abierta la calibración y
con el botón «Pausa». La derrota por tiempo no se deshace (hay que empezar una «Nueva partida»). El PGN guarda
`Result`, `TimeControl` y `Termination`. Lógica en `partida.py` (`Reloj`, `leer_control`, `formato_tiempo`,
`Partida.revisar_reloj`, `resultado_pgn`); interfaz en `App.tick_reloj`/`draw_reloj`. Probado con un tiempo
simulado (incremento, pausa, jugada inválida, bandera, material insuficiente, bot con y sin cámara) y con la
ventana bajo xvfb; todavía no con la webcam real.

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

### 4.1 Continuación con Stockfish (`runs/final_sf`, 15/11)

150.000 partidas más desde `ck_150000`, con Stockfish (25 %), versiones viejas (15 %) y autojuego (60 %).

- **El currículo nunca salió del nivel 0.** En 37.795 partidas contra Stockfish nivel 0 el agente sacó 21 % al
  principio y 22,5 % al final (hace falta más de 60 % para subir).
- `|valor|` estable en ≈ 0,59 (sin colapso), pérdida estable en ≈ 0,049.
- **Contra Stockfish no se veía cambio** con 100 partidas por punto (`runs/confirmar`, profundidad 2 + q4, en la Mac).
  **Corrección (18/11): sí mejoró.** En enfrentamiento directo (1.000 partidas) 300k le gana a 150k por **+31 Elo
  [+20; +43]**, y predice mejor el resultado (sección 4.4). Esta prueba no tenía potencia para verlo:

| Checkpoint | vs minimax2 | vs Stockfish 1320 | IC Wilson 95 % | Elo estimado |
|---|---|---|---|---|
| 150.000 | 0,92 | 0,375 | 0,29 a 0,47 | ≈ 1230 |
| 230.000 | 0,875 | 0,39 | 0,30 a 0,49 | ≈ 1240 |
| 300.000 | 0,86 | 0,39 | 0,30 a 0,49 | ≈ 1240 |

Juntando las 300 partidas: 0,385, **≈ 1240 de Elo (IC 1200 a 1280)**. Con la medición más fina, `final_sf`
300k supera a 150k, pero la mejor red medida es `runs/mejoras/ck_180000.pt` (4.4).

**Lección de medición:** las curvas de 20 partidas (`runs/final_sf/curva_d2q4.csv`, `curva_fuerza.png`)
mostraban un pico de 0,65 en 230.000 y una caída a 0,20 en 300.000. Con 100 partidas el pico desapareció:
era ruido. No sacar conclusiones de puntos sueltos de 20 partidas.

Gráficos del log en `runs/final_sf`: `entrenamiento_stockfish.png` y `entrenamiento_loss_valor.png`.

### 4.2 Diagnóstico: por qué se estancó (15/11, hecho con Claude Fable 5.2 sobre el código y los logs)

**Matiz (18/11):** el estancamiento no fue total. La continuación aprendió algo entre 150k y 200k (+31 Elo,
AUC +0,008) y desde 200k la predicción ya no mejora (200k, 250k y 300k iguales). Las causas de abajo explican
por qué no siguió subiendo; lo que estaba mal era la medición que decía "no cambió nada".

Por qué el currículo no se movió:

1. **El nivel 0 no es fácil.** El Elo mínimo de Stockfish es 1320 y por dentro equivale a Skill Level 0. El
   primer escalón del currículo era más o menos el mismo rival contra el que se mide el robot.
2. **A profundidad 1 el agente es ciego a las amenazas.** Mira su jugada y después solo capturas. El 73 % de
   las partidas contra Stockfish terminaron con el agente recibiendo mate (mediana: jugada 50); dio mate en el 5 %.
3. **El currículo medía al agente con exploración** (temperatura 0,05 también contra Stockfish).

Por qué no mejoró la fuerza:

| Causa | Evidencia |
|---|---|
| El maestro ve solo una jugada más que el alumno: TreeStrap a profundidad 1 ya había convergido (**causa principal**) | Contra versiones viejas de sí mismo saca 0,26 a 0,28 sin importar la edad del rival (22.000 partidas); en la primera corrida iba de 0,33 (recientes) a 0,42 (muy viejas). **Ojo (15/11):** que esté por debajo de 0,5 incluso contra versiones mucho más débiles se explica por la exploración: el agente juega con temperatura 0,05 y el viejo con 0. Medido con el mismo evaluador en los dos lados: 0,34 con temperatura 0,05 y 0,52 con 0. Lo que sí indica meseta es que no cambie con la edad del rival |
| El ancla externa es casi toda material | El 88 % del autojuego se corta a las 30 jugadas y se adjudica por material, que la red ya conocía |
| No sabe rematar | De las partidas contra Stockfish que llegan a 200 jugadas, en el 78 % iba ganando en material |
| Memoria corta | 400.000 muestras ≈ 1.170 partidas ≈ 4 minutos de juego; con tasa fija, altibajos entre checkpoints |
| Entrada incompleta | Sin enroque, al paso, repeticiones ni regla de 50 jugadas |
| Medición no comparable entre máquinas | Stockfish limitado por tiempo (0,05 s) |

### 4.3 Mejoras implementadas (rama `mejoras-meseta`, fusionada; `runs/mejoras` evaluada en 4.4: **funcionaron**)

Todas son opcionales: sin las opciones nuevas el código se comporta como antes.

| Causa | Cambio | Opción |
|---|---|---|
| Maestro poco profundo | Entrenar a profundidad 2 con poda alfa-beta | `--depth 2 --poda` |
| Ceguera al mate | Detección de mate en 1 en las hojas | `--mates` |
| Currículo sin escalones bajos | Stockfish que a veces juega al azar | `--sf-azar` |
| Currículo medido con exploración | Sin temperatura contra Stockfish | `--temp-stockfish 0` |
| Ancla casi toda material | Parte del autojuego se juega entero | `--p-completas 0.2` |
| Memoria corta | Buffer de 2 millones | `--buffer 2000000` |
| La exploración cuesta ≈ 0,17 de puntaje en todas las jugadas | Explorar solo al principio de cada partida o tramo | `--temp-jugadas 10` |
| Tasa de aprendizaje fija | Baja lineal | `--lr-final 1e-4` |
| Medición entre máquinas | Stockfish por nodos | `evaluar.py --sf-nodes N` |

**No implementado:** planos extra de entrada (enroque, al paso, repeticiones, regla de 50). Se pueden agregar
sin perder lo aprendido si los pesos de los canales nuevos arrancan en cero.

**Probado:** equivalencia de la poda con el árbol completo, detección de mates contra la versión lenta (52.670
posiciones) y corridas cortas de entrenamiento y evaluación con todas las opciones. **No probado:** con
Stockfish real ni con GPU (se usó un motor falso que juega al azar). La primera corrida real es la prueba.

**Expectativa honesta:** es probable que el currículo se mueva y que el agente reciba menos mates. Que eso suba
el Elo contra Stockfish 1320 es incierto: es la hipótesis que la corrida pone a prueba.

### 4.4 Cómo atribuir la mejora al RL (plan del 17/11)

Atribuir al RL = mostrar que el agente mejora **porque cambiaron los pesos aprendiendo de partidas**, con todo lo
demás fijo (misma búsqueda, mismo rival, mismas posiciones). Ya hay dos resultados así:
- `runs/titular` (100 partidas, d2q4): contra minimax2, `ck_000000` 0,695 → `ck_150000` 0,835 (≈ 2,4 errores
  estándar, p ≈ 0,02). Contra Stockfish 1320 por nodos: 0,23 → 0,295 (no significativo).
- `runs/desde_cero`: desde pesos al azar (sin saber ni el valor de las piezas) el RL llega de 0,05 a 0,85 contra
  el codicioso. Todo lo que juega lo aprendió por RL.

Medir contra un tercero con 100 partidas da IC de ± 0,1: no alcanza para diferencias chicas. Dos mediciones
más sensibles (`lanzar_mediciones.sh` las corre en orden, resultados en `runs/mediciones/`):
1. **Predicción del resultado** (`medir_valor.py`): el objetivo del TD es predecir quién gana. Posiciones de
   Lichess de **febrero de 2013** (el entrenamiento usó enero: así la red no "vio el examen"), con los cambios
   resueltos por quiescencia. Brier y LogLoss (bajo es mejor) y **AUC** (no depende de la escala del valor: si
   sube, la red ordena mejor las posiciones, no solo se recalibró). IC por bootstrap **por partida**. También
   la curva de calibración y el predictor de material puro (sin red) como referencia.
2. **Enfrentamiento directo** (`duelo.py`): A contra B con la misma búsqueda (d2 con poda, sin `--mates`).
   Las redes son deterministas: desde la posición inicial todas las partidas serían iguales, por eso cada par
   arranca de una posición real distinta (jugada 5 a 15, material parejo) y se juega con los dos colores. IC
   por par. 500 pares ≈ 25 min por enfrentamiento en la Mac con 6 procesos; detecta diferencias de ≈ 30 Elo.
   Control de cordura: una red contra sí misma da 0,500.

Si falta `runs/final/ck_000000.pt`, el lanzador regenera la red de material con `preentrenar_material.py`
(mismo procedimiento, no los mismos pesos) y la rotula `sin_rl_regen`. Para el informe conviene la original.

**Resultados (18/11, Mac, línea de base `ck_000000` original; gráficos `runs/mediciones/resumen_atribucion.png`
y `runs/mediciones/desde_cero/resumen_atribucion.png`):**

| Enfrentamiento directo (500 pares, 1.000 partidas, d2 con poda, sin `--mates`) | Puntaje de A | IC 95 % | Elo de A |
|---|---|---|---|
| RL 150k contra sin RL (`ck_000000`) | 0,552 | [0,537; 0,567] | **+36** [+26; +47] |
| 300k (continuación, prof. 1) contra 150k | 0,545 | [0,528; 0,562] | **+31** [+20; +43] |
| **mejoras 180k (prof. 2) contra 150k** | **0,658** | [0,641; 0,675] | **+114** [+100; +127] |
| **mejoras 180k contra 300k** | **0,634** | [0,617; 0,651] | **+95** [+83; +108] |
| (RL 150k contra sin RL regenerada, 17/11) | 0,568 | [0,552; 0,583] | +47 [+36; +58] |

Controles: los colores no influyen (por ejemplo mejoras contra 150k: 0,674 con blancas y 0,642 con negras);
ninguna partida llegó al tope de 300 jugadas; las tablas son todas por triple repetición (≈ 70 % entre redes
parecidas, 55 % en los de mejoras, que define más partidas: 448 terminan en mate, contra 266 en RL contra sin RL). Las dos redes usan la misma búsqueda,
así que la diferencia está en los pesos.

| Predicción del resultado (47.484 posiciones de 6.000 partidas de febrero de 2013) | AUC | AUC vs `ck_000000` (IC 95 %, pareado) | Brier vs `ck_000000` |
|---|---|---|---|
| Sin RL (`ck_000000`) | 0,767 | referencia | referencia |
| Material puro (sin red) | 0,775 | +0,008 [+0,005; +0,012] | −0,001, sign. |
| Sin RL regenerada | 0,772 | +0,005 [+0,002; +0,009] | igual |
| RL 150k | 0,782 | **+0,015** [+0,011; +0,020] | −0,002, sign. |
| `final_sf` 200k / 250k / 300k | 0,790 | **+0,023 a +0,024** [≈ +0,019; +0,029] | −0,004, sign. |
| mejoras 160k / 170k / **180k** | 0,793 / 0,795 / **0,799** | +0,026 / +0,028 / **+0,033** [+0,028; +0,038] | **−0,007**, sign. |

| RL desde cero (`runs/mediciones/desde_cero/`, mismas posiciones) | AUC |
|---|---|
| Pesos al azar (`desde_cero/ck_000000`) | 0,477 |
| TD-Leaf v1 a 5.000 / 10.000 / 20.000 partidas | 0,624 / 0,662 / **0,731** |
| Material puro (referencia) | 0,775 |
| TreeStrap puro a 20.000 (`treestrap_cero`) | 0,527 (colapsó: predice 50 % en toda posición) |

Lectura:
- **Desde cero**, el RL aprende a evaluar casi como el material (AUC 0,48 → 0,73) solo jugando contra sí mismo;
  TreeStrap sin el ancla del resultado colapsa.
- **Desde el material**, el RL suma +36 Elo y ordena mejor quién va ganando (AUC no depende de la escala del
  valor: no es solo recalibrar).
- **Continuar a profundidad 1** da un escalón más (+31 Elo, de 150k a 200k) y se estanca (200k = 250k = 300k).
- **Con las mejoras** (profundidad 2 con poda, mates, currículo con escalones fáciles, buffer grande, tasa que
  baja), 30.000 partidas valen +114 Elo, más que 150.000 a profundidad 1, y la predicción **sigue subiendo** de
  160k a 180k: la meseta se rompió. Lo que la causaba era sobre todo la profundidad del "maestro" (4.2).
- Medir contra un tercero con 100 partidas no veía estas diferencias; el enfrentamiento directo pareado sí.
  Los IC de cada AUC por separado se superponen, pero la comparación es pareada: cuenta el IC de la diferencia.

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
python chesscam/chesscam.py --ck runs/final/ck_150000.pt --sin-camara        # simulador con flechas
python chesscam/chesscam.py --ck runs/final/ck_150000.pt --bot blancas       # tablero real con webcam
python chesscam/chesscam.py --ck runs/mejoras/ck_180000.pt --reloj 5+3      # con reloj (5 min + 3 s)
python chesscam/camtest.py                                                   # si la cámara no abre
python jugar.py --ck runs/final/ck_150000.pt --color negras

# Continuar el bot final con Stockfish como rival (cada máquina en su propia carpeta)
python entrenar.py --out runs/final_sf --init runs/final/ck_150000.pt --resume --algo mixto \
    --start lichess --fens data/fens.txt --stockfish /opt/homebrew/bin/stockfish \
    --workers 6 --games 150000 --checkpoint-every 2000
#   (en una carpeta nueva los "viejos" aparecen recién cuando hay 2 checkpoints; se puede copiar
#    ck_150000.pt adentro para tenerlo como rival desde el principio)

# Plan del 15/11 a la noche (ver pendientes 2 y 3). Rutas de Windows; en la Mac, / y /opt/homebrew/bin/stockfish.
# a) Resultado principal del informe con 100 partidas: antes (ck_000000) y despues (ck_150000) del RL
mkdir runs\titular
copy runs\final\ck_000000.pt runs\titular\
copy runs\final\ck_150000.pt runs\titular\
python evaluar.py --run runs\titular --games 100 --depth 2 --q 4 --poda --rivals minimax2 --stockfish RUTA\stockfish.exe --sf-nodes 20000
# b) Corrida con las mejoras a profundidad 2 (la hipotesis principal)
mkdir runs\mejoras
copy runs\final\ck_150000.pt runs\mejoras\
python entrenar.py --out runs\mejoras --init runs\final\ck_150000.pt --resume --algo mixto --start lichess --fens data\fens.txt --stockfish RUTA\stockfish.exe --workers 7 --games 30000 --checkpoint-every 1000 --depth 2 --poda --mates --sf-azar --temp-stockfish 0 --temp-jugadas 10 --p-completas 0.2 --buffer 2000000 --lr-final 1e-4
#   --games 30000 son unas 2 o 3 noches en la laptop (≈ 1.000 a 1.500 partidas/h a profundidad 2). La tasa baja
#   de 5e-4 a 1e-4 a lo largo de esas 30.000. Se puede cortar con Ctrl+C y seguir con
#   --init runs\mejoras\ck_XXXXXX.pt --resume (mismos argumentos): conserva nivel de Stockfish y plan de tasa.
#   La GPU no acelera las partidas: la busqueda es Python en CPU. Mas --workers (uno por nucleo) da mas partidas/h.
#   Mirar al principio: partidas/h, que el "nivel stockfish" suba (arranca en 0; el Stockfish sin azar es el
#   nivel 6) y que |valor| no baje de 0,02.
# c) Control para separar el efecto de la profundidad (otra maquina, la misma noche): igual que b) pero en
#    runs\mejoras_d1, sin --depth 2 --poda y con --games 100000 (≈ 10 h a profundidad 1).

# Evaluar esa corrida (misma busqueda que en el entrenamiento, Stockfish por nodos)
python evaluar.py --run runs/mejoras --games 20 --depth 2 --q 4 --poda --mates --rivals minimax2 --stockfish RUTA --sf-nodes 20000 --every 4

# Mediciones para atribuir al RL (seccion 4.4); todo queda en runs/mediciones/
bash lanzar_mediciones.sh                                  # Mac; WORKERS=4 PARES=300 bash ... para ir mas rapido
python medir_valor.py --pgn data/lichess_db_standard_rated_2013-02.pgn.zst --ck sin_rl=runs/final/ck_000000.pt rl=runs/final/ck_150000.pt
python duelo.py --a runs/final/ck_150000.pt --b runs/final/ck_000000.pt --pares 500 --poda --out runs/mediciones/duelo_rl_vs_sinrl.csv

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

1. **Terminar `runs/mejoras` y medirla** (la predicción sigue subiendo en cada checkpoint). Plan acordado el 19/11:
   - **19/11 (Mac):** 185k, 190k y 195k ya están en `runs/mejoras/` de la Mac. Correr `bash lanzar_mediciones.sh`
     (≈ 2,5 h): predicción de 160k a 195k, y enfrentamientos de 195k contra **180k** (¿sigue aprendiendo?), contra
     150k y contra 300k. El lanzador elige solo el último checkpoint de `runs/mejoras/` y, desde el 19/11,
     también lo enfrenta contra `mej_180k` (para la predicción toma cada 10.000, y cada 5.000 desde 180.000).
     Subir el lanzador con los resultados.
   - **Noche del 19/11 (laptop con GPU):** la receta b de la sección 7 con `--init runs\mejoras\ck_195000.pt
     --resume` y `--games 15000`: llega a 210.000. Al reanudar conserva el nivel de Stockfish (iba en 8 a 9). La
     tasa sigue en 1e-4 constante: el plan de `--lr-final` (5e-4 a 1e-4 en 30.000 partidas, desde 150k) ya
     terminó, y para afinar está bien; para un plan nuevo, cambiar `--lr`.
   - **20/11 (Mac):** pasar a `runs/mejoras/` de la Mac al menos `ck_200000.pt`, `ck_205000.pt` y `ck_210000.pt`
     y relanzar `bash lanzar_mediciones.sh`: suma 210k a la predicción y lo enfrenta contra 180k, 150k y 300k
     (lo ya hecho se saltea).
   - **22/11: cierre de experimentos.** Si 210k sigue mejorando y queda una noche antes, se puede estirar; si
     no, elegir el mejor checkpoint medido para el robot (pendiente 2).
2. **Usar `runs/mejoras/ck_180000.pt` (o el último medido) en el robot**, `interfaz.py` y ChessCam, con
   `--depth 2` (es la búsqueda con la que entrenó).
3. **Elo absoluto del mejor checkpoint** (opcional, para la presentación): `evaluar.py` con `--depth 2 --q 4
   --poda --stockfish ... --sf-nodes 20000` y 100 a 200 partidas sobre `ck_180000` y `ck_150000` en una carpeta
   aparte. El ≈ 1240 actual es de 150k con Stockfish por tiempo.
4. Control de la búsqueda (opcional): `ck_180000` con y sin `--mates` en `duelo.py` contra el mismo rival, para
   separar lo que aporta la búsqueda de lo que aportan los pesos en el robot (en 4.4 todo se midió sin `--mates`).
5. Si `runs/mejoras` deja de mejorar: agregar los planos de entrada que faltan (enroque, al paso, repeticiones,
   regla de 50) con pesos nuevos en cero, o una red más grande.
6. Análisis "qué aprendió el RL": comparar la red de ck_0 y la final en posiciones de prueba (peón pasado, rey
   expuesto, actividad de piezas), y Grad-CAM con `last_features`.
7. Comparaciones para el informe: λ (0 / 0,7 / 1), TD-Leaf v2 vs mixto, arranque supervisado vs material.
8. Robot: `robot.py` con `elegir_jugada(tablero) -> (jugada, valor, sorpresa, emocion)`; visión
   `leer_tablero()` / `detectar_jugada()`; cara y voz `mostrar(emocion, texto)`. Cámara a 45 a 50 cm, ≈ 52°.
9. ~~ChessCam: el detector de movimiento compara el brillo absoluto entre cuadros~~ **Resuelto el 19/11** en la
   rama `vision-robusta` (sección 3.7): movimiento normalizado, promedio de cuadros, espera inicial,
   diferencia tolerante a corrimientos y `--fijar-camara`. El mismo día se sumó la detección manual por ráfaga
   (botón o espacio) y los filtros para piezas negras; con la webcam real (`--cam 1`) funciona. Rama fusionada
   en `main` (pull request #2). Falta: jugar 15 a 20 jugadas con `--capturas` (muchas de negras en casillas
   oscuras) y elegir el filtro con
   `diagnostico_negras.py`; terminar las filas pendientes de `prueba_robustez.py` y hacer un escenario de
   negras más difícil que (e).
10. Informe (marco teórico: MDP, valor, TD(λ), TD-Leaf, TreeStrap, búsqueda minimax, quiescencia y poda
   alfa-beta; caso de aplicación: el robot) y presentación del 26/11. Relato (4.4): (1) desde cero el RL aprende
   a evaluar casi como el material; (2) desde el material suma +36 Elo y predice mejor quién gana; (3) seguir
   igual da +31 y se estanca; (4) diagnosticamos la meseta (4.2), la corregimos (4.3) y la red da +114 Elo y
   sigue mejorando. Incluir la lección de medición: con 100 partidas contra un tercero no se veía nada, con
   enfrentamiento directo pareado sí. Gráficos: `runs/mediciones/resumen_atribucion.png`,
   `runs/mediciones/desde_cero/resumen_atribucion.png`, `runs/mediciones/valor_calibracion.png`.

## 9. Problemas conocidos y soluciones

| Síntoma | Causa / solución |
|---|---|
| `torch.cuda.is_available()` da False en Windows | Se instaló torch de PyPI (solo CPU). Reinstalar desde el índice de CUDA |
| `|valor|` del log cae a ≈ 0 | Colapso. Revisar que `--algo` sea `mixto` o `tdleaf` y `--adjudicar material` |
| "los procesos que juegan se cerraron" | Un worker tiró error (ruta de Stockfish o de `--fens` mal). Probar con `--workers 0` para ver el error |
| `PermissionError` al guardar `latest.pt` en Windows | `safe_save` reintenta; si persiste, antivirus o un proceso leyendo el archivo |
| El "nivel stockfish" no sube nunca | Sin `--sf-azar` el primer nivel ya es ≈ 1320 de Elo (sección 4.2). Usar `--sf-azar --temp-stockfish 0` |
| La corrida a profundidad 2 va muy lenta | Falta `--poda` (sin poda es ≈ 30 veces más lenta que profundidad 1; con poda, ≈ 8) |
| Push rechazado | `git pull --rebase` y `git push` |
| `No module named '_tkinter'` en la Mac (ChessCam) | El Python de Homebrew no trae tkinter: `brew install python-tk@3.14` (la misma versión que el `.venv`) |
| ChessCam: `No module named 'board_vision'` | Correrlo como script desde la raíz del repo: `python chesscam/chesscam.py`, no `python -m chesscam.chesscam` |
| ChessCam: aparecen cambios en casillas que nadie tocó, o tarda en dar la imagen por quieta | La webcam reajusta exposición, foco o balance de blancos. Usar `--fijar-camara` (mira en la terminal qué propiedades aceptó el driver: cada webcam acepta cosas distintas; en Mac no se puede) y, si la imagen queda oscura o clara, `--exposicion N`. Con MSMF el significado de la exposición automática varía: probar `--backend dshow` |
| La interfaz no abre ventana | `pip install pygame`; en Mac correrla desde la Terminal con el venv activado |
| `UCI_Elo` fuera de rango | Algunas versiones de Stockfish tienen otro mínimo; `StockfishPlayer` ya lo ajusta |

## 10. Referencias

- Sutton y Barto (2018). *Reinforcement Learning: An Introduction*, 2.ª ed.
- Sutton (1988). Learning to predict by the methods of temporal differences. *Machine Learning*.
- Tesauro (1995). Temporal difference learning and TD-Gammon. *CACM*.
- Baxter, Tridgell y Weaver (2000). Learning to play chess using temporal differences (KnightCap, TD-Leaf). *Machine Learning*.
- Veness, Silver, Uther y Blair (2009). Bootstrapping from game tree search (TreeStrap, Meep). *NeurIPS*.
- Lai (2015). Giraffe: using deep reinforcement learning to play chess. Imperial College London.
