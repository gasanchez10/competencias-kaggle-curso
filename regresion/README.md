# Nueva competencia: regresión de precios de departamentos (Daegu)

Clon de `mae-competencia-regresion-202613` con un **conjunto de evaluación nuevo**.
El train es el original sin cambios. El test original salía de una partición
aleatoria del dataset público de Daegu (5891 filas), así que sus etiquetas se
pueden encontrar en internet.

## Archivos (`nueva_competencia/`)

| Archivo | Se publica | Contenido |
|---|---|---|
| `Train real state.csv` | Sí | 4124 filas originales, con `SalePrice` |
| `Test real state.csv` | Sí | 1767 filas generadas, sin `SalePrice` |
| `sample_submission.csv` | Sí | `id,SalePrice` (mediana del train) |
| `solution.csv` | **No (solo para el host)** | `id,SalePrice,Usage` (Public 31 % / Private 69 %) |

Cambio respecto al original: la primera columna, que antes no tenía nombre, ahora se llama `id`.
Los ids del test van de 100001 a 101767, sin relación con el dataset público.

## Configurar en Kaggle (Host → Create Competition)

1. **Data:** subir `Train real state.csv`, `Test real state.csv` y `sample_submission.csv`.
2. **Evaluation → Solution file:** subir `solution.csv`.
   - Métrica: **Mean Absolute Error (MAE)**.
   - Columna de id: `id`. Columna objetivo: `SalePrice`.
   - La columna `Usage` define qué filas cuentan en el leaderboard público y cuáles en el privado.
3. Copiar el texto de Overview, Evaluation y Rules de la competencia original.

## Regenerar o verificar

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # todo desde la raíz del repo
.venv/bin/python regresion/generar_test.py            # --seed, --n, --public, --noise-scale
.venv/bin/python regresion/validar_test.py            # distribuciones, validación adversarial, MAE
```

Con otra `--seed` se obtiene otro test distinto pero equivalente; sirve para hacer una competencia nueva cada semestre.

## Cómo se genera

1. Se toma un complejo (son 26) con probabilidad proporcional a su frecuencia en el train, con todos sus atributos de edificio y entorno.
2. Se elige un tamaño de planta real de ese complejo. El piso lleva un jitter de ±2 y la fecha de venta uno de ±3 meses, siempre con año de venta ≥ año de construcción.
3. El precio se calcula como `exp(f(x) + 0.55·e)`:
   - `f` es el promedio de LightGBM, Random Forest y Ridge con splines, ajustados sobre log(precio).
   - `e` es un bootstrap de los residuos out-of-fold dentro del mismo complejo.

## Validación (seed 2026)

| Modelo entrenado en train | MAE en CV barajada | MAE en test nuevo |
|---|---|---|
| Mediana | 86,157 | 85,085 |
| Regresión lineal | 27,075 | 22,817 |
| kNN | 19,510 | 20,263 |
| Random forest | 10,312 | 10,166 |
| LightGBM | 9,805 | 10,045 |

- Las distribuciones de precio, tamaño, piso, fecha y estaciones son prácticamente iguales.
- AUC adversarial: 0.56 sin precio y 0.64 con precio. Hay un desplazamiento leve, propio de los datos sintéticos.
- El ranking de modelos se mantiene. La regresión lineal sale algo favorecida porque el generador incluye un componente suave.
- Nota: el train está ordenado por fecha (correlación 0.998 entre id y fecha), así que hay que barajar la CV.

## Modelos de referencia (`modelos/`)

Cada carpeta tunea un modelo con CV barajada de 5 folds en el train (métrica MAE), predice el test nuevo
y lo puntúa contra `solution.csv` como lo haría Kaggle. Guarda `submission.csv` y `resultado.json`.

| Modelo | MAE en CV (train) | MAE público | MAE privado | MAE total |
|---|---|---|---|---|
| Random Forest | 9,830 | 10,019 | 9,417 | 9,604 |
| Red neuronal (MLP) | 10,977 | 10,845 | 10,626 | 10,694 |
| Regresión lineal (Ridge + splines, log) | 19,723 | 14,183 | 14,180 | 14,181 |

```bash
.venv/bin/python regresion/modelos/regresion_lineal/modelo.py   # ~10 s
.venv/bin/python regresion/modelos/red_neuronal/modelo.py       # ~70 s
.venv/bin/python regresion/modelos/random_forest/modelo.py      # ~2 min
```

- Random Forest y la red quedan dentro de un ±3 % de su CV.
- La regresión lineal sale un 28 % mejor en el test que en CV, porque el generador incluye un componente
  Ridge con splines. El ranking no cambia, pero el modelo lineal queda favorecido. **Punto a revisar.**

## Estructura

```
original/               datos originales de Kaggle (train + test sin etiquetas)
nueva_competencia/      archivos de la competencia nueva (solution.csv es solo para el host)
generar_test.py         generador del test
validar_test.py         chequeos del test generado
modelos/                modelos de referencia
```
