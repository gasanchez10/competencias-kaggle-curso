# Nueva competencia: clasificación Bank Marketing (suscripción a depósito)

Clon de `mae-competencia-clasificacion-202613` con un **conjunto de evaluación nuevo**.
El train es el original sin cambios. El test original salía de una partición aleatoria del dataset
público *Bank Marketing* de UCI (`bank-full`, 45211 filas), así que sus etiquetas se pueden encontrar
en internet.

## Archivos (`nueva_competencia/`)

| Archivo | Se publica | Contenido |
|---|---|---|
| `Train bank.csv` | Sí | 31648 filas originales, con `Subscription` (0/1, 11.7 % positivos) |
| `Test bank.csv` | Sí | 13563 filas generadas, sin `Subscription` |
| `sample_submission.csv` | Sí | `id,Subscription` (probabilidad constante 0.1168) |
| `solution.csv` | **No (solo para el host)** | `id,Subscription,Usage` (Public 31 % / Private 69 %) |

Cambios respecto al original:
- La primera columna, que antes no tenía nombre, ahora se llama `id`.
- Los ids del test van de 100001 a 113563.

**Ojo:** en el dataset original el id está ordenado por fecha y la tasa de positivos sube de 3 % a 47 %
a lo largo del id. En la competencia original, usar el id como variable era una fuga de información.
Con los ids nuevos esa fuga ya no existe, y un modelo que use el id fallará en el test.

## Configurar en Kaggle

1. **Data:** subir `Train bank.csv`, `Test bank.csv` y `sample_submission.csv`.
2. **Evaluation → Solution file:** subir `solution.csv`.
   - Métrica: **AUC (Area Under ROC Curve)**.
   - Columna de id: `id`. Columna objetivo: `Subscription`.
   - Los envíos deben ser probabilidades.

## Regenerar o verificar (desde la raíz del repo)

```bash
.venv/bin/python clasificacion/generar_test.py     # --seed, --n, --public, --temp
.venv/bin/python clasificacion/validar_test.py     # distribuciones, validación adversarial, AUC
```

## Cómo se genera

1. **Variables (estilo SMOTE-NC).** Se toma una fila base del train y uno de sus 5 vecinos más cercanos
   del mismo mes de contacto.
   - Edad, saldo, duración y campaña se interpolan entre las dos filas.
   - Mes y día vienen de la fila base.
   - El bloque de campaña previa (`Pdays`, `Previous`, `Poutcome`) se copia completo de una de las dos
     filas, para que siga siendo coherente.
   - Las demás categóricas se toman al azar de una o de la otra.
2. **Etiqueta.** Se sortea `y ~ Bernoulli(p)`, con `logit(p) = (logit f(x, tiempo) + b0) / T`.
   - `f` es el promedio de logística con splines, Random Forest y LightGBM. Se ajusta con las variables
     más el **tiempo latente** (el id original de la fila base), que no se publica.
   - `T = 0.85` está calibrado para que el AUC del test se parezca al de CV.
   - `b0` mantiene la prevalencia del train.

## Validación (seed 2026)

- **Distribuciones:** prácticamente iguales a las del train.
- **AUC adversarial:** **0.514**, es decir, el test casi no se distingue del train.
- **Benchmarks sin tunear:**

| Modelo | AUC en CV | AUC en test nuevo |
|---|---|---|
| Logística | 0.9028 | 0.9048 |
| kNN (k=50) | 0.8604 | 0.8758 |
| Random forest | 0.9292 | 0.9279 |
| LightGBM | 0.9342 | 0.9282 |

## Modelos de referencia tuneados (`modelos/`)

Cada carpeta tunea un modelo con CV estratificada barajada de 5 folds (AUC), predice el test nuevo y lo
puntúa contra `solution.csv`. Guarda `submission.csv` y `resultado.json`.

| Modelo | AUC en CV (train) | AUC público | AUC privado | AUC total |
|---|---|---|---|---|
| Random Forest | 0.9278 | 0.9302 | 0.9235 | 0.9255 |
| Red neuronal (MLP) | 0.9258 | 0.9278 | 0.9220 | 0.9238 |
| Regresión logística (L1 + splines) | 0.9143 | 0.9211 | 0.9100 | 0.9135 |

```bash
.venv/bin/python clasificacion/modelos/regresion_logistica/modelo.py   # ~2.5 min
.venv/bin/python clasificacion/modelos/red_neuronal/modelo.py          # ~40 s
.venv/bin/python clasificacion/modelos/random_forest/modelo.py         # ~2.5 min
```

- Los tres quedan dentro de ±0.002 de su AUC de CV, y el ranking se mantiene.
- La diferencia entre público y privado es de ~0.01, algo esperable con unos 4200 ejemplos públicos.
