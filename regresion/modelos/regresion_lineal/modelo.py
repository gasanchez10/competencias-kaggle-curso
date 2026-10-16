"""Regresión lineal regularizada (Ridge/Lasso/ElasticNet) con tuning por CV.

Opciones que se tunean:
  - transformación del objetivo: precio vs log(precio)
  - numéricas lineales vs splines (permite no linealidad en tamaño, piso, tiempo...)
  - penalización (alpha) y tipo (Ridge / ElasticNet)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comun import CAT_COLS, CV, SCORING, evaluar, load, num_cols  # noqa: E402

X, y, X_te, ids = load()

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols(X)),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
])
pipe = Pipeline([("pre", pre), ("reg", Ridge())])
model = TransformedTargetRegressor(regressor=pipe)

spline = Pipeline([("s", StandardScaler()), ("sp", SplineTransformer(n_knots=5, degree=3))])
grid = []
for f, g in [(None, None), (np.log, np.exp)]:  # func e inverse_func van emparejados
    grid.append({
        "func": [f], "inverse_func": [g],
        "regressor__pre__num": [StandardScaler(), spline],
        "regressor__reg": [Ridge()],
        "regressor__reg__alpha": [0.01, 0.1, 1, 10, 100],
    })
grid.append({
    "func": [np.log], "inverse_func": [np.exp],
    "regressor__pre__num": [StandardScaler(), spline],
    "regressor__reg": [ElasticNet(max_iter=20000)],
    "regressor__reg__alpha": [1e-4, 1e-3, 1e-2],
    "regressor__reg__l1_ratio": [0.2, 0.5, 0.8],
})

search = GridSearchCV(model, grid, cv=CV, scoring=SCORING, n_jobs=-1)
search.fit(X, y)
evaluar("Regresión lineal", Path(__file__).parent, search, X_te, ids)
