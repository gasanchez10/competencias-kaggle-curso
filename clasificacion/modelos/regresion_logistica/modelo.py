"""Regresión logística regularizada con tuning por CV.

Opciones que se tunean:
  - numéricas lineales vs splines (no linealidad en edad, duración, saldo...)
  - penalización L1 / L2 y su fuerza (C)
  - balanceo de clases (no cambia mucho el AUC, pero se prueba)
"""
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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
model = Pipeline([("pre", pre), ("lr", LogisticRegression(solver="liblinear", max_iter=5000))])

spline = Pipeline([("s", StandardScaler()), ("sp", SplineTransformer(n_knots=6, degree=3))])
grid = {
    "pre__num": [StandardScaler(), spline],
    "lr__penalty": ["l1", "l2"],
    "lr__C": [0.01, 0.1, 1, 10],
    "lr__class_weight": [None, "balanced"],
}
search = GridSearchCV(model, grid, cv=CV, scoring=SCORING, n_jobs=-1)
search.fit(X, y)
evaluar("Regresión logística", Path(__file__).parent, search, X_te, ids)
