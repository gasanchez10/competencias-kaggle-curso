"""Red neuronal (MLP de scikit-learn) con búsqueda aleatoria de hiperparámetros.

Entradas estandarizadas + one-hot; objetivo en log(precio) estandarizado.
Early stopping para evitar sobreajuste.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comun import CAT_COLS, CV, SCORING, evaluar, load, num_cols  # noqa: E402

X, y, X_te, ids = load()

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols(X)),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
])
mlp = MLPRegressor(max_iter=2000, early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=30, random_state=0)
pipe = Pipeline([("pre", pre), ("mlp", mlp)])

# log(precio) y luego estandarizado: escala cómoda para la red
log_std = Pipeline([("log", FunctionTransformer(np.log, np.exp)), ("std", StandardScaler())])
model = TransformedTargetRegressor(regressor=pipe, transformer=log_std)

dist = {
    "regressor__mlp__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64),
                                           (256, 128), (128, 64, 32), (256, 128, 64)],
    "regressor__mlp__activation": ["relu", "tanh"],
    "regressor__mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "regressor__mlp__learning_rate_init": [3e-4, 1e-3, 3e-3],
    "regressor__mlp__batch_size": [32, 64, 128],
}
search = RandomizedSearchCV(model, dist, n_iter=40, cv=CV, scoring=SCORING,
                            random_state=0, n_jobs=-1)
search.fit(X, y)
evaluar("Red neuronal (MLP)", Path(__file__).parent, search, X_te, ids)
