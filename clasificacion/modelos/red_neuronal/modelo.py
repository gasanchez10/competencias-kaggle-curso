"""Red neuronal (MLP de scikit-learn) con búsqueda aleatoria de hiperparámetros.

Entradas estandarizadas + one-hot. Early stopping para evitar sobreajuste.
"""
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comun import CAT_COLS, CV, SCORING, evaluar, load, num_cols  # noqa: E402

X, y, X_te, ids = load()

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols(X)),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
])
mlp = MLPClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=20, random_state=0)
model = Pipeline([("pre", pre), ("mlp", mlp)])

dist = {
    "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "mlp__learning_rate_init": [3e-4, 1e-3, 3e-3],
    "mlp__batch_size": [64, 128, 256],
}
search = RandomizedSearchCV(model, dist, n_iter=20, cv=CV, scoring=SCORING,
                            random_state=0, n_jobs=-1)
search.fit(X, y)
evaluar("Red neuronal (MLP)", Path(__file__).parent, search, X_te, ids)
