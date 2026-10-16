"""Random Forest con búsqueda aleatoria de hiperparámetros por CV."""
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comun import CAT_COLS, CV, SCORING, evaluar, load  # noqa: E402

X, y, X_te, ids = load()

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
                        remainder="passthrough")
pipe = Pipeline([("pre", pre), ("rf", RandomForestRegressor(random_state=0, n_jobs=-1))])
model = TransformedTargetRegressor(regressor=pipe)

dist = {
    "func": [None], "inverse_func": [None],
    "regressor__rf__n_estimators": [300, 500, 800],
    "regressor__rf__max_features": [0.3, 0.5, 0.7, 1.0, "sqrt"],
    "regressor__rf__min_samples_leaf": [1, 2, 3, 5, 8],
    "regressor__rf__max_depth": [None, 12, 20, 30],
}
dists = [dist, dict(dist, func=[np.log], inverse_func=[np.exp])]  # precio vs log(precio)

search = RandomizedSearchCV(model, dists, n_iter=40, cv=CV, scoring=SCORING,
                            random_state=0, n_jobs=1)
search.fit(X, y)
evaluar("Random Forest", Path(__file__).parent, search, X_te, ids)
