"""Random Forest con búsqueda aleatoria de hiperparámetros por CV."""
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comun import CAT_COLS, CV, SCORING, evaluar, load  # noqa: E402

X, y, X_te, ids = load()

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
                        remainder="passthrough")
model = Pipeline([("pre", pre), ("rf", RandomForestClassifier(random_state=0, n_jobs=-1))])

dist = {
    "rf__n_estimators": [300, 500],
    "rf__max_features": [0.2, 0.3, 0.5, "sqrt"],
    "rf__min_samples_leaf": [1, 3, 5, 10, 20],
    "rf__max_depth": [None, 10, 20],
    "rf__class_weight": [None, "balanced_subsample"],
}
search = RandomizedSearchCV(model, dist, n_iter=20, cv=CV, scoring=SCORING,
                            random_state=0, n_jobs=1)
search.fit(X, y)
evaluar("Random Forest", Path(__file__).parent, search, X_te, ids)
