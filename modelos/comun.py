"""Utilidades compartidas: carga de datos, features, CV y puntuación tipo Kaggle."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

DATA = Path(__file__).resolve().parent.parent / "nueva_competencia"
TARGET = "SalePrice"
CAT_COLS = ["HallwayType", "HeatingType", "AptManageType", "TimeToBusStop",
            "TimeToSubway", "SubwayStation"]
# El train está ordenado por fecha: la CV debe ser barajada.
CV = KFold(5, shuffle=True, random_state=42)
SCORING = "neg_mean_absolute_error"


def add_features(df):
    X = df.drop(columns=[c for c in ("id", TARGET) if c in df]).copy()
    X["t"] = (X["YrSold"] - 2007) * 12 + X["MonthSold"] - 1  # mes continuo
    X["age"] = X["YrSold"] - X["YearBuilt"]
    X["log_size"] = np.log(X["Size.sqf."])
    return X


def load():
    tr = pd.read_csv(DATA / "Train real state.csv")
    te = pd.read_csv(DATA / "Test real state.csv")
    return add_features(tr), tr[TARGET], add_features(te), te["id"]


def num_cols(X):
    return [c for c in X.columns if c not in CAT_COLS]


def evaluar(nombre, carpeta, search, X_te, ids):
    """Imprime/guarda CV y score público/privado del mejor modelo del search."""
    carpeta = Path(carpeta)
    pred = search.best_estimator_.predict(X_te)
    sub = pd.DataFrame({"id": ids, TARGET: pred})
    sub.to_csv(carpeta / "submission.csv", index=False)

    sol = pd.read_csv(DATA / "solution.csv").merge(sub, on="id", suffixes=("", "_pred"))
    mae = lambda m: mean_absolute_error(sol.loc[m, TARGET], sol.loc[m, TARGET + "_pred"])
    res = {
        "modelo": nombre,
        "cv_mae": round(-search.best_score_, 1),
        "public_mae": round(mae(sol.Usage == "Public"), 1),
        "private_mae": round(mae(sol.Usage == "Private"), 1),
        "total_mae": round(mae(slice(None)), 1),
        "mejores_parametros": {k: str(v) for k, v in search.best_params_.items()},
    }
    (carpeta / "resultado.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res
