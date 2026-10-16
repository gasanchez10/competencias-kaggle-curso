"""Utilidades compartidas: carga de datos, features, CV y puntuación tipo Kaggle (AUC)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent / "nueva_competencia"
TARGET = "Subscription"
CAT_COLS = ["Job", "Marital.Status", "Education", "Credit", "Housing.Loan",
            "Personal.Loan", "Contact", "Last.Contact.Month", "Poutcome"]
# El train está ordenado por fecha: la CV debe ser barajada.
CV = StratifiedKFold(5, shuffle=True, random_state=42)
SCORING = "roc_auc"


def add_features(df):
    X = df.drop(columns=[c for c in ("id", TARGET) if c in df]).copy()
    X["log_duration"] = np.log1p(X["Last.Contact.Duration"])
    X["log_balance"] = np.sign(X["Balance..euros."]) * np.log1p(np.abs(X["Balance..euros."]))
    X["log_campaign"] = np.log(X["Campaign"])
    X["contactado_antes"] = (X["Pdays"] >= 0).astype(int)
    return X


def load():
    tr = pd.read_csv(DATA / "Train bank.csv")
    te = pd.read_csv(DATA / "Test bank.csv")
    return add_features(tr), tr[TARGET], add_features(te), te["id"]


def num_cols(X):
    return [c for c in X.columns if c not in CAT_COLS]


def evaluar(nombre, carpeta, search, X_te, ids):
    """Imprime/guarda CV y score público/privado (AUC) del mejor modelo del search."""
    carpeta = Path(carpeta)
    pred = search.best_estimator_.predict_proba(X_te)[:, 1]
    sub = pd.DataFrame({"id": ids, TARGET: pred})
    sub.to_csv(carpeta / "submission.csv", index=False)

    sol = pd.read_csv(DATA / "solution.csv").merge(sub, on="id", suffixes=("", "_pred"))
    auc = lambda m: roc_auc_score(sol.loc[m, TARGET], sol.loc[m, TARGET + "_pred"])
    res = {
        "modelo": nombre,
        "cv_auc": round(search.best_score_, 4),
        "public_auc": round(auc(sol.Usage == "Public"), 4),
        "private_auc": round(auc(sol.Usage == "Private"), 4),
        "total_auc": round(auc(slice(None)), 4),
        "mejores_parametros": {k: str(v) for k, v in search.best_params_.items()},
    }
    (carpeta / "resultado.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res
