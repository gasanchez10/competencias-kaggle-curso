"""
Chequeos del test generado:
  1. Distribuciones train vs test nuevo (numéricas y categóricas).
  2. Validación adversarial: AUC de un clasificador que intenta distinguir
     train vs test nuevo (ideal ~0.5).
  3. Benchmark: MAE en CV 5-fold sobre train vs MAE sobre el test nuevo, para
     varias familias de modelos. Deben ser parecidos y con el mismo ranking.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "nueva_competencia"
SKF = StratifiedKFold(5, shuffle=True, random_state=0)
KF = KFold(5, shuffle=True, random_state=0)  # barajado: el train está ordenado por fecha
tr = pd.read_csv(D / "Train real state.csv")
te = pd.read_csv(D / "Test real state.csv").merge(pd.read_csv(D / "solution.csv"), on="id")
T = "SalePrice"
cat = tr.select_dtypes("object").columns.tolist()
num = [c for c in tr.columns if c not in cat + ["id", T]]

print("== 1. Distribuciones ==")
print(pd.DataFrame({"train": tr[[T] + ["YrSold", "Size.sqf.", "Floor"]].describe().loc[["mean", "50%", "std", "min", "max"]].stack(),
                    "test":  te[[T] + ["YrSold", "Size.sqf.", "Floor"]].describe().loc[["mean", "50%", "std", "min", "max"]].stack()}).round(0))
for c in ["SubwayStation", "HallwayType"]:
    print(pd.concat([tr[c].value_counts(normalize=True), te[c].value_counts(normalize=True)],
                    axis=1, keys=["train", "test"]).round(3))

enc = lambda d: pd.get_dummies(d[num + cat], columns=cat, dtype=float)
allX = enc(pd.concat([tr, te], ignore_index=True))
lab = np.r_[np.zeros(len(tr)), np.ones(len(te))]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), allX, lab, cv=SKF,
                      method="predict_proba")[:, 1]
print(f"\n== 2. AUC adversarial (sin precio): {roc_auc_score(lab, p):.3f}")
allX[T] = np.r_[tr[T], te[T]]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), allX, lab, cv=SKF,
                      method="predict_proba")[:, 1]
print(f"   AUC adversarial (con precio): {roc_auc_score(lab, p):.3f}")

Xtr, Xte = enc(tr), enc(te).reindex(columns=enc(tr).columns, fill_value=0)
ohe = lambda: ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
models = {
    "mediana (baseline)": None,
    "regresión lineal": LinearRegression(),
    "kNN (k=10)": make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
    "random forest": RandomForestRegressor(300, n_jobs=-1, random_state=1),
    "lightgbm": lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, verbose=-1, random_state=1),
}
print("\n== 3. MAE: CV en train vs test nuevo ==")
for name, m in models.items():
    if m is None:
        cv = mean_absolute_error(tr[T], np.full(len(tr), tr[T].median()))
        ho = mean_absolute_error(te[T], np.full(len(te), tr[T].median()))
    else:
        cv = mean_absolute_error(tr[T], cross_val_predict(m, Xtr, tr[T], cv=KF))
        ho = mean_absolute_error(te[T], m.fit(Xtr, tr[T]).predict(Xte))
    print(f"  {name:20s} CV={cv:9,.0f}   test nuevo={ho:9,.0f}   ratio={ho / cv:.2f}")
