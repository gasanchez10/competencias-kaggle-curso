"""
Chequeos del test generado (clasificación, métrica AUC):
  1. Distribuciones train vs test nuevo.
  2. Validación adversarial: AUC de un clasificador que distingue train vs test nuevo.
  3. Benchmark: AUC en CV estratificada barajada sobre train vs AUC en el test nuevo.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "nueva_competencia"
SKF = StratifiedKFold(5, shuffle=True, random_state=0)  # barajado: el train está ordenado por fecha
T = "Subscription"
tr = pd.read_csv(D / "Train bank.csv")
te = pd.read_csv(D / "Test bank.csv").merge(pd.read_csv(D / "solution.csv"), on="id")
cat = tr.select_dtypes("object").columns.tolist()
num = [c for c in tr.columns if c not in cat + ["id", T]]

print("== 1. Distribuciones ==")
stats = ["mean", "50%", "std"]
print(pd.DataFrame({"train": tr[num + [T]].describe().loc[stats].T.stack(),
                    "test": te[num + [T]].describe().loc[stats].T.stack()}).round(2).to_string())
for c in ["Job", "Poutcome", "Last.Contact.Month"]:
    print(pd.concat([tr[c].value_counts(normalize=True), te[c].value_counts(normalize=True)],
                    axis=1, keys=["train", "test"]).round(3).T.to_string())

enc = lambda d: pd.get_dummies(d[num + cat], columns=cat, dtype=float)
allX = enc(pd.concat([tr, te], ignore_index=True))
lab = np.r_[np.zeros(len(tr)), np.ones(len(te))]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), allX, lab, cv=SKF,
                      method="predict_proba")[:, 1]
print(f"\n== 2. AUC adversarial (sin etiqueta): {roc_auc_score(lab, p):.3f}")

Xtr, Xte = enc(tr), enc(te).reindex(columns=enc(tr).columns, fill_value=0)
models = {
    "logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "kNN (k=50)": make_pipeline(StandardScaler(), KNeighborsClassifier(50)),
    "random forest": RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=1),
    "lightgbm": lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, verbose=-1, random_state=1),
}
print("\n== 3. AUC: CV en train vs test nuevo ==")
for name, m in models.items():
    cv = roc_auc_score(tr[T], cross_val_predict(m, Xtr, tr[T], cv=SKF, method="predict_proba")[:, 1])
    ho = roc_auc_score(te[T], m.fit(Xtr, tr[T]).predict_proba(Xte)[:, 1])
    print(f"  {name:15s} CV={cv:.4f}   test nuevo={ho:.4f}   dif={ho - cv:+.4f}")
