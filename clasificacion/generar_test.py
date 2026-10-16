"""
Genera un nuevo conjunto de evaluación (test + solución) para la competencia de
clasificación Bank Marketing (Subscription 0/1, métrica AUC), a partir del train original.

Diseño del generador
--------------------
1. Variables (estilo SMOTE-NC): para cada fila nueva se toma una fila base b del
   train y uno de sus k=5 vecinos más cercanos n *del mismo mes de contacto*.
   - Age, Balance, Duration, Campaign: interpolación b + u*(n-b), u ~ U(0,1), redondeada.
   - Mes, día y tiempo latente: de b.
   - Bloque de campaña previa (Pdays, Previous, Poutcome): completo de b o de n, para
     mantener la coherencia (p. ej. Pdays=-1 <=> Previous=0 <=> Poutcome=unknown).
   - Resto de categóricas: cada una de b o de n con prob. 1/2.
2. Etiqueta: y ~ Bernoulli(p), con
   logit(p) = (logit(f(x, tiempo)) + b0) / T
   - f es un ensamble (logística + Random Forest + LightGBM) de probabilidades
     ajustado en el train usando también el *tiempo latente* (id original, ordenado
     cronológicamente), que no se publica. Así la tasa cambia en el tiempo como en
     los datos reales, sin que el id nuevo lo revele.
   - T (temperatura) controla la dificultad. Se calibra con validar_test.py para que
     el AUC del test se parezca al AUC de CV.
   - b0 se ajusta para conservar la prevalencia del train.

Uso:  .venv/bin/python clasificacion/generar_test.py [--n 13563] [--seed 2026] [--public 0.3] [--temp T]
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ROOT = Path(__file__).parent
TRAIN = ROOT / "original" / "Train bank.csv"
OUT = ROOT / "nueva_competencia"
TARGET = "Subscription"

INTERP = ["Age", "Balance..euros.", "Last.Contact.Duration", "Campaign"]
FROM_BASE = ["Last.Contact.Month", "Last.Contact.Day", "time"]
PREV_BLOCK = ["Pdays", "Previous", "Poutcome"]
SWAP_CAT = ["Job", "Marital.Status", "Education", "Credit", "Housing.Loan",
            "Personal.Loan", "Contact"]
CAT_COLS = SWAP_CAT + ["Last.Contact.Month", "Poutcome"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def load_train():
    df = pd.read_csv(TRAIN).rename(columns={"Unnamed: 0": "id"})
    df["time"] = df["id"] / df["id"].max()  # tiempo latente en [0, 1]
    return df


def features(df):
    X = df.drop(columns=[c for c in ("id", TARGET) if c in df]).copy()
    X["log_duration"] = np.log1p(X["Last.Contact.Duration"])
    X["month_num"] = X["Last.Contact.Month"].map({m: i for i, m in enumerate(MONTHS)})
    return X


def build_models(X):
    num = [c for c in X.columns if c not in CAT_COLS]
    ohe = lambda: OneHotEncoder(handle_unknown="ignore")
    logistic = make_pipeline(
        ColumnTransformer([
            ("spl", make_pipeline(StandardScaler(), SplineTransformer(n_knots=6)), num),
            ("cat", ohe(), CAT_COLS),
        ]),
        LogisticRegression(C=1.0, max_iter=3000),
    )
    rf = make_pipeline(
        ColumnTransformer([("cat", ohe(), CAT_COLS)], remainder="passthrough"),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=0),
    )
    gbm = make_pipeline(
        ColumnTransformer([("cat", ohe(), CAT_COLS)], remainder="passthrough"),
        lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=0, verbose=-1),
    )
    return [logistic, rf, gbm]


def sample_rows(df, n, rng, k=5):
    num_space = ["Age", "Balance..euros.", "Last.Contact.Duration", "Campaign",
                 "Pdays", "Previous", "Last.Contact.Day"]
    Z = df[num_space].astype(float)
    Z["Balance..euros."] = np.sign(Z["Balance..euros."]) * np.log1p(np.abs(Z["Balance..euros."]))
    Z["Last.Contact.Duration"] = np.log1p(Z["Last.Contact.Duration"])
    Z = (Z - Z.mean()) / Z.std()
    Z = pd.concat([Z, pd.get_dummies(df[SWAP_CAT], dtype=float) * 0.5], axis=1).values

    # vecinos por mes (mantiene la coherencia temporal)
    neigh = {}
    for m, idx in df.groupby("Last.Contact.Month").indices.items():
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(idx))).fit(Z[idx])
        neigh[m] = (idx, nn.kneighbors(Z[idx], return_distance=False)[:, 1:])
    pos_in_month = {i: (m, j) for m, (idx, _) in neigh.items() for j, i in enumerate(idx)}

    base_idx = rng.integers(len(df), size=n)
    rows = []
    for bi in base_idx:
        m, j = pos_in_month[bi]
        idx, nbrs = neigh[m]
        ni = idx[rng.choice(nbrs[j])]
        b, nb = df.iloc[bi], df.iloc[ni]
        r = {c: b[c] for c in FROM_BASE}
        u = rng.random()
        for c in INTERP:
            r[c] = int(round(b[c] + u * (nb[c] - b[c])))
        src = b if rng.random() < 0.5 else nb
        r.update({c: src[c] for c in PREV_BLOCK})
        for c in SWAP_CAT:
            r[c] = (b if rng.random() < 0.5 else nb)[c]
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=13563)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--public", type=float, default=0.3, help="fracción del leaderboard público")
    ap.add_argument("--temp", type=float, default=0.85,
                    help="temperatura del logit (>1 más difícil); calibrar con validar_test.py")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = load_train()
    X, y = features(df), df[TARGET]

    print("Muestreando filas nuevas...")
    new = sample_rows(df, args.n, rng)
    Xnew = features(new)[X.columns]

    print("Ajustando el modelo generador de etiquetas...")
    p = np.mean([m.fit(X, y).predict_proba(Xnew)[:, 1] for m in build_models(X)], axis=0)
    z = logit(np.clip(p, 1e-4, 1 - 1e-4))
    prev = y.mean()
    b0 = brentq(lambda b: expit((z + b) / args.temp).mean() - prev, -10, 10)
    p_final = expit((z + b0) / args.temp)
    new[TARGET] = (rng.random(args.n) < p_final).astype(int)
    print(f"  prevalencia train={prev:.4f}  test={new[TARGET].mean():.4f}  (T={args.temp}, b0={b0:.2f})")

    new["id"] = rng.permutation(np.arange(100001, 100001 + args.n))
    new = new.sort_values("id").reset_index(drop=True)
    cols = [c for c in pd.read_csv(TRAIN, nrows=0).columns if c not in ("Unnamed: 0", TARGET)]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    new[["id"] + cols].to_csv(out / "Test bank.csv", index=False)
    usage = np.where(rng.random(args.n) < args.public, "Public", "Private")
    pd.DataFrame({"id": new["id"], TARGET: new[TARGET], "Usage": usage}).to_csv(
        out / "solution.csv", index=False)
    pd.DataFrame({"id": new["id"], TARGET: round(prev, 4)}).to_csv(
        out / "sample_submission.csv", index=False)
    df[["id"] + cols + [TARGET]].to_csv(out / "Train bank.csv", index=False)

    print(f"Listo: {args.n} filas en {out} (Public={np.mean(usage == 'Public'):.0%}, seed={args.seed})")


if __name__ == "__main__":
    main()
