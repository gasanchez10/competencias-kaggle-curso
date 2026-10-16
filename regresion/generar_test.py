"""
Genera un nuevo conjunto de evaluación (test + solución) para la competencia de
regresión de precios de departamentos (Daegu), a partir del train original.

Diseño del generador
--------------------
1. Atributos del complejo: cada fila nueva toma uno de los 26 complejos del train
   (prob. proporcional a su frecuencia), con todas sus variables de edificio y
   entorno intactas, así la coherencia interna se mantiene.
2. Atributos de la venta (por complejo):
   - Size.sqf.: tomado de los tamaños de planta reales del complejo.
   - Floor: piso observado en el complejo para ese tamaño, con jitter de +-2
     limitado a [1, piso máximo del complejo].
   - YrSold/MonthSold: fecha observada en el complejo con jitter de +-3 meses,
     limitada al rango 2007-01..2017-12 y a YrSold >= YearBuilt.
3. Precio: f(x) * exp(e), donde
   - f es un ensamble de 3 familias distintas (LightGBM, Random Forest y Ridge con
     splines) ajustado en log(SalePrice) del train, para no favorecer a una
     sola familia de modelos;
   - e se toma por bootstrap de los residuos out-of-fold (5-fold) del ensamble,
     por complejo, así el ruido tiene la forma real (colas, heterocedasticidad).

Uso:  .venv/bin/python regresion/generar_test.py [--n 1767] [--seed 2026] [--public 0.3] [--noise-scale s]
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ROOT = Path(__file__).parent
TRAIN = ROOT / "original" / "Train real state.csv"
OUT = ROOT / "nueva_competencia"
TARGET = "SalePrice"

COMPLEX_COLS = [
    "YearBuilt", "HallwayType", "HeatingType", "AptManageType",
    "N_Parkinglot.Ground.", "N_Parkinglot.Basement.", "TimeToBusStop", "TimeToSubway",
    "N_APT", "N_manager", "N_elevators", "SubwayStation",
    "N_FacilitiesNearBy.PublicOffice.", "N_FacilitiesNearBy.Hospital.",
    "N_FacilitiesNearBy.Dpartmentstore.", "N_FacilitiesNearBy.Mall.",
    "N_FacilitiesNearBy.ETC.", "N_FacilitiesNearBy.Park.",
    "N_SchoolNearBy.Elementary.", "N_SchoolNearBy.Middle.", "N_SchoolNearBy.High.",
    "N_SchoolNearBy.University.", "N_FacilitiesInApt",
    "N_FacilitiesNearBy.Total.", "N_SchoolNearBy.Total.",
]
CAT_COLS = ["HallwayType", "HeatingType", "AptManageType", "TimeToBusStop",
            "TimeToSubway", "SubwayStation"]


def load_train():
    df = pd.read_csv(TRAIN).rename(columns={"Unnamed: 0": "id"})
    df["complex"] = df.groupby(COMPLEX_COLS, sort=True).ngroup()
    return df


def features(df):
    X = df.drop(columns=[c for c in ("id", TARGET) if c in df]).copy()
    X["t"] = (X["YrSold"] - 2007) * 12 + X["MonthSold"] - 1
    X["age"] = X["YrSold"] - X["YearBuilt"]
    for c in CAT_COLS + ["complex"]:
        X[c] = X[c].astype("category")
    return X


def build_models():
    num_lin = ["Size.sqf.", "Floor", "t", "age"]
    ridge = make_pipeline(
        ColumnTransformer([
            ("spl", SplineTransformer(n_knots=6, degree=3), num_lin),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["complex"] + CAT_COLS),
        ]),
        Ridge(alpha=1.0),
    )
    rf = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
                          remainder="passthrough"),
        RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0),
    )
    gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                            min_child_samples=10, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, random_state=0, verbose=-1)
    return {"gbm": gbm, "rf": rf, "ridge": ridge}


def fit_predict(models, Xtr, ytr, Xte):
    preds = []
    for name, m in models.items():
        Xa, Xb = Xtr, Xte
        if name == "rf":  # RF no acepta 'category' en columnas passthrough
            Xa = Xtr.assign(complex=Xtr["complex"].cat.codes)
            Xb = Xte.assign(complex=Xte["complex"].cat.codes)
        m.fit(Xa, ytr)
        preds.append(m.predict(Xb))
    return np.mean(preds, axis=0)


def oof_residuals(df, X, y, seed):
    oof = np.zeros(len(df))
    for tr_idx, va_idx in KFold(5, shuffle=True, random_state=seed).split(X):
        oof[va_idx] = fit_predict(build_models(), X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx])
    return y.values - oof


def sample_rows(df, n, rng):
    freq = df["complex"].value_counts(normalize=True).sort_index()
    complexes = rng.choice(freq.index.values, size=n, p=freq.values)
    rows = []
    for c in complexes:
        g = df[df["complex"] == c]
        base = g.iloc[rng.integers(len(g))]
        r = base[COMPLEX_COLS].to_dict()
        r["complex"] = c
        r["Size.sqf."] = int(base["Size.sqf."])
        fmax = int(g["Floor"].max())
        r["Floor"] = int(np.clip(base["Floor"] + rng.integers(-2, 3), 1, fmax))
        t = (base["YrSold"] - 2007) * 12 + base["MonthSold"] - 1 + rng.integers(-3, 4)
        tmin = max(0, (int(r["YearBuilt"]) - 2007) * 12)
        t = int(np.clip(t, tmin, 11 * 12 - 1))
        r["YrSold"], r["MonthSold"] = 2007 + t // 12, t % 12 + 1
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1767)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--public", type=float, default=0.3, help="fracción del leaderboard público")
    ap.add_argument("--noise-scale", type=float, default=0.55,
                    help="factor sobre los residuos OOF (calibrado con validar_test.py)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = load_train()
    X, y = features(df), np.log(df[TARGET])

    print("Calculando residuos out-of-fold...")
    resid = oof_residuals(df, X, y, args.seed)
    print(f"  MAE OOF del ensamble (escala original): "
          f"{np.mean(np.abs(df[TARGET] - np.exp(y - resid))):,.0f}")
    resid_by_complex = pd.Series(resid).groupby(df["complex"].values).apply(np.array)

    print("Muestreando filas nuevas...")
    new = sample_rows(df, args.n, rng)
    Xnew = features(new.assign(**{c: pd.Categorical(new[c], categories=X[c].cat.categories)
                                  for c in CAT_COLS + ["complex"]}))
    Xnew = Xnew[X.columns]
    for c in CAT_COLS + ["complex"]:
        Xnew[c] = pd.Categorical(Xnew[c], categories=X[c].cat.categories)

    mu = fit_predict(build_models(), X, y, Xnew)
    eps = args.noise_scale * np.array([rng.choice(resid_by_complex[c]) for c in new["complex"]])
    new[TARGET] = np.round(np.exp(mu + eps)).astype(int)
    lo, hi = df[TARGET].min(), df[TARGET].max()
    new[TARGET] = new[TARGET].clip(int(lo * 0.9), int(hi * 1.1))

    # ids nuevos, sin relación con los del dataset público
    new["id"] = rng.permutation(np.arange(100001, 100001 + args.n))
    new = new.sort_values("id").reset_index(drop=True)
    cols = [c for c in pd.read_csv(TRAIN, nrows=0).columns if c not in ("Unnamed: 0", TARGET)]

    out = args.out
    out.mkdir(exist_ok=True)
    test = new[["id"] + cols]
    test.to_csv(out / "Test real state.csv", index=False)

    usage = np.where(rng.random(args.n) < args.public, "Public", "Private")
    sol = pd.DataFrame({"id": new["id"], TARGET: new[TARGET], "Usage": usage})
    sol.to_csv(out / "solution.csv", index=False)

    pd.DataFrame({"id": new["id"], TARGET: int(df[TARGET].median())}).to_csv(
        out / "sample_submission.csv", index=False)

    tr_out = df.drop(columns=["complex"])[["id", TARGET] + cols]
    tr_out.to_csv(out / "Train real state.csv", index=False)

    print(f"Listo: {args.n} filas en {out} "
          f"(Public={np.mean(usage == 'Public'):.0%}, seed={args.seed})")


if __name__ == "__main__":
    main()
