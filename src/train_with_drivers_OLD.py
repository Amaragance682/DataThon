# OLD DONT USE


































# src/train_with_drivers.py
import argparse, pathlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- metrics & helpers ----------
def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = (np.abs((y_true - y_pred) / y_true)
              .replace([np.inf, -np.inf], np.nan)
              .dropna()).mean() * 100
    return {"MAE": float(mae), "RMSE": float(rmse),
            "MAPE_%": None if np.isnan(mape) else float(mape)}

def make_ts_features(df, target="y", lags=(1,2,3,6,12), rolls=(3,6,12)):
    df = df.sort_values("date").copy()
    # lags/rolls of target
    for L in lags:
        df[f"lag_{L}"] = df[target].shift(L)
    for w in rolls:
        df[f"rollmean_{w}"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f"rollstd_{w}"]  = df[target].shift(1).rolling(w, min_periods=2).std()
    # simple seasonal dummies (safe)
    q = df["date"].dt.quarter
    df = pd.concat([df, pd.get_dummies(q, prefix="Q", dtype=int)], axis=1)
    return df

def time_split(df, frac_test=0.2):
    n_test = max(1, int(round(len(df) * frac_test)))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()

# ---------- drivers ----------
def load_fx_monthly(drivers_dir: pathlib.Path) -> pd.DataFrame:
    path = next(iter(
        list(drivers_dir.glob("Exchange-rates_2015-2025.csv")) or
        list(drivers_dir.glob("**/Exchange-rates_2015-2025.csv"))
    ), None)
    if path is None:
        raise FileNotFoundError("Exchange-rates_2015-2025.csv not found")

    fx = pd.read_csv(path, sep=";", decimal=",", thousands=".", encoding="utf-8-sig",
                     engine="python", on_bad_lines="skip", na_values=["-"])
    if "Dagsetning" not in fx.columns:
        raise ValueError(f"'Dagsetning' missing. Columns: {list(fx.columns)}")

    fx["Dagsetning"] = pd.to_datetime(fx["Dagsetning"], format="%d.%m.%Y",
                                      errors="coerce", dayfirst=True)
    fx = fx.dropna(subset=["Dagsetning"])
    q = pd.PeriodIndex(fx["Dagsetning"], freq="M").asfreq("Q").to_timestamp(how="end")

    def pick(bits):
        cols = [c for c in fx.columns if all(s in c.lower() for s in bits)]
        return cols[0] if cols else None

    eur = pick(["eur","mið"]) or pick(["eur","mid"])
    usd = pick(["usd","mið"]) or pick(["usd","mid"])
    if eur is None or usd is None:
        raise ValueError(f"EUR/USD mid-rate columns not found. Columns: {list(fx.columns)}")

    fx = (fx.assign(q=q)
            .groupby("q")[[eur, usd]].mean(numeric_only=True).reset_index()
            .rename(columns={"q":"date", eur:"EURISK", usd:"USDISK"}))
    for c in ["EURISK", "USDISK"]:
        fx[c+"_yoy"] = fx[c].pct_change(4)
    return fx

def load_cpi_monthly(drivers_dir: pathlib.Path) -> pd.DataFrame:
    path = next(iter(
        list(drivers_dir.glob("Inflation-Consumer price index.csv")) or
        list(drivers_dir.glob("**/Inflation-Consumer price index.csv"))
    ), None)
    if path is None:
        raise FileNotFoundError("Inflation-Consumer price index.csv not found")

    cpi = pd.read_csv(path, encoding="utf-8-sig")
    if "Month" not in cpi.columns:
        raise ValueError(f"'Month' missing. Columns: {list(cpi.columns)}")

    yy_mm = cpi["Month"].astype(str).str.extract(r"(?P<y>\d{4})M(?P<m>\d{2})")
    cpi["date"] = pd.to_datetime(yy_mm["y"] + "-" + yy_mm["m"] + "-01", errors="coerce")
    cpi["q"] = pd.PeriodIndex(cpi["date"], freq="M").asfreq("Q").to_timestamp(how="end")

    ccol = "Consumer price index Index"
    if ccol not in cpi.columns:
        nums = [c for c in cpi.columns if pd.api.types.is_numeric_dtype(cpi[c])]
        if not nums: raise ValueError("No numeric CPI column found.")
        ccol = nums[0]

    cpi_q = (cpi.groupby("q")[[ccol]].mean(numeric_only=True).reset_index()
               .rename(columns={"q":"date", ccol:"CPI"}))
    cpi_q["CPI_yoy"] = cpi_q["CPI"].pct_change(4)
    return cpi_q

def load_cargo_monthly(drivers_dir: pathlib.Path) -> pd.DataFrame:
    xls = next(iter(
        list(drivers_dir.glob("Cargo_flights- KEF frakt.xlsx")) or
        list(drivers_dir.glob("**/Cargo_flights- KEF frakt.xlsx"))
    ), None)
    if xls is None:
        raise FileNotFoundError("Cargo_flights- KEF frakt.xlsx not found.")

    cg = pd.read_excel(xls, engine="openpyxl")
    cg = cg.loc[:, ~cg.columns.astype(str).str.startswith("Unnamed")]
    cg.columns = [str(c).strip() for c in cg.columns]

    # rename Year/Month, keep any “Total” as-is
    ren = {}
    for c in cg.columns:
        lc = c.lower()
        if lc in {"ár","ar","year"}: ren[c] = "Year"
        elif lc in {"mán","mán.","mánuður","man","month"}: ren[c] = "Month"
        elif lc.strip().startswith("total"): ren[c] = "Total"
    cg = cg.rename(columns=ren)
    if "Year" not in cg.columns or "Month" not in cg.columns:
        cg = cg.rename(columns={cg.columns[0]:"Year", cg.columns[1]:"Month"})

    import unicodedata
    def deaccent(s): return "".join(ch for ch in unicodedata.normalize("NFKD", str(s))
                                    if not unicodedata.combining(ch))
    m = cg["Month"].astype(str).str.strip().str.lower().map(deaccent)
    mmap = {"jan":1,"feb":2,"mar":3,"apr":4,"mai":5,"may":5,
            "jun":6,"jún":6,"juni":6,"jul":7,"júl":7,
            "agu":8,"ágú":8,"aug":8,"sep":9,"okt":10,"oct":10,
            "nov":11,"nóv":11,"des":12,"dec":12}
    m_num = m.map(mmap)
    m_num = pd.to_numeric(m_num, errors="coerce").fillna(pd.to_numeric(cg["Month"], errors="coerce"))
    y_num = pd.to_numeric(cg["Year"], errors="coerce")

    cg["date"] = pd.to_datetime(dict(year=y_num, month=m_num, day=1), errors="coerce")
    cg = cg.dropna(subset=["date"])

    if "Total" in cg.columns:
        total = cg["Total"]
    else:
        goods_tot = next((c for c in cg.columns if "vörur" in c.lower() and "samtals" in c.lower()), None)
        post_tot  = next((c for c in cg.columns if "póstur" in c.lower() and "samtals" in c.lower()), None)
        if goods_tot and post_tot:
            total = pd.to_numeric(cg[goods_tot], errors="coerce") + pd.to_numeric(cg[post_tot], errors="coerce")
        else:
            nums = [c for c in cg.columns if c not in {"Year","Month","date"} and pd.api.types.is_numeric_dtype(cg[c])]
            if not nums: raise ValueError("No numeric cargo column found.")
            total = cg[nums[0]]

    if total.dtype == object:
        total = (total.astype(str)
                 .str.replace(r"[^\d,.\-]", "", regex=True)
                 .str.replace(".", "", regex=False)
                 .str.replace(",", ".", regex=False))
        total = pd.to_numeric(total, errors="coerce")

    cg = cg.assign(cargo_cnt=total)
    cg = cg.loc[cg["date"] >= "2015-01-01", ["date","cargo_cnt"]].sort_values("date")

    cg["q"] = pd.PeriodIndex(cg["date"], freq="M").asfreq("Q").to_timestamp(how="end")
    cg_q = (cg.groupby("q", as_index=False)
              .agg(cargo_cnt=("cargo_cnt","sum"))
              .rename(columns={"q":"date"}))
    cg_q["cargo_cnt_yoy"] = cg_q["cargo_cnt"].astype(float).pct_change(4)
    return cg_q

# ---------- merge ----------
def merge_drivers(target_q: pd.DataFrame, drivers: list[pd.DataFrame]) -> pd.DataFrame:
    df = target_q.copy()
    for d in drivers:
        df = df.merge(d, on="date", how="left")
    for c in df.columns:
        if c not in ("date","y"):
            df[c] = df[c].ffill()
    return df

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_parquet", default="data/processed/aluminium_exports.parquet")
    ap.add_argument("--drivers_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_log", action="store_true", help="disable log1p target")
    ap.add_argument("--early_stop", type=int, default=0, help="rounds for early stopping (0=off)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # target
    df_t = pd.read_parquet(args.target_parquet)[["date","value"]].rename(columns={"value":"y"})
    df_t = df_t.sort_values("date").reset_index(drop=True)

    # drivers
    drivers_dir = pathlib.Path(args.drivers_dir)
    fx_q    = load_fx_monthly(drivers_dir)
    cpi_q   = load_cpi_monthly(drivers_dir)
    cargo_q = load_cargo_monthly(drivers_dir)

    # feature set definitions
    sets = {
        "base_lags":             [],
        "base+fx":               [fx_q],
        "base+fx+cpi":           [fx_q, cpi_q],
        "base+fx+cpi+cargo":     [fx_q, cpi_q, cargo_q],
    }

    PLOT_DIV = 1000.0
    USE_LOG  = (not args.no_log)
    results = {}

    for name, drivers in sets.items():
        data = merge_drivers(df_t, drivers)
        data = make_ts_features(data, target="y")

        # create lagged versions of ALL exogenous columns (no leakage)
        exo_raw = [c for c in data.columns if c not in ("date","y") and not c.startswith(("lag_","roll","Q_"))]
        for c in exo_raw:
            data[c+"_lag1"] = data[c].shift(1)
            data[c+"_lag4"] = data[c].shift(4)
        data.drop(columns=exo_raw, inplace=True)

        # split AFTER feature engineering
        train_df, test_df = time_split(data, frac_test=0.2)

        # target transform
        target_col = "y"
        if USE_LOG:
            train_df.loc[:, "y_log"] = np.log1p(train_df["y"])
            test_df.loc[:,  "y_log"] = np.log1p(test_df["y"])
            target_col = "y_log"

        feat_cols = [c for c in data.columns if c not in ("date","y","y_log")]

        # keep rows where target exists and AT LEAST ONE feature is present
        mask_tr = train_df[target_col].notna() & train_df[feat_cols].notna().any(axis=1)
        train_df = train_df.loc[mask_tr].copy()

        # in TEST: keep y; allow NaNs in features (XGBoost handles them)
        test_df = test_df.loc[test_df["y"].notna()].copy()

        # safety guard
        if len(train_df) < 5 or len(test_df) < 1:
            raise RuntimeError(
                f"Too few rows after cleaning. train={len(train_df)}, test={len(test_df)}. "
                "Likely a driver ends before the test window; keep NaNs or trim drivers."
            )

        # small validation tail; if too small, skip early stopping
        val_h = max(1, min(6, len(train_df)//8))
        split_i = max(1, len(train_df) - val_h)
        X_tr, y_tr = train_df.iloc[:split_i][feat_cols], train_df.iloc[:split_i][target_col]
        X_va, y_va = train_df.iloc[split_i:][feat_cols],  train_df.iloc[split_i:][target_col]
        use_es = (args.early_stop > 0 and len(X_va) > 0)


        model = xgb.XGBRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            missing=np.nan,
            eval_metric="rmse",
        )

        def fit_xgb_robust(model, X_tr, y_tr, X_va=None, y_va=None, early_stop=0):
            # If no validation or early stopping requested, just fit.
            if X_va is None or early_stop <= 0:
                model.fit(X_tr, y_tr)
                return model

            # 1) Try new callback API (xgboost >= 1.6)
            try:
                import xgboost as xgb
                cb = getattr(xgb.callback, "EarlyStopping", None)
                if cb is not None:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False,
                        callbacks=[cb(rounds=early_stop, save_best=True)],
                    )
                    return model
            except Exception:
                pass

            # 2) Try legacy kwarg (older xgboost)
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                    early_stopping_rounds=early_stop,
                )
                return model
            except TypeError:
                pass

            # 3) Final fallback: fit without early stopping
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            return model

        model = fit_xgb_robust(model, X_tr, y_tr, X_va if use_es else None, y_va if use_es else None, args.early_stop)
        
        preds = model.predict(test_df[feat_cols])
        if USE_LOG:
            preds = np.expm1(preds)

        metrics = eval_metrics(test_df["y"], preds)

        # baselines
        naive_last   = data["y"].shift(1).loc[test_df.index]
        naive_season = data["y"].shift(4).loc[test_df.index]
        naive_m = {"naive_last": eval_metrics(test_df["y"], naive_last),
                   "naive_season": eval_metrics(test_df["y"], naive_season)}

        impr_mae  = 1 - metrics["MAE"]/naive_m["naive_season"]["MAE"]
        impr_rmse = 1 - metrics["RMSE"]/naive_m["naive_season"]["RMSE"]

        results[name] = {
            "metrics": metrics,
            "naive": naive_m,
            "improvement_vs_t4": {"MAE_pct": float(100*impr_mae),
                                  "RMSE_pct": float(100*impr_rmse)}
        }

        # ---------- plot ----------
        fig = plt.figure(figsize=(12,4)); ax = plt.gca()
        ax.plot(train_df["date"], train_df["y"]/PLOT_DIV, label="train",  color="#2ca02c")
        ax.plot(test_df["date"],  test_df["y"]/PLOT_DIV,  label="actual", color="#1f77b4")
        ax.plot(test_df["date"],  (pd.Series(preds, index=test_df.index)/PLOT_DIV),
                label="pred", color="#ff7f0e")
        ax.plot(test_df["date"],  (naive_season/PLOT_DIV),
                label="naive (t-4)", color="gray", linestyle="--")

        split_date = test_df["date"].min() - pd.Timedelta(days=1)
        ax.axvline(split_date, linestyle=":", color="black", alpha=0.7, linewidth=1.2, label="train/test split")
        ax.axvspan(split_date, data["date"].max(), color="gray", alpha=0.06, label="_nolegend_")

        ax.set_xlim(pd.Timestamp("2015-01-01"), data["date"].max())
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_title(f"Aluminium exports (ISK) — {name}")
        ax.set_ylabel("Value (ISK bn)")
        ax.grid(True, alpha=0.25); ax.legend()

        txt = (f"MAE={metrics['MAE']/PLOT_DIV:,.2f} bn | "
               f"Naive(t-4)={naive_m['naive_season']['MAE']/PLOT_DIV:,.2f} bn "
               f"(impr {100*(1 - metrics['MAE']/naive_m['naive_season']['MAE']):.1f}%)")
        ax.text(0.01, 0.02, txt, transform=ax.transAxes, fontsize=9, alpha=0.9)

        fig.tight_layout()
        fig.savefig(out_dir / f"pred_vs_actual_{name}.png", dpi=160)
        plt.close(fig)

    # comparison table
    comp = [{
        "model": k,
        "MAE": v["metrics"]["MAE"],
        "RMSE": v["metrics"]["RMSE"],
        "MAPE_%": v["metrics"]["MAPE_%"],
        "naive_t4_MAE": v["naive"]["naive_season"]["MAE"],
        "impr_MAE_%": v["improvement_vs_t4"]["MAE_pct"],
    } for k,v in results.items()]
    pd.DataFrame(comp).sort_values("MAE").to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "comparison.csv")

if __name__ == "__main__":
    main()
