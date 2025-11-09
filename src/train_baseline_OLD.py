# OLD DONT USE




































# src/train_baseline.py
from __future__ import annotations

import argparse, json, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# Parsing / normalization
# =========================
Q_END = "end"
PLOT_DIV = 1000.0  # m.kr. -> ISK bn

DATE_HINTS = {
    "date","quarter","árs­fjórðungur","ársfjórðungur","arsfjordungur","month"
}

def to_quarter_end(series: pd.Series) -> pd.Series:
    """Robustly map strings/datetimes to quarter-end timestamps."""
    if np.issubdtype(series.dtype, np.datetime64):
        return pd.PeriodIndex(series, freq="M").asfreq("Q").to_timestamp(how=Q_END)

    s = series.astype(str).str.strip()

    # 2019Q3 / 2024q1
    mask_q = s.str.match(r"^\d{4}Q[1-4]$", case=False, na=False)
    if mask_q.any():
        return pd.PeriodIndex(s.str.upper(), freq="Q").to_timestamp(how=Q_END)

    # 2015M01
    mask_m = s.str.match(r"^\d{4}M\d{2}$", na=False)
    if mask_m.any():
        y = s.str.slice(0, 4)
        m = s.str.slice(5, 7)
        dt = pd.to_datetime(y + "-" + m + "-01", errors="coerce")
        return pd.PeriodIndex(dt, freq="M").asfreq("Q").to_timestamp(how=Q_END)

    # Fallback: any date-ish string (supports day-first)
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return pd.PeriodIndex(dt, freq="M").asfreq("Q").to_timestamp(how=Q_END)

def pick_date_col(df: pd.DataFrame, user_col: str | None) -> str:
    if user_col and user_col in df.columns: return user_col
    # heuristic match on known names
    cand = [c for c in df.columns if str(c).lower() in DATE_HINTS]
    if not cand:
        # any datetime-like
        cand = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if not cand:
        cand = [df.columns[0]]
    return cand[0]

def pick_value_col(df: pd.DataFrame, user_col: str | None, date_col: str) -> str:
    if user_col and user_col in df.columns: return user_col
    numeric = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if numeric: return numeric[0]
    # try coercion
    for c in df.columns:
        if c == date_col: continue
        co = pd.to_numeric(df[c], errors="coerce")
        if co.notna().sum() > 0:
            df[c] = co
            return c
    raise ValueError("No numeric value column found; pass --value-col.")

def load_target_any(path: pathlib.Path,
                    value_col: str | None,
                    date_col: str | None,
                    sheet: str | int | None,
                    agg: str = "sum") -> pd.DataFrame:
    """Load parquet/json/xlsx/csv -> quarterly DataFrame with columns [date, value]."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(path)
    elif suf == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        from src.parse_pxweb import pxweb_to_frame
        df = pxweb_to_frame(raw)
    elif suf in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, sheet_name=(sheet if sheet is not None else 0), engine="openpyxl")
        except ImportError as e:
            raise ImportError("openpyxl is required to read .xlsx; install it in your env.") from e
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suf}")

    dcol = pick_date_col(df, date_col)
    vcol = pick_value_col(df, value_col, dcol)

    q = to_quarter_end(df[dcol])
    val = pd.to_numeric(df[vcol], errors="coerce")
    out = (pd.DataFrame({"date": q, "value": val})
             .dropna(subset=["date", "value"])
             .sort_values("date"))

    if agg not in {"sum","mean"}:
        raise ValueError("--agg must be 'sum' or 'mean'")
    out = (out.groupby("date", as_index=False)["value"]
              .agg(agg, numeric_only=True))
    return out

# ======================
# Features / modeling
# ======================
def make_lag_features(df: pd.DataFrame,
                      target: str,
                      lags=(1,2,3,6,12),
                      rolls=(3,6,12)) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for L in lags:
        df[f"lag_{L}"] = df[target].shift(L)
    for w in rolls:
        s1 = df[target].shift(1)
        df[f"rollmean_{w}"] = s1.rolling(w, min_periods=1).mean()
        df[f"rollstd_{w}"]  = s1.rolling(w, min_periods=2).std()
    q = pd.Categorical(df["date"].dt.quarter, categories=[1, 2, 3, 4])
    df = pd.concat([df, pd.get_dummies(q, prefix="Q", dtype=int)], axis=1)
    return df


def time_split(df: pd.DataFrame, frac_test=0.2):
    n_test = max(1, int(round(len(df) * frac_test)))
    return df.iloc[:-n_test], df.iloc[-n_test:]

def eval_metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = (np.abs((y_true - y_pred) / y_true)
              .replace([np.inf, -np.inf], np.nan)
              .dropna()).mean() * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_%": (None if np.isnan(mape) else float(mape))}

# ======================
# CLI / main
# ======================
def main():
    ap = argparse.ArgumentParser(description="Baseline XGB on quarterly aluminium exports.")
    ap.add_argument("--target", required=True, help="Path to parquet/json/xlsx/csv.")
    ap.add_argument("--value-col", default=None, help="Optional explicit value column.")
    ap.add_argument("--date-col",  default=None, help="Optional explicit date/quarter column.")
    ap.add_argument("--sheet",     default=None, help="Excel sheet (name or index).")
    ap.add_argument("--agg",       default="sum", choices=["sum","mean"], help="Monthly->quarter aggregation.")
    ap.add_argument("--test-frac", type=float, default=0.20, help="Fraction for test split (default 0.20).")
    ap.add_argument("--seed",      type=int,   default=42, help="Random seed for XGB.")
    ap.add_argument("--plot-start", default="2015-01-01", help="Left x-axis bound.")
    ap.add_argument("--out_dir",   required=True, help="Output directory.")
    args = ap.parse_args()

    p = pathlib.Path(args.target)
    sheet = int(args.sheet) if (args.sheet and str(args.sheet).isdigit()) else args.sheet

    # load + features
    df = load_target_any(p, args.value_col, args.date_col, sheet, agg=args.agg)
    df = df.rename(columns={"value":"y"})
    df = make_lag_features(df, target="y").dropna()

    train_df, test_df = time_split(df, frac_test=args.test_frac)

    features = [c for c in train_df.columns if c not in ("date","y")]
    model = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
        random_state=args.seed, missing=np.nan
    )
    model.fit(train_df[features], train_df["y"])
    preds = model.predict(test_df[features])
    metrics = eval_metrics(test_df["y"], preds)

    # baselines
    naive_last   = df["y"].shift(1).loc[test_df.index]
    naive_season = df["y"].shift(4).loc[test_df.index]
    naive = {
        "naive_last":   eval_metrics(test_df["y"], naive_last),
        "naive_season": eval_metrics(test_df["y"], naive_season),
    }
    impr_mae  = 1 - (metrics["MAE"]  / naive["naive_season"]["MAE"])
    impr_rmse = 1 - (metrics["RMSE"] / naive["naive_season"]["RMSE"])
    improvement = {"vs_naive_t4": {"MAE_pct": float(100*impr_mae),
                                   "RMSE_pct": float(100*impr_rmse)}}

    # outputs
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "naive_metrics.json").write_text(json.dumps(naive, indent=2), encoding="utf-8")
    (out_dir / "improvement.json").write_text(json.dumps(improvement, indent=2), encoding="utf-8")

    # plot
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(train_df["date"], train_df["y"]/PLOT_DIV, label="train",  color="#2ca02c")
    ax.plot(test_df["date"],  test_df["y"]/PLOT_DIV,  label="actual", color="#1f77b4")
    ax.plot(test_df["date"],  (pd.Series(preds, index=test_df.index)/PLOT_DIV),
            label="pred", color="#ff7f0e")
    ax.plot(test_df["date"],  (naive_season/PLOT_DIV), label="naive (t-4)",
            color="gray", linestyle="--", linewidth=1.5)

    split_date = test_df["date"].min()
    ax.axvline(split_date, linestyle=":", color="black", alpha=0.7, linewidth=1.2, label="train/test split")
    ax.axvspan(split_date, df["date"].max(), color="gray", alpha=0.06, label="_nolegend_")

    ax.set_xlim(pd.Timestamp(args.plot_start), df["date"].max())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title("Aluminium exports (ISK) — baseline model")
    ax.set_ylabel("Value (ISK bn)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    txt = (f"Model  MAE={metrics['MAE']/PLOT_DIV:,.2f} bn, "
           f"RMSE={metrics['RMSE']/PLOT_DIV:,.2f} bn, "
           f"MAPE={metrics['MAPE_%']:.1f}%  |  "
           f"Naive(t-4) MAE={naive['naive_season']['MAE']/PLOT_DIV:,.2f} bn  "
           f"(Improvement: {100*(impr_mae):.1f}% MAE)")
    ax.text(0.01, 0.02, txt, transform=ax.transAxes, fontsize=9, alpha=0.9)

    fig.tight_layout()
    fig_path = out_dir / "pred_vs_actual.png"
    fig.savefig(fig_path, dpi=160)

    # predictions CSV
    pred_df = test_df[["date"]].copy()
    pred_df["y_true"] = test_df["y"].values
    pred_df["y_pred"] = preds
    pred_df["naive_last"] = naive_last.values
    pred_df["naive_season"] = naive_season.values
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print("Saved:", fig_path)
    print("Metrics:", metrics, "| Naive:", naive)

if __name__ == "__main__":
    main()
