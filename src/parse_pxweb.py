import json, pathlib, pandas as pd, numpy as np
from typing import Tuple

def pxweb_to_frame(px: dict) -> pd.DataFrame:
    # include both regular dims ("d") and time dimensionss ("t")
    dim_codes = [c["code"] for c in px["columns"] if c["type"] in ("d", "t")]

    records = []
    for row in px["data"]:
        key = row["key"]  # aligns with dim_codes
        rec = {dim_codes[i]: key[i] for i in range(len(dim_codes))}
        v = row.get("values")
        if isinstance(v, list):
            v = v[0] if len(v) else None
        rec["value"] = None if v in (None, ".", "..", ":") else float(v.replace(",", ".")) if isinstance(v, str) else v
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # normalize common names (add Icelandic -> English), think we did this in train.py too? just to be safe lmao
    rename = {
        "Ár": "year",
        "Ársfjórðungur": "quarter",
        "Mánuður": "month",
        "Tími": "time",
        "Time": "time",
        "Skipting": "category",
        "Vara": "product",
        "Eining": "unit",
        "Unit": "unit",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # build datetime
    if "month" in df.columns:
        df["date"] = pd.to_datetime(df["month"].astype(str).str.replace("M", "-"), errors="coerce")
    elif "quarter" in df.columns:
        df["date"] = pd.PeriodIndex(df["quarter"].astype(str), freq="Q").to_timestamp(how="end")
    elif "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce")
    elif "year" in df.columns:
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-12-31", errors="coerce")
    else:
        df["date"] = pd.NaT
    return df.sort_values("date").reset_index(drop=True)

def save_parquet(df: pd.DataFrame, path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    px = json.loads(pathlib.Path(args.raw).read_text(encoding="utf-8"))
    df = pxweb_to_frame(px)
    save_parquet(df, args.out)
    print(df.head())
