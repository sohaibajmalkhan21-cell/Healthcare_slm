import os
import glob
import pandas as pd
import numpy as np

VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "DBP", "Resp"]
PLAUSIBLE_RANGES = {
    "HR": (20, 300), "O2Sat": (50, 100), "Temp": (25, 45),
    "SBP": (40, 300), "DBP": (20, 200), "Resp": (2, 80),
}
DATA_DIR = "../data/physionet.org/files/challenge-2019/1.0.0/training"
OUT_PATH = "../outputs/patients_combined.parquet"

def load_all_patients():
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.psv"), recursive=True)
    print(f"Found {len(files)} psv files")
    frames = []
    for i, f in enumerate(files):
        df = pd.read_csv(f, sep="|")
        df["patient_id"] = os.path.basename(f).replace(".psv", "")
        df["hour"] = range(len(df))
        frames.append(df)
        if (i+1) % 5000 == 0:
            print(f"Loaded {i+1}/{len(files)}")
    return pd.concat(frames, ignore_index=True)

def remove_outliers(df):
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        mask = (df[col] < lo) | (df[col] > hi)
        if mask.sum() > 0:
            print(f"{col}: removing {mask.sum()} out-of-range values")
            df.loc[mask, col] = np.nan
    return df

def clean(df):
    df = df.dropna(subset=["SepsisLabel"])
    df = remove_outliers(df)
    df[VITAL_COLS] = df.groupby("patient_id")[VITAL_COLS].transform(lambda x: x.ffill().bfill())
    residual = df[VITAL_COLS].isna().mean() * 100
    print("\nResidual missingness after fill (%):")
    print(residual)
    before = len(df)
    df = df.dropna(subset=VITAL_COLS, how="all")
    print(f"Dropped {before - len(df)} rows with zero usable vitals")
    return df

if __name__ == "__main__":
    os.makedirs("../outputs", exist_ok=True)
    df = load_all_patients()
    df = clean(df)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nFinal: {len(df)} rows, {df['patient_id'].nunique()} patients")
    print(f"Sepsis-positive rows: {df['SepsisLabel'].sum()} ({100*df['SepsisLabel'].mean():.2f}%)")
    print(f"Saved to {OUT_PATH}")
