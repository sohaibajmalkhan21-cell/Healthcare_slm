import pandas as pd

VITAL_MAP = {
    "HR": "heart rate", "O2Sat": "SpO2", "Temp": "temperature",
    "SBP": "systolic blood pressure", "DBP": "diastolic blood pressure", "Resp": "respiratory rate",
}
UNIT_MAP = {"HR": "bpm", "O2Sat": "%", "Temp": "C", "SBP": "mmHg", "DBP": "mmHg", "Resp": "breaths/min"}
# Same normal-range logic as data/knowledge_base/vitals_reference.json
NORMAL_RANGES = {
    "HR": (60, 100), "O2Sat": (95, 100), "Temp": (36.1, 37.2),
    "SBP": (90, 120), "DBP": (60, 80), "Resp": (12, 20),
}

def summarize_window(window_df: pd.DataFrame) -> str:
    parts = []
    abnormal_count = 0
    for col, name in VITAL_MAP.items():
        vals = window_df[col].dropna().tolist()
        if len(vals) == 0:
            continue
        start, end, vmin, vmax = vals[0], vals[-1], min(vals), max(vals)
        lo, hi = NORMAL_RANGES[col]
        trend = "rose from" if end > start else "dropped from" if end < start else "stayed at"
        if trend == "stayed at":
            desc = f"{name} stayed at {round(end,1)} {UNIT_MAP[col]}"
        else:
            desc = f"{name} {trend} {round(start,1)} to {round(end,1)} {UNIT_MAP[col]} (range {round(vmin,1)}-{round(vmax,1)})"
        if end < lo or end > hi:
            desc += f" [OUTSIDE normal range {lo}-{hi} {UNIT_MAP[col]}]"
            abnormal_count += 1
        parts.append(desc)
    if not parts:
        return None
    summary = "Over the observed period, " + "; ".join(parts) + "."
    summary += f" {abnormal_count} of {len(parts)} vital signs are currently outside normal clinical range."
    return summary

def build_summary_for_patient_hour(patient_df: pd.DataFrame, hour: int, window_size: int = 6):
    start = max(0, hour - window_size + 1)
    window = patient_df[(patient_df["hour"] >= start) & (patient_df["hour"] <= hour)]
    if len(window) == 0:
        return None
    return summarize_window(window)
