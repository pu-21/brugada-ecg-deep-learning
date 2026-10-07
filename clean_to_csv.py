import os
import pandas as pd


LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_header_features(header_path: str):
    with open(header_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("empty header file")

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("invalid first header line")

    record_name = first[0]
    n_leads = int(first[1])
    fs = float(first[2])
    n_samples = int(first[3])

    features = {
        "record_name": record_name,
        "n_leads": n_leads,
        "fs": fs,
        "n_samples": n_samples,
        "duration_sec": n_samples / fs if fs else None,
    }

    seen_leads = []
    for line in lines[1:1 + n_leads]:
        parts = line.split()
        if len(parts) < 9:
            continue

        file_name = parts[0]
        fmt = parts[1]
        adc_info = parts[2]
        adc_res = int(parts[3])
        adc_zero = int(parts[4])
        baseline = int(parts[5])
        checksum = int(parts[6])
        block_size = int(parts[7])
        lead_name = parts[8]

        gain_str = adc_info.split("(")[0]
        gain = float(gain_str.split("/")[0]) if "/" in gain_str else float(gain_str)
        unit = adc_info.split("/")[-1] if "/" in adc_info else ""

        safe_lead = lead_name.replace("/", "_")
        seen_leads.append(lead_name)

        features[f"lead_present_{safe_lead}"] = 1
        features[f"gain_{safe_lead}"] = gain
        features[f"adc_res_{safe_lead}"] = adc_res
        features[f"adc_zero_{safe_lead}"] = adc_zero
        features[f"baseline_{safe_lead}"] = baseline
        features[f"checksum_{safe_lead}"] = checksum
        features[f"block_size_{safe_lead}"] = block_size
        features[f"format_{safe_lead}"] = fmt
        features[f"file_{safe_lead}"] = file_name
        features[f"unit_{safe_lead}"] = unit

    for lead in LEAD_ORDER:
        safe_lead = lead.replace("/", "_")
        if lead not in seen_leads:
            features[f"lead_present_{safe_lead}"] = 0
            features[f"gain_{safe_lead}"] = None
            features[f"adc_res_{safe_lead}"] = None
            features[f"adc_zero_{safe_lead}"] = None
            features[f"baseline_{safe_lead}"] = None
            features[f"checksum_{safe_lead}"] = None
            features[f"block_size_{safe_lead}"] = None
            features[f"format_{safe_lead}"] = None
            features[f"file_{safe_lead}"] = None
            features[f"unit_{safe_lead}"] = None

    features["lead_names"] = "|".join(seen_leads)
    features["all_expected_12_leads_present"] = int(all(lead in seen_leads for lead in LEAD_ORDER))
    return features


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, "metadata.csv")
    files_dir = os.path.join(base_dir, "files")
    out_path = os.path.join(base_dir, "model_ready_ecg_header_features.csv")
    out_waveforms_path = os.path.join(base_dir, "ecg_waveforms_1200.npy")

    print("[clean_to_csv] This script exports header features only.")
    print("[clean_to_csv] For waveform training data, use preprocess_waveforms.py.")

    metadata = pd.read_csv(metadata_path)
    metadata["patient_id"] = metadata["patient_id"].astype(str)

    rows = []
    failed = []

    for _, row in metadata.iterrows():
        pid = row["patient_id"]
        hea_path = os.path.join(files_dir, pid, f"{pid}.hea")

        if not os.path.exists(hea_path):
            failed.append((pid, "missing .hea"))
            continue

        try:
            feat = parse_header_features(hea_path)
            row_out = {
                "patient_id": pid,
                "basal_pattern": row["basal_pattern"],
                "sudden_death": row["sudden_death"],
                "brugada": row["brugada"],
            }
            row_out.update(feat)
            rows.append(row_out)
        except Exception as e:
            failed.append((pid, str(e)))

    df = pd.DataFrame(rows)

    numeric_cols = [c for c in df.columns if c not in ["patient_id", "record_name", "lead_names"] and not c.startswith("format_") and not c.startswith("file_") and not c.startswith("unit_")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    text_fill_cols = [c for c in df.columns if c.startswith("format_") or c.startswith("file_") or c.startswith("unit_")]
    for c in text_fill_cols:
        df[c] = df[c].fillna("missing")

    df = df.sort_values("patient_id").reset_index(drop=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Cols: {len(df.columns)}")
    print(f"Failed records: {len(failed)}")
    print(
        "Note: This script only exports header features. For waveform arrays with the full 1200 time steps, use `preprocess_waveforms.py`."
    )
    print(f"Waveform array target (not generated here): {out_waveforms_path}")

    if failed:
        fail_path = os.path.join(base_dir, "failed_records.csv")
        pd.DataFrame(failed, columns=["patient_id", "reason"]).to_csv(fail_path, index=False, encoding="utf-8-sig")
        print(f"Saved failures: {fail_path}")


if __name__ == "__main__":
    main()
