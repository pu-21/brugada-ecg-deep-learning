import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError as exc:
    raise ImportError(
        "This script requires the `wfdb` package. Install it with `pip install wfdb`."
    ) from exc


LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXPECTED_TIME_STEPS = 1200


def load_record(record_dir: Path, record_name: str):
    record_path = record_dir / record_name
    signal, fields = wfdb.rdsamp(str(record_path))
    return signal, fields


def main():
    base_dir = Path(__file__).resolve().parent
    metadata_path = base_dir / "metadata.csv"
    files_dir = base_dir / "files"
    out_x = base_dir / "ecg_waveforms_1200.npy"
    out_y = base_dir / "ecg_labels.npy"
    out_meta = base_dir / "ecg_waveforms_index.csv"
    failed_path = base_dir / "failed_waveform_records.csv"

    metadata = pd.read_csv(metadata_path)
    metadata["patient_id"] = metadata["patient_id"].astype(str)

    x_list = []
    y_list = []
    rows = []
    failed = []

    for _, row in metadata.iterrows():
        pid = row["patient_id"]
        record_dir = files_dir / pid
        record_name = pid
        hea_path = record_dir / f"{pid}.hea"

        if not hea_path.exists():
            failed.append((pid, "missing .hea"))
            continue

        try:
            signal, fields = load_record(record_dir, record_name)
            sig_len = signal.shape[0]
            n_leads = signal.shape[1]

            if sig_len != EXPECTED_TIME_STEPS:
                failed.append((pid, f"unexpected time steps: {sig_len}"))
                continue

            if n_leads != 12:
                failed.append((pid, f"unexpected lead count: {n_leads}"))
                continue

            x_list.append(signal.astype(np.float32))
            y_list.append(int(row["brugada"]))
            rows.append(
                {
                    "patient_id": pid,
                    "basal_pattern": int(row["basal_pattern"]),
                    "sudden_death": int(row["sudden_death"]),
                    "brugada": int(row["brugada"]),
                    "n_time_steps": sig_len,
                    "n_leads": n_leads,
                    "fs": fields.get("fs"),
                }
            )
        except Exception as exc:
            failed.append((pid, str(exc)))

    if not x_list:
        raise RuntimeError("No waveform records were successfully loaded.")

    x = np.stack(x_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    np.save(out_x, x)
    np.save(out_y, y)
    pd.DataFrame(rows).to_csv(out_meta, index=False, encoding="utf-8-sig")

    print(f"Saved X: {out_x} shape={x.shape}")
    print(f"Saved y: {out_y} shape={y.shape}")
    print(f"Saved index: {out_meta}")
    print(f"Loaded records: {len(rows)}")
    print(f"Failed records: {len(failed)}")

    if failed:
        pd.DataFrame(failed, columns=["patient_id", "reason"]).to_csv(
            failed_path, index=False, encoding="utf-8-sig"
        )
        print(f"Saved failures: {failed_path}")


if __name__ == "__main__":
    main()
