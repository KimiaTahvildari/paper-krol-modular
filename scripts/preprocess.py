import sys
import os

# 🔑 Add this to make sure the root folder is in your Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your modules
from config import BASE_DIR, OUTPUT_DIR, RECORDS
from utils.preprocessing import (
    load_ecg,
    load_rhythm_annotations,
    segment_by_rhythm,
    save_scalogram_as_array
)

import os
import gc
import pandas as pd
from config import BASE_DIR, OUTPUT_DIR, RECORDS
from utils.preprocessing import (
    load_ecg,
    load_rhythm_annotations,
    segment_by_rhythm,
    save_scalogram_as_array
)

# Make sure output folders exist
os.makedirs(f"{OUTPUT_DIR}/AFIB", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/nonAFIB", exist_ok=True)

global_metadata = []

for record in RECORDS:
    try:
        record_path = f"{BASE_DIR}/{record}"
        ecg, fs = load_ecg(record_path)
        rhythm_ann = load_rhythm_annotations(record_path)
        afib_segs, nonafib_segs = segment_by_rhythm(ecg, rhythm_ann, fs)

        record_metadata = []

        for i, seg in enumerate(afib_segs):
            fname = f"{record}_afib_{i}.npy"
            fpath = f"{OUTPUT_DIR}/AFIB/{fname}"
            save_scalogram_as_array(seg, fpath)
            record_metadata.append({"file": fname, "label": 1, "record": record})

        for i, seg in enumerate(nonafib_segs):
            fname = f"{record}_nonafib_{i}.npy"
            fpath = f"{OUTPUT_DIR}/nonAFIB/{fname}"
            save_scalogram_as_array(seg, fpath)
            record_metadata.append({"file": fname, "label": 0, "record": record})

        df = pd.DataFrame(record_metadata)
        df.to_csv(f"{OUTPUT_DIR}/metadata_{record}.csv", index=False)

        global_metadata.extend(record_metadata)
        print(f"✅ {record}: {len(afib_segs)} AFIB, {len(nonafib_segs)} non-AFIB")

        gc.collect()

    except Exception as e:
        print(f"❌ {record} failed: {e}")

# Save the full metadata
pd.DataFrame(global_metadata).to_csv(f"{OUTPUT_DIR}/metadata.csv", index=False)
print("✅ metadata.csv saved with all records.")
