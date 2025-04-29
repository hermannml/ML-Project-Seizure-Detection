from pathlib import Path
import pandas as pd
import pickle


def parse_chb_summary(summary_path):
    seizure_dict = {}
    current_file = None
    seizure_start = None

    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                current_file = line.split(":")[1].strip()
                seizure_dict[current_file] = []
            elif line.startswith("Seizure Start Time"):
                seizure_start = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Seizure End Time") and seizure_start is not None:
                seizure_end = float(line.split(":")[1].strip().split()[0])
                seizure_dict[current_file].append((seizure_start, seizure_end))
                seizure_start = None

    return seizure_dict


def parse_edf_seizures_file(seizures_path):
    seizure_events = []
    with open(seizures_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    start = float(parts[1])
                    end = float(parts[2])
                    seizure_events.append((start, end))
                except ValueError:
                    continue
    return seizure_events


def parse_tuh_csv_bi(csv_path):
    df = pd.read_csv(csv_path, comment="#")
    seiz_df = df[df['label'].str.lower() == 'seiz']
    return list(zip(seiz_df['start_time'], seiz_df['stop_time']))


def find_edf_files(base_dir):
    return list(Path(base_dir).rglob("*.edf"))


# ========== Build Annotations ==========

seizure_annotations = {}

# --- CHB-MIT ---
chb_root = Path("physionet.org")
for summary_file in chb_root.rglob("chb*-summary.txt"):
    patient_dir = summary_file.parent
    seizure_dict = parse_chb_summary(summary_file)

    for edf_path in patient_dir.glob("*.edf"):
        events = seizure_dict.get(edf_path.name, [])

        # Fallback to .seizures file if summary gives nothing
        if not events:
            seizures_path = edf_path.with_suffix(edf_path.suffix + ".seizures")
            if seizures_path.exists():
                events = parse_edf_seizures_file(seizures_path)

        seizure_annotations[str(edf_path)] = events


# --- TUH EEG ---
tuh_dir = Path("tuh_eeg_seizure_data")
for edf_path in find_edf_files(tuh_dir):
    csv_path = edf_path.with_suffix(".csv_bi")
    if csv_path.exists():
        df = pd.read_csv(csv_path, comment="#")
        labels = df['label'].str.lower().values

        if 'seiz' in labels:
            events = parse_tuh_csv_bi(csv_path)
            seizure_annotations[str(edf_path)] = events
        elif 'bckg' in labels:
            # Explicitly mark as background-only (no seizures)
            seizure_annotations[str(edf_path)] = []


# --- Save to Pickle ---
with open("real_seizure_annotations.pkl", "wb") as f:
    pickle.dump(seizure_annotations, f)

print(f"Saved {len(seizure_annotations)} EEG files to real_seizure_annotations.pkl")
