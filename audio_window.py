#!/usr/bin/env python3
"""
Puma Bioacoustics Segment Extractor
====================================
Processes puma audio recordings and their corresponding Raven-style label tables
to produce labeled 5-second audio segments and a master metadata CSV.

Inputs:
    - Label table (.txt): Tab-separated Raven selection table with Begin/End times
      and Annotation columns.
    - Audio folder: Directory of .wav files named puma-[PumaID]_audio_[YYMMDD]-[HHMMSS].wav

Outputs (in ./output/):
    - Individual 5-second .wav segments: [PumaID]_[Date]_[AbsoluteStartSecond].wav
    - master_metadata.csv with columns:
        window_filename, label, source_file, start_time_in_source,
        absolute_start_time, puma_id, date
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path


# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 32000          # Target sample rate in Hz
WINDOW_SEC = 5               # Window duration in seconds
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC  # 160,000 samples per window
MAJORITY_THRESHOLD = 0.5     # Label must cover >50% of window (i.e., >2.5 s)


# ─── Filename Parsing ─────────────────────────────────────────────────────────
# Expected pattern: puma-[PumaID]_audio_[YYMMDD]-[HHMMSS].wav
FILENAME_RE = re.compile(
    r"puma-(?P<puma_id>[A-Za-z0-9]+)_audio_(?P<date>\d{6})-(?P<time>\d{6})\.wav"
)


def parse_filename(filename: str) -> dict:
    """
    Extract metadata from a puma audio filename.

    Returns dict with keys: puma_id, date, file_start_sec
      - file_start_sec is the absolute second-of-day derived from HHMMSS.

    Time-alignment math:
        The HHMMSS encodes when the recorder started.  For example,
        '000238' → 00h 02m 38s → 0*3600 + 2*60 + 38 = 158 seconds after midnight.
        The label table's Begin/End times are also in absolute seconds since midnight,
        so:  label_time_in_file = label_absolute_time - file_start_sec
    """
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    puma_id = m.group("puma_id")
    date_str = m.group("date")       # e.g. '211202'
    time_str = m.group("time")       # e.g. '000238'

    # Convert HHMMSS → total seconds since midnight
    hh, mm, ss = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
    file_start_sec = hh * 3600 + mm * 60 + ss

    return {
        "puma_id": puma_id,
        "date": date_str,
        "file_start_sec": file_start_sec,
    }


# ─── Label Table Loading ──────────────────────────────────────────────────────

def load_label_table(table_path: str) -> pd.DataFrame:
    """
    Load a Raven selection table, deduplicate, and filter out 'uncertain' labels.

    Deduplication: Raven exports both a 'Waveform' and a 'Spectrogram' row for
    each selection with identical time bounds.  We keep only Spectrogram rows
    (which include Avg Power Density) — but the key point is to avoid counting
    each event twice.  We then drop exact duplicates on the time/annotation
    columns as an extra safeguard.

    Filtering: Any row whose Annotation is 'uncertain' is discarded entirely.
    """
    df = pd.read_csv(table_path, sep="\t")

    # Keep only Spectrogram rows to remove Waveform duplicates
    df = df[df["View"] == "Spectrogram 1"].copy()

    # Further deduplicate on the columns that matter
    df = df.drop_duplicates(subset=["Begin Time (s)", "End Time (s)", "Annotation"])

    # Remove uncertain labels
    df = df[df["Annotation"].str.strip().str.lower() != "uncertain"]

    # Keep only the columns we need and sort by start time
    df = df[["Begin Time (s)", "End Time (s)", "Annotation"]].copy()
    df = df.rename(columns={
        "Begin Time (s)": "begin",
        "End Time (s)": "end",
        "Annotation": "label",
    })
    df["label"] = df["label"].str.strip().str.lower()
    df = df.sort_values("begin").reset_index(drop=True)

    return df


# ─── Label Assignment ─────────────────────────────────────────────────────────

def assign_label(win_start: float, win_end: float, labels_df: pd.DataFrame) -> str | None:
    """
    Determine the label for a 5-second window.

    Rules (applied in order):
      1. Collect every labeled interval that overlaps this window and sum
         coverage per distinct label.
      2. If more than one distinct label has any overlap, skip the window
         (return None).  This ensures every kept segment is unambiguously
         one behavior.
      3. If exactly one label overlaps but covers ≤50% of the window,
         skip it (too little signal).
      4. Otherwise assign that single label.

    Returns the label string, or None if the window should be skipped.
    """
    coverage = {}  # label → total seconds of overlap within this window

    for _, row in labels_df.iterrows():
        # Overlap = max(0, min(ends) - max(starts))
        overlap_start = max(win_start, row["begin"])
        overlap_end = min(win_end, row["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > 0:
            coverage[row["label"]] = coverage.get(row["label"], 0.0) + overlap

    if not coverage:
        return None

    # Skip windows where more than one distinct label is present
    if len(coverage) > 1:
        return None

    # Exactly one label — check the majority threshold
    label, secs = next(iter(coverage.items()))
    if secs > MAJORITY_THRESHOLD * WINDOW_SEC:
        return label

    return None


# ─── Main Processing ──────────────────────────────────────────────────────────

def process_file(
    audio_path: str,
    labels_df: pd.DataFrame,
    output_dir: str,
) -> list[dict]:
    """
    Process a single audio file against its label table.

    Steps:
      1. Parse the filename to get puma_id, date, and the recorder's start
         time in absolute seconds since midnight (file_start_sec).
      2. Load audio at 32 kHz mono.
      3. Walk through non-overlapping 5-second windows.
      4. For each window, compute its absolute time range:
             abs_start = file_start_sec + (window_index * 5)
             abs_end   = abs_start + 5
         and look up whether a label covers >50% of that range.
      5. If yes, write the segment and record its metadata.

    Returns a list of metadata dicts (one per saved segment).
    """
    filename = os.path.basename(audio_path)
    meta = parse_filename(filename)
    puma_id = meta["puma_id"]
    date_str = meta["date"]
    file_start_sec = meta["file_start_sec"]

    # Load audio — librosa resamples to sr and mixes to mono automatically
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    total_samples = len(y)
    total_duration = total_samples / sr  # in seconds

    num_windows = int(total_duration // WINDOW_SEC)  # full 5-s windows only

    rows = []
    for i in range(num_windows):
        # ── Time math ──────────────────────────────────────────────────
        # start_in_file: how far into the .wav file this window begins
        start_in_file = i * WINDOW_SEC  # seconds

        # absolute_start: the label-table-aligned clock time (seconds since midnight)
        # The label table uses absolute times, so we shift by the recorder offset.
        absolute_start = file_start_sec + start_in_file
        absolute_end = absolute_start + WINDOW_SEC
        # ───────────────────────────────────────────────────────────────

        label = assign_label(absolute_start, absolute_end, labels_df)
        if label is None:
            continue  # unlabeled gap or no majority — skip

        # Extract the audio samples for this window
        sample_start = i * WINDOW_SAMPLES
        sample_end = sample_start + WINDOW_SAMPLES
        segment = y[sample_start:sample_end]

        # Build output filename:  [PumaID]_[Date]_[AbsoluteStartSecond].wav
        # AbsoluteStartSecond is an integer for a clean filename
        out_name = f"{puma_id}_{date_str}_{int(absolute_start)}.wav"
        out_path = os.path.join(output_dir, out_name)

        # Write segment at 32 kHz mono
        sf.write(out_path, segment, samplerate=SAMPLE_RATE)

        rows.append({
            "window_filename": out_name,
            "label": label,
            "source_file": filename,
            "start_time_in_source": round(start_in_file, 6),
            "absolute_start_time": round(absolute_start, 6),
            "puma_id": puma_id,
            "date": date_str,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Segment puma audio into labeled 5-second windows."
    )
    parser.add_argument(
        "audio_dir",
        help="Directory containing puma .wav files.",
    )
    parser.add_argument(
        "table_dir",
        help="Directory containing label table .txt files. "
             "Tables are matched to audio by [PumaID]_[YYMMDD].",
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory for segments and CSV (default: ./output).",
    )
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Discover audio files
    audio_files = sorted(
        f for f in os.listdir(args.audio_dir) if f.lower().endswith(".wav")
    )
    if not audio_files:
        print(f"No .wav files found in {args.audio_dir}")
        return

    all_metadata: list[dict] = []

    for audio_file in audio_files:
        audio_path = os.path.join(args.audio_dir, audio_file)
        meta = parse_filename(audio_file)

        # Match label table by PumaID + Date
        # Expected table name pattern: [PumaID]_[YYMMDD]_table.txt
        table_name = f"{meta['puma_id']}_{meta['date']}_table.txt"
        table_path = os.path.join(args.table_dir, table_name)

        if not os.path.isfile(table_path):
            print(f"WARNING: No label table found for {audio_file} "
                  f"(expected {table_path}). Skipping.")
            continue

        print(f"Processing {audio_file} with {table_name} ...")
        labels_df = load_label_table(table_path)
        print(f"  Loaded {len(labels_df)} label intervals "
              f"(after dedup + uncertain removal)")

        rows = process_file(audio_path, labels_df, output_dir)
        all_metadata.extend(rows)
        print(f"  Saved {len(rows)} segments")

    # Write master CSV
    if all_metadata:
        csv_path = os.path.join(output_dir, "master_metadata.csv")
        df_out = pd.DataFrame(all_metadata)
        df_out.to_csv(csv_path, index=False)
        print(f"\nDone. Wrote {len(df_out)} rows to {csv_path}")
    else:
        print("\nNo segments produced.")


if __name__ == "__main__":
    main()