# src/data_processing

Scripts that transform raw MELD and IEMOCAP downloads into the final benchmark CSVs consumed by all downstream steps.

Run the scripts in the order listed below.

## Sub-packages

```
data_processing/
├── syllable_nuclei.py      # Speech-rate estimation (shared by both datasets)
├── meld/
│   ├── init_meld_dataset.py           # Step 1: parse raw CSVs
│   ├── convert_meld_mp4_to_wav.py     # Step 2: extract WAV audio
│   ├── add_audio_features_meld.py     # Step 3: compute acoustic features
│   └── extend_meld_categories.py      # Step 4: add categorical columns
└── iemocap/
    ├── init_iemocap_dataset.py        # Step 1: parse annotation files
    ├── add_audio_features_iemocap.py  # Step 2: compute acoustic features
    └── extend_iemocap_categories.py   # Step 3: add categorical columns
```

## Processing steps

### MELD

**Step 1 — Parse raw CSVs**

```bash
python -m src.data_processing.meld.init_meld_dataset \
    [--root /path/to/MELD.Raw] \
    [--out  data/benchmark/meld/meld_erc_init.csv]
```

Reads `train_sent_emo.csv`, `dev_sent_emo.csv`, and `test_sent_emo.csv` from the MELD root, infers speaker gender from a built-in character list, cleans Unicode artefacts in the utterance text, and adds `dialog_idx` / `turn_idx` identifiers.

**Step 2 — Convert MP4 to WAV** (requires `ffmpeg`)

```bash
python -m src.data_processing.meld.convert_meld_mp4_to_wav \
    [--root /path/to/MELD.Raw] [--overwrite]
```

Converts every `.mp4` under `train_splits/`, `dev_splits/`, and `test_splits/` to a mono 16 kHz WAV written to `audio/train_audio_splits/` etc.

**Step 3 — Extract acoustic features**

```bash
python -m src.data_processing.meld.add_audio_features_meld \
    [--csv  data/benchmark/meld/meld_erc_init.csv] \
    [--root /path/to/MELD.Raw] \
    [--out  data/benchmark/meld/meld_erc_with_audio.csv] \
    [--limit N]   # optional: process only first N rows
```

Appends seven columns: `intensity_mean_db`, `intensity_std_db`, `pitch_mean_hz`, `pitch_std_hz`, `pitch_range_hz`, `articulation_rate_syll_per_s`, `hnr_mean_db`.

**Step 4 — Categorise features**

```bash
python -m src.data_processing.meld.extend_meld_categories \
    [--csv_in  data/benchmark/meld/meld_erc_with_audio.csv] \
    [--csv_out data/benchmark/meld/meld_erc_final.csv] \
    [--categories 3]   # 2, 3, or 5 bins
```

Fits quantile thresholds on the training split and adds `intensity_level`, `pitch_level` (gender-stratified), `rate_level`, `mapped_emotion`, and `idx`.

---

### IEMOCAP

**Step 1 — Parse annotation files**

```bash
python -m src.data_processing.iemocap.init_iemocap_dataset \
    [--root /path/to/IEMOCAP_full_release] \
    [--out  data/benchmark/iemocap/iemocap_erc_init.csv]
```

Reads `EmoEvaluation/*.txt` and `transcriptions/*.txt` from all five sessions. Assigns train/dev (Sessions 1–4) and test (Session 5) splits. Adds `erc_target` flag for the six primary emotions.

**Step 2 — Extract acoustic features**

```bash
python -m src.data_processing.iemocap.add_audio_features_iemocap \
    [--csv  data/benchmark/iemocap/iemocap_erc_init.csv] \
    [--root /path/to/IEMOCAP_full_release] \
    [--out  data/benchmark/iemocap/iemocap_erc_with_audio.csv]
```

**Step 3 — Categorise features**

```bash
python -m src.data_processing.iemocap.extend_iemocap_categories \
    [--csv_in  data/benchmark/iemocap/iemocap_erc_with_audio.csv] \
    [--csv_out data/benchmark/iemocap/iemocap_erc_final.csv] \
    [--categories 3]
```

---

## Output columns (final CSVs)

Both final CSVs share these key columns after all steps:

| Column | Description |
|--------|-------------|
| `split` | `train` / `dev` / `test` |
| `dialog_idx` | Global zero-based dialogue index |
| `turn_idx` | Zero-based turn index within dialogue |
| `speaker` | Speaker identifier |
| `gender` | `M` / `F` / `U` |
| `emotion` | Raw emotion label from the dataset |
| `mapped_emotion` | Unified canonical label (e.g. `"joyful"`, `"angry"`) |
| `utterance` | Utterance text |
| `intensity_level` | `low` / `medium` / `high` |
| `pitch_level` | `low` / `medium` / `high` (gender-stratified) |
| `rate_level` | `low` / `medium` / `high` |
| `idx` | Composite key (`"m_<dialog_idx>_<turn_idx>"` for MELD, `"i_…"` for IEMOCAP) |
