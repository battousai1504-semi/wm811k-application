# WM-811K Application

Utilities and scripts for exploring the WM-811K wafer map dataset, extracting
handcrafted defect features, and training classical machine-learning baselines.

## Project Layout

```text
src/wm811k/                         Shared reusable project code
scripts/Phase_1-Dataset_explaration Exploratory dataset scripts
scripts/Phase_2-Preprocessing_on_wm811k
                                    Feature extraction and analysis scripts
scripts/Phase_3_Machine_learning    ML data preparation and baseline training
data/raw/LSWMD.pkl                  Raw dataset, ignored by git
data/processed/wafer_features.csv   Main feature dataset used by Phase 3
models/                             Saved preprocessing objects and models
results/                            Evaluation outputs
outputs/                            Visualizations and exploratory outputs
```

## Setup

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Main Pipeline

Extract handcrafted features:

```bash
python scripts/Phase_2-Preprocessing_on_wm811k/02_batch_features.py
```

Prepare train, validation, and test splits:

```bash
python scripts/Phase_3_Machine_learning/01_prepare_ml_data.py
```

Train baseline models:

```bash
python scripts/Phase_3_Machine_learning/02_train_baselines.py
```

Run feature analysis:

```bash
python scripts/Phase_2-Preprocessing_on_wm811k/05_feature_analysis.py
```

## Notes

- Morphology cleaning has been removed from the project pipeline.
- `data/processed/wafer_features.csv` is the main feature source used by Phase 3.
- Generated data, models, and outputs are ignored by git.

