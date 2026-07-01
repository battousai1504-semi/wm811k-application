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

By default this extracts features for every labeled WM-811K wafer. To recreate a
smaller stratified development CSV, pass a cap explicitly:

```bash
python scripts/Phase_2-Preprocessing_on_wm811k/02_batch_features.py --max-per-class 500
```

Run feature analysis and save the correlation-filtered feature list:

```bash
python scripts/Phase_2-Preprocessing_on_wm811k/05_feature_analysis.py
```

Train the proposal-style classical CV baseline with selected geometric features,
LBP/GLCM texture features, and SVM RBF 5-fold CV:

```bash
python scripts/Phase_3_Machine_learning/03_train_classical_svm.py
```

Prepare the 64x64 3-channel image dataset for deep learning:

```bash
python scripts/Phase_3_Machine_learning/04_prepare_dl_image_dataset.py
```

Create class weights and sample weights for deep-learning imbalance handling:

```bash
python scripts/Phase_4_Deep_learning/01_handle_class_imbalance.py
```

Train the ResNet-50 deep-learning baseline:

```bash
python scripts/Phase_4_Deep_learning/02_train_resnet50_baseline.py
```

Prepare train, validation, and test splits:

```bash
python scripts/Phase_3_Machine_learning/01_prepare_ml_data.py
```

This uses `outputs/phase2/feature_analysis/05_selected_features.csv` when it
exists, so Phase 3 trains on the features kept after high-correlation removal.
Use `--feature-set default` to force the original handcrafted feature list.

Train baseline models:

```bash
python scripts/Phase_3_Machine_learning/02_train_baselines.py
```

## Notes

- Morphology cleaning has been removed from the project pipeline.
- `data/processed/wafer_features.csv` is the main feature source used by Phase 3.
- Deep-learning training requires PyTorch and torchvision in addition to the base requirements.
- Generated data, models, and outputs are ignored by git.

