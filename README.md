# EuroSAT CNN classification

Standalone SqueezeNet 1.1 training and evaluation, converted from
`SqueezeNet_EuroSAT.ipynb`, with fixed splits shared across future models.

## Shared splits for every model

`eurosat_classes.json` defines the label order. `eurosat_splits.csv` preserves the
original row order and assigns all 27,000 RGB images to 18,900 training, 4,050
validation, and 4,050 test examples. These two supplied files are copied unchanged.
Use `eurosat_data.load_splits()` in future models: it checks counts, duplicate paths,
labels, class folders, and class coverage. Never regenerate or randomly split these
files when comparing models. Select checkpoints using validation data only.

## Development checks

Run `python -m unittest discover -s tests -v`. These checks validate split integrity
without training or downloading the dataset.

## Run SqueezeNet

Use Python 3.11 or 3.12 in a virtual environment, then install the dependencies:

```sh
python -m pip install -r requirements.txt
python SqueezeNet_EuroSAT.py --data-dir /path/to/EuroSAT
```

The data directory must contain `AnnualCrop/`, `Forest/`, and the other class
folders referenced by the manifest. Omit `--data-dir` to download
`apollo2506/eurosat-dataset` with KaggleHub. Pretrained ImageNet weights are also
downloaded on first use. Use `python SqueezeNet_EuroSAT.py --help` for options.
The entry-point guard and module-level dataset class support Windows workers.
Use `--num-workers 0` if your environment cannot start worker processes.

Defaults preserve the notebook: seed 42, 64x64 RGB images, ImageNet normalization,
no augmentation, batches of 64, and two data-loader workers. SqueezeNet 1.1 starts
with pretrained frozen features and a dropout/1x1-convolution/global-pooling
classifier. Adam trains that head for five epochs at 1e-3. The best validation-loss
checkpoint then initializes 20 fine-tuning epochs, with feature LR 1e-5 and head
LR 1e-4. The lowest validation loss across both stages selects the final checkpoint.
The test split is evaluated once after training. CUDA is selected when available.
Seeds and deterministic cuDNN settings match the notebook; exact results may
still vary across hardware and library versions.

Outputs default to `squeezenet_results/` (ignored by Git):

- Best weights: `squeezenet_best_state.pt`
- Training history and curves: `training_history.csv`, `training_curves.png`
- Test results: `classification_report.csv`, `test_predictions.csv`
- Confusion matrix: `confusion_matrix.csv`, `confusion_matrix.png`
- Comparison summary: `model_summary.csv`, `model_summary.json`
- Split provenance: `class_names.json`, `eurosat_splits.csv`, and input SHA-256
  hashes in the summary

Plots save without a display by default; add `--show-plots` to open plot windows.
Use a fresh `--output-dir` for each experiment. `--head-epochs` and
`--fine-tune-epochs` allow shorter smoke runs, but change the experiment settings.
No notebook output images or reported metrics are copied as fresh training results.
