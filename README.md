# EuroSAT CNN classification

SqueezeNet 1.1 is being converted from `SqueezeNet_EuroSAT.ipynb` into a standalone
`SqueezeNet_EuroSAT.py` script in ten planned commits. The first commit establishes
the shared data contract and automation; the next nine arrive progressively.

## Shared splits for every model

`eurosat_classes.json` defines the label order. `eurosat_splits.csv` preserves the
original row order and assigns all 27,000 RGB images to 18,900 training, 4,050
validation, and 4,050 test examples. These two supplied files are copied unchanged.
Use `eurosat_data.load_splits()` in future models: it checks counts, duplicate paths,
labels, class folders, and class coverage. Never regenerate or randomly split these
files when comparing models. Select checkpoints using validation data only.

## Commit pipeline

| Commit | Change |
| --- | --- |
| 1 | Shared split files, validation helper, dependencies, and timed pipeline |
| 2 | Standalone Python entry point and reproducible setup |
| 3 | Load and validate the shared split manifest |
| 4 | Dataset resolution, preprocessing, and data loaders |
| 5 | Pretrained SqueezeNet and replacement classifier |
| 6 | Epoch loops, validation, history, and best checkpoint |
| 7 | Five-epoch classifier training |
| 8 | Twenty-epoch fine-tuning |
| 9 | Training curves, test metrics, and predictions |
| 10 | Confusion matrix, comparison summaries, and usage guide |

GitHub Actions checks approximately every ten minutes and publishes at most one
queued change once two hours have elapsed since the previous planned commit.
The first automatic commit is due two hours after commit 1. Normal completion is
about 18 hours later, plus check intervals and GitHub scheduling delays. Scheduled
runs are best effort, so this is not an exact wall-clock timer.

The nine pending changes are stored in the `eurosat-commit-queue` Actions artifact
with 30-day retention, outside Git commit history. Repository readers can download
artifacts when signed in. The committed plan pins the queue's SHA-256 checksum.
Each run verifies the expected previous file contents, commits only the planned
paths and progress file, and pushes normally to `main`. Conflicts fail for review;
the workflow never force-pushes or includes unrelated edits. Avoid editing the
queued Python script or this README until the sequence finishes.

The workflow disables itself after commit 10. To pause sooner, disable **EuroSAT
timed commits** in the repository's Actions tab; re-enable to resume. Manual runs
with an empty setup payload respect the two-hour gate. A delayed run makes one
commit, not a burst of missed commits.

Setup uploads a gzip/base64 payload through `workflow_dispatch`; the workflow
validates it before storing it. If the artifact expires or is deleted, re-upload
the same payload. A local backup is kept under ignored `.automation-local/` on
the setup computer. The final script is also available there as a preview while
the commits are rolling out. Don't commit the backup directory.

The workflow needs Actions enabled and permission for `GITHUB_TOKEN` to write
repository contents and disable its own workflow. Branch rules must allow its
normal pushes. See [GitHub's schedule documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

## Development checks

Run `python -m unittest discover -s tests -v`. These checks validate split integrity
and exercise the commit pipeline in disposable repositories without training or
downloading the dataset.
