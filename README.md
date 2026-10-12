# SignalCheck

**A reproducible fake news classification project using TF-IDF and RoBERTa.**

Maintained by **Mohammed Shoaib** · [GitHub profile](https://github.com/shoebtech98)

[Quick start](#quick-start) · [Use WELFake](#use-the-welfake-dataset) · [Results](#results) · [Project layout](#project-layout)

> [!IMPORTANT]
> The bundled articles are **fictional test data**. They let you run every notebook without a dataset download or GPU, but their scores do **not** measure real-world accuracy.

## Overview

SignalCheck compares a TF-IDF and logistic regression baseline with two RoBERTa workflows. Shared code in `src/pipeline.py` handles loading, stratified splitting, training, evaluation, and output files.

| Notebook | Workflow | Full-data input |
| --- | --- | --- |
| [01 · TF-IDF baseline](notebooks/01_tfidf_baseline.ipynb) | Fit TF-IDF on training data only; evaluate logistic regression | Prepared WELFake CSV |
| [02 · RoBERTa, exact deduplication](notebooks/02_roberta_finetune.ipynb) | Remove exact duplicates; fine-tune RoBERTa | Raw WELFake CSV |
| [03 · RoBERTa, prepared data](notebooks/03_roberta_finetune_deduped.ipynb) | Fine-tune RoBERTa after semantic deduplication | Prepared WELFake CSV |

Each notebook starts in **sample mode**. In that mode, RoBERTa is a tiny randomly initialized network trained for one CPU epoch. Full mode uses pretrained `roberta-base` and requires a CUDA GPU.

## Quick start

Tested locally with **Python 3.13**. From the repository root:

```powershell
python -m pip install -r requirements.txt
python tools/run_notebook.py
```

The runner executes every code cell and saves visible outputs back into the three notebooks. New metrics go to `results/sample/`; locally generated sample models go to `models/sample/` and are ignored by Git.

To explore one notebook interactively, open it in VS Code with the Jupyter extension and select **Run All**. Keep `DATA_MODE = "sample"` for the first run.

## Use the WELFake dataset

1. Download [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place `WELFake_Dataset.csv` at `data/raw/WELFake_Dataset.csv`. The CSV is ignored by Git.
2. Prepare the semantically deduplicated input for notebooks 01 and 03:

   ```powershell
   python -m pip install -r requirements-prep.txt
   python tools/prepare_data.py --method semantic
   ```

   This writes `data/processed/final_dataset.csv`. It combines title and text, removes exact duplicates, embeds articles, and groups near duplicates using a cosine-similarity threshold of `0.98`. Processing the complete dataset can take a long time.
3. Change `DATA_MODE` in each notebook:

   | Notebook | Set `DATA_MODE` to |
   | --- | --- |
   | 01 · TF-IDF baseline | `"processed"` |
   | 02 · RoBERTa, exact deduplication | `"raw"` |
   | 03 · RoBERTa, prepared data | `"processed"` |

4. Run the notebooks on a machine with a CUDA GPU. Full RoBERTa runs download `roberta-base`; the code reports an error if CUDA is unavailable. Full outputs go to `results/full/` and `models/full/`.

The data loader removes duplicate text before splitting. The split function checks that train, validation, and test text do not overlap. Different data, hardware, or package versions may produce different metrics.

## Results

| Output | Location | How to interpret it |
| --- | --- | --- |
| Local sample run | [`results/sample/`](results/sample/) | Confirms that the code executes; **not** a real accuracy benchmark |
| New WELFake run | `results/full/` | Created only after you run full mode |
| Supplied historical files | [`results/`](results/) | Predate this adaptation and were not reproduced during the audit |

Each new experiment writes `metrics.json`, `classification_report.txt`, and `predictions.csv`. The repository includes saved sample outputs for the [TF-IDF baseline](results/sample/tfidf_baseline/metrics.json), [RoBERTa exact workflow](results/sample/roberta_exact/metrics.json), and [RoBERTa prepared workflow](results/sample/roberta_prepared/metrics.json).

To run the automated data checks:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Predict with a saved model

After a **full** run of notebook 03:

```powershell
python app/predict.py "Paste an article here"
```

To inspect a locally generated **sample** model instead, first run the notebooks, then use the explicit sample flag:

```powershell
python app/predict.py "An invented sample headline" --model models/sample/roberta_exact --allow-sample
```

A classification score is not a fact check. Evaluate a trained model on current, independently labeled articles before using its predictions for decisions.

## Project layout

```text
app/predict.py             Command-line prediction tool
data/sample_news.csv        Fictional data for the offline smoke test
data/raw/                  WELFake download location (ignored by Git)
data/processed/            Prepared dataset location (ignored by Git)
notebooks/                 Three executable experiments
src/pipeline.py            Shared data, training, and evaluation code
tools/prepare_data.py      Exact and semantic deduplication
tools/run_notebook.py      Execute and save all notebook cells
results/                   Supplied historical and new run outputs
models/                    Locally generated model files
```

## License and provenance

This adaptation is maintained by Mohammed Shoaib. The original MIT copyright and permission notice is preserved in [LICENSE](LICENSE). Supplied historical metrics and predictions predate this adaptation; new runs are stored separately to keep their provenance clear.
