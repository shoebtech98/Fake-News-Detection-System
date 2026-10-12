# SignalCheck

**Maintained by Mohammed Shoaib** · [GitHub profile](https://github.com/shoebtech98)

SignalCheck is a notebook-based comparison of TF-IDF with logistic regression and RoBERTa. The repository now has a repeatable sample run, separate output folders, a data preparation command, and a small prediction CLI. The bundled sample articles are fictional and exist only to check that the code runs.

## Run every notebook now

From the project folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
python tools/run_notebook.py
```

The command executes every code cell in all three notebooks and saves the outputs in the notebooks. It also writes new result files under `results/sample/` and small test models under `models/sample/`. To run interactively, open `notebooks/` in VS Code with the Jupyter extension and use **Run All**. All notebooks start with `DATA_MODE = "sample"`.

| Notebook | What it checks |
| --- | --- |
| `01_tfidf_baseline.ipynb` | Data split, TF-IDF fit, evaluation, saved results |
| `02_roberta_finetune.ipynb` | RoBERTa training on the sample with the raw-data workflow |
| `03_roberta_finetune_deduped.ipynb` | RoBERTa training on the sample with the prepared-data workflow |

**Sample scores are smoke-test results, not real-world accuracy estimates.** The sample RoBERTa models are randomly initialized and trained for one short CPU epoch.

## Run on WELFake

1. Download [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and put `WELFake_Dataset.csv` in `data/raw/`.
2. For the prepared-data notebooks, install the optional dependency and create the processed CSV:

   ```powershell
   python -m pip install -r requirements-prep.txt
   python tools/prepare_data.py --method semantic
   ```

   The script creates `data/processed/final_dataset.csv` by combining title and text, removing exact duplicates, embedding articles, and grouping pairs above the cosine-similarity threshold. Full semantic preparation can take a long time.
3. Change `DATA_MODE` in notebook 01 and 03 to `"processed"`. Change it in notebook 02 to `"raw"`.
4. Run the notebooks in VS Code or run `python tools/run_notebook.py` again. Full RoBERTa runs download `roberta-base` and require a CUDA GPU. The code stops with a clear message if the GPU is unavailable.

Full runs write to `results/full/` and `models/full/`. Training uses stratified splits and checks for text overlap between them. Package versions, hardware, and the exact processed dataset can affect the metrics.

To run the automated data checks, install `requirements-dev.txt` and run `python -m pytest -q`.

## Inspect results and try a saved model

- `results/sample/<experiment>/metrics.json` contains metrics from a local smoke run.
- `results/sample/<experiment>/classification_report.txt` contains class-level results.
- `results/sample/<experiment>/predictions.csv` contains individual predictions and probabilities.
- The files directly in `results/` are **historical supplied results**. They were not reproduced in this audit.

After a full RoBERTa run:

```powershell
python app/predict.py "Paste an article here"
```

To inspect only the bundled synthetic model, run:

```powershell
python app/predict.py "An invented sample headline" --model models/sample/roberta_exact --allow-sample
```

A classifier prediction is not a fact check. Evaluate on current, independently labeled articles before using it for decisions.

## Project map

```text
data/sample_news.csv             fictional offline smoke-test data
data/raw/                     WELFake input location
data/processed/               prepared input location
src/pipeline.py               shared loading, training, and evaluation code
tools/prepare_data.py         exact and semantic deduplication
tools/run_notebook.py         run and save every notebook cell
notebooks/                    three editable experiments
app/predict.py                local inference with a saved RoBERTa model
results/                      historical and newly generated metrics
models/                       locally generated model files
```

## License and provenance

This adaptation is maintained by Mohammed Shoaib. The original MIT copyright and permission notice remains in [LICENSE](LICENSE). The historical metrics and saved predictions predate this adaptation; new experiments are stored in separate folders so their provenance stays clear.
