# Fake News Detection

*Fake news classification using TF-IDF and fine-tuned RoBERTa-base, with LIME explainability and a Gradio demo. Built on WELFake (72k articles).*

---

<p align="center">
  <a href="#results">Results</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#explainability">Explainability</a>
</p>

---

## Results

Both models trained and evaluated on the same sentence-transformer-deduplicated dataset (see [Methodology](#methodology)).

| Model | Accuracy | Macro F1 | Fake Precision | Fake Recall | Real Precision | Real Recall |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 94.76% | 94.71% | 95.10% | 95.32% | 94.34% | 94.07% |
| RoBERTa-base (fine-tuned) | **99.88%** | **99.88%** | **99.92%** | **99.93%** | **99.84%** | **99.84%** |

### Confusion Matrices

**TF-IDF + Logistic Regression**
```
              Predicted Fake   Predicted Real
Actual Fake        4934              242
Actual Real         254             4032
```

**RoBERTa-base**
```
              Predicted Fake   Predicted Real
Actual Fake        5172                4
Actual Real           7             4279
```

> RoBERTa makes 11 errors on 9,462 test samples.

---

## Methodology

### Dataset

**WELFake** — 72,134 labeled news articles (title + text + label). Source: [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).

- Label 0: Fake
- Label 1: Real
- Input: `title + " " + text` concatenated

### Deduplication

Deduplication was treated as an iterative data quality problem, not a one-time step.

**Step 1 — Exact deduplication:** Removed exact duplicate `(combined_text, label)` pairs. Brought dataset from 72,134 → 63,676 articles.

**Step 2 — Near-duplicate discovery:** On inspecting the deduplicated dataset, a large number of articles were found with cosine similarity > 0.98 — same story, slightly reworded. Exact string matching misses these entirely.

**Step 3 — Semantic deduplication:** Used Sentence Transformers to embed all articles and removed near-duplicates above the 0.98 cosine similarity threshold. Final dataset: ~63,100 articles.

Both models were trained on this final semantically-deduplicated dataset to ensure a fair comparison.

### Splits

70% train / 15% validation / 15% test — stratified by label. Zero train-test and train-validation overlap verified explicitly.

### TF-IDF + Logistic Regression

- TF-IDF fitted on training data only (no leakage)
- `max_features=50000`, `ngram_range=(1,2)`
- Logistic Regression with `class_weight='balanced'`

### RoBERTa Fine-tuning

- Base model: `roberta-base`
- Full fine-tune (all encoder layers unfrozen)
- Dynamic padding via `DataCollatorWithPadding`
- 3 epochs, early stopping on validation loss
- Trained on RTX 5060

**Training curve:**

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1 |
|---|---|---|---|---|
| 1 | 0.011323 | 0.010590 | 99.78% | 99.78% |
| 2 | 0.004020 | 0.010326 | 99.86% | 99.86% |
| 3 | 0.000012 | 0.007843 | 99.89% | 99.89% |

---

## Explainability

LIME (Local Interpretable Model-agnostic Explanations) applied to both models, identifying which tokens most influenced each prediction.

- Token-level attribution for TF-IDF + LR
- Token-level attribution for RoBERTa
- Side-by-side comparison of classical vs transformer explanations
- Analysis of model disagreements and misclassified examples

---

## Project Structure

```
Fake-News-Detection/
│
├── notebooks/
│   ├── 01_tfidf_baseline.ipynb           # TF-IDF + LR pipeline
│   ├── 02_roberta_finetune.ipynb         # RoBERTa fine-tune (combined dedup)
│   ├── 03_roberta_finetune_deduped.ipynb # RoBERTa fine-tune (semantic dedup)
│   └── 04_lime_explainability.ipynb      # LIME on both models
│
├── app/
│   ├── app.py                            # Gradio demo
│   └── requirements.txt
│
├── results/
│   ├── confusion_matrix.csv
│   ├── classification_report.txt
│   ├── test_metrics.json
│   └── predictions.csv
│
├── assets/                               # Images for README
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick Start

### Requirements

```bash
pip install -r app/requirements.txt
```

### Run Demo

```bash
python app/app.py
```

Paste any news article. The app returns a prediction, confidence score, and LIME token attributions for both models side by side.

### Notebooks

Run notebooks in order on Kaggle (GPU required for notebook 03).

---

## Model

The fine-tuned RoBERTa model will be hosted on HuggingFace Hub. *(Coming soon)*

---

## Dataset

WELFake dataset via Kaggle: [saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

The `data/` directory is not committed to this repository. See the Kaggle link above.

---
