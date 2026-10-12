"""Reproducible training helpers for the project notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "data" / "sample_news.csv"
RAW_DATA = ROOT / "data" / "raw" / "WELFake_Dataset.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "final_dataset.csv"


def load_data(kind: str = "sample") -> pd.DataFrame:
    """Load a dataset and standardize it to combined text and binary labels."""
    paths = {"sample": SAMPLE_DATA, "raw": RAW_DATA, "processed": PROCESSED_DATA}
    if kind not in paths:
        raise ValueError(f"kind must be one of {tuple(paths)}")
    path = paths[kind]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. See README.md for data setup.")
    frame = pd.read_csv(path)
    if "label" not in frame:
        raise ValueError(f"{path} needs a label column")
    if "combined" not in frame:
        if not {"title", "text"}.issubset(frame.columns):
            raise ValueError(f"{path} needs combined, or title and text columns")
        frame["combined"] = frame["title"].fillna("").astype(str) + " " + frame["text"].fillna("").astype(str)
    frame = frame.dropna(subset=["label"]).copy()
    frame["combined"] = frame["combined"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["combined"].ne("")].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 (fake) or 1 (real)")
    frame = frame.drop_duplicates(subset=["combined"]).reset_index(drop=True)
    if frame["label"].nunique() != 2:
        raise ValueError("Both labels are required")
    return frame


def split_data(frame: pd.DataFrame, *, sample: bool = False):
    """Stratified train, validation, and test split with overlap checks."""
    fraction = 0.2 if sample else 0.15
    train, remainder = train_test_split(
        frame, test_size=2 * fraction, stratify=frame["label"], random_state=42
    )
    valid, test = train_test_split(
        remainder, test_size=0.5, stratify=remainder["label"], random_state=42
    )
    texts = [set(part["combined"]) for part in (train, valid, test)]
    if any(texts[i] & texts[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Text overlap between splits")
    return tuple(part.reset_index(drop=True) for part in (train, valid, test))


def evaluate(true, predicted) -> dict:
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "classification_report": classification_report(
            true, predicted, labels=[0, 1], target_names=["fake", "real"], zero_division=0
        ),
        "confusion_matrix": confusion_matrix(true, predicted, labels=[0, 1]).tolist(),
    }


def save_results(name: str, scope: str, true, predicted, probabilities=None) -> Path:
    """Keep sample runs separate from the supplied historical results."""
    directory = ROOT / "results" / scope / name
    directory.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(true, predicted)
    (directory / "metrics.json").write_text(
        json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2),
        encoding="utf-8",
    )
    (directory / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")
    data = {"true_label": list(true), "predicted_label": list(predicted)}
    if probabilities is not None:
        data["prob_fake"] = np.asarray(probabilities)[:, 0]
        data["prob_real"] = np.asarray(probabilities)[:, 1]
    pd.DataFrame(data).to_csv(directory / "predictions.csv", index=False)
    return directory


def train_baseline(train: pd.DataFrame):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
    ])
    model.fit(train["combined"], train["label"])
    return model


def _sample_tokenizer(texts):
    """Tiny local tokenizer for offline execution checks; never a pretrained model."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = sorted({word.lower() for text in texts for word in re.findall(r"\b\w+\b|[^\w\s]", text)})
    vocab = {token: index for index, token in enumerate(["<pad>", "<unk>"] + words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>", model_max_length=64
    )


def train_roberta(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, *, sample: bool):
    """Fine-tune RoBERTa; sample mode uses a small randomly initialized network."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(42)
    if sample:
        tokenizer = _sample_tokenizer(train["combined"])
        config = RobertaConfig(
            vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=64, max_position_embeddings=66,
            pad_token_id=tokenizer.pad_token_id, num_labels=2,
        )
        model = RobertaForSequenceClassification(config)
        epochs, batch_size, max_length = 1, 4, 64
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Full RoBERTa training needs a CUDA GPU. Use sample mode for a CPU smoke run.")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        epochs, batch_size, max_length = 3, 4, 256
        device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5 if not sample else 1e-3)

    def batches(frame, shuffle=False):
        indices = torch.randperm(len(frame)).tolist() if shuffle else list(range(len(frame)))
        for offset in range(0, len(indices), batch_size):
            batch = frame.iloc[indices[offset:offset + batch_size]]
            encoded = tokenizer(
                batch["combined"].tolist(), padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            yield {key: value.to(device) for key, value in encoded.items()}, torch.tensor(
                batch["label"].tolist(), dtype=torch.long, device=device
            )

    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for encoded, labels in batches(train, shuffle=True):
            optimizer.zero_grad()
            loss = model(**encoded, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        valid_predictions, _ = predict_roberta(model, tokenizer, valid, device, max_length, batch_size)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_macro_f1": evaluate(valid["label"], valid_predictions)["macro_f1"]})

    predictions, probabilities = predict_roberta(model, tokenizer, test, device, max_length, batch_size)
    return model, tokenizer, history, predictions, probabilities


def predict_roberta(model, tokenizer, frame, device, max_length, batch_size):
    import torch

    model.eval()
    scores = []
    with torch.inference_mode():
        for offset in range(0, len(frame), batch_size):
            batch = frame.iloc[offset:offset + batch_size]
            encoded = tokenizer(batch["combined"].tolist(), padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            scores.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probabilities = np.concatenate(scores)
    return probabilities.argmax(axis=1), probabilities
