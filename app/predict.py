"""Predict with a model produced by notebook 02 or 03."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def predict(text: str, model_dir: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"No saved model at {model_dir}. Run a RoBERTa notebook first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    inputs = tokenizer(text, truncation=True, max_length=min(tokenizer.model_max_length, 256), return_tensors="pt")
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    label = int(probabilities[1] > probabilities[0])
    return label, probabilities[label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a news article with a saved model")
    parser.add_argument("text", help="Article or headline to classify")
    parser.add_argument("--model", type=Path, default=ROOT / "models/full/roberta_prepared")
    parser.add_argument("--allow-sample", action="store_true", help="Allow the synthetic smoke-test model")
    args = parser.parse_args()
    if "sample" in args.model.parts and not args.allow_sample:
        parser.error("The sample model is only a smoke test; add --allow-sample to inspect it")
    label, confidence = predict(args.text, args.model)
    if "sample" in args.model.parts:
        print("SYNTHETIC SAMPLE MODEL — output is not a real news assessment")
    print(f"Prediction: {'real' if label == 1 else 'fake'} (model confidence {confidence:.1%})")
