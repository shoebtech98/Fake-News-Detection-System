"""Build the three editable notebooks from concise, shared workflows."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return nbf.v4.new_markdown_cell(source)


def code(source):
    return nbf.v4.new_code_cell(source)


SETUP = '''from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
if not (ROOT / "src").is_dir():
    raise RuntimeError("Open this notebook from the project root or notebooks/ directory")
sys.path.insert(0, str(ROOT))'''


def build_baseline():
    cells = [
        markdown("# SignalCheck · TF-IDF baseline\n\nA reproducible text classifier. **Sample mode** runs immediately on fictional data and checks the pipeline. Switch to `processed` after preparing WELFake to train on real data. Historical metrics in the root `results/` folder are separate from this run."),
        code(SETUP + '\nfrom src.pipeline import load_data, split_data, train_baseline, evaluate, save_results'),
        markdown("## 1 · Choose data\n\nThe bundled sample is synthetic and must not be used to claim real-world accuracy. For the full run, place the prepared CSV at `data/processed/final_dataset.csv`."),
        code('DATA_MODE = "sample"  # change to "processed" for the full dataset\nframe = load_data(DATA_MODE)\nprint(f"Rows: {len(frame):,}; labels: {frame.label.value_counts().to_dict()}")\ndisplay(frame[["combined", "label"]].head(3))'),
        markdown("## 2 · Split and fit\n\nText is split before the vectorizer is fitted, preventing vocabulary leakage."),
        code('train, valid, test = split_data(frame, sample=DATA_MODE == "sample")\nprint("Train / validation / test:", len(train), len(valid), len(test))\nmodel = train_baseline(train)'),
        markdown("## 3 · Evaluate\n\nValidation helps inspect the model. The held-out test split is evaluated once here."),
        code('validation_pred = model.predict(valid["combined"])\nprint("Validation:", evaluate(valid["label"], validation_pred)["macro_f1"])\ntest_pred = model.predict(test["combined"])\nmetrics = evaluate(test["label"], test_pred)\nprint(metrics["classification_report"])\nprint("Confusion matrix [fake, real]:", metrics["confusion_matrix"])'),
        markdown("## 4 · Save this run\n\nNew outputs go under `results/sample/` or `results/full/`; supplied historical files stay intact."),
        code('scope = "sample" if DATA_MODE == "sample" else "full"\noutput_dir = save_results("tfidf_baseline", scope, test["label"], test_pred, model.predict_proba(test["combined"]))\nprint("Saved to:", output_dir)'),
    ]
    return cells


def build_roberta(*, deduped):
    title = "prepared data" if deduped else "exact deduplication"
    full_mode = "processed" if deduped else "raw"
    name = "roberta_prepared" if deduped else "roberta_exact"
    description = (
        "For a full run, prepare `data/processed/final_dataset.csv` with the script described in the README."
        if deduped else
        "For a full run, place WELFake at `data/raw/WELFake_Dataset.csv`. Exact duplicates are removed during loading."
    )
    return [
        markdown(f"# SignalCheck · RoBERTa on {title}\n\nThis notebook trains and evaluates a classifier. **Sample mode** uses a tiny randomly initialized RoBERTa network to test every step offline. It is not a valid news detector. Full mode fine-tunes pretrained `roberta-base` and requires a CUDA GPU and model download."),
        code(SETUP + '\nfrom src.pipeline import load_data, split_data, train_roberta, evaluate, save_results'),
        markdown(f"## 1 · Choose data\n\n{description}"),
        code(f'DATA_MODE = "sample"  # change to "{full_mode}" for the full dataset\nframe = load_data(DATA_MODE)\nprint(f"Rows: {{len(frame):,}}; labels: {{frame.label.value_counts().to_dict()}}")\ndisplay(frame[["combined", "label"]].head(3))'),
        markdown("## 2 · Stratified split\n\nThe text overlap check in `split_data` raises an error if a duplicate crosses a split."),
        code('train, valid, test = split_data(frame, sample=DATA_MODE == "sample")\nprint("Train / validation / test:", len(train), len(valid), len(test))'),
        markdown("## 3 · Fine-tune\n\nSample mode is a CPU smoke test. Full mode uses three epochs of pretrained RoBERTa on CUDA; training time depends on the GPU."),
        code('model, tokenizer, history, test_pred, probabilities = train_roberta(\n    train, valid, test, sample=DATA_MODE == "sample"\n)\nprint("Training history:", history)'),
        markdown("## 4 · Evaluate and save\n\nThe test metrics are from this run. Synthetic sample scores do not represent real news performance."),
        code(f'metrics = evaluate(test["label"], test_pred)\nprint(metrics["classification_report"])\nprint("Confusion matrix [fake, real]:", metrics["confusion_matrix"])\nscope = "sample" if DATA_MODE == "sample" else "full"\noutput_dir = save_results("{name}", scope, test["label"], test_pred, probabilities)\nmodel_dir = ROOT / "models" / scope / "{name}"\nmodel.save_pretrained(model_dir)\ntokenizer.save_pretrained(model_dir)\nprint("Results:", output_dir)\nprint("Model:", model_dir)'),
    ]


def main():
    notebooks = {
        "01_tfidf_baseline.ipynb": build_baseline(),
        "02_roberta_finetune.ipynb": build_roberta(deduped=False),
        "03_roberta_finetune_deduped.ipynb": build_roberta(deduped=True),
    }
    for filename, cells in notebooks.items():
        notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}})
        nbf.write(notebook, ROOT / "notebooks" / filename)
        print("Built", filename)


if __name__ == "__main__":
    main()
