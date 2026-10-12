"""Prepare WELFake for the processed-data notebooks.

Semantic mode embeds articles and removes connected near-duplicate groups before
training/test splitting. It can take a long time on the full WELFake dataset.
"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def prepare(input_path: Path, output_path: Path, *, method: str, threshold: float = 0.98):
    frame = pd.read_csv(input_path)
    required = {"title", "text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input needs columns: {sorted(required)}")
    frame = frame.dropna(subset=["label"]).copy()
    frame["combined"] = (frame["title"].fillna("").astype(str) + " " +
                         frame["text"].fillna("").astype(str)).str.strip()
    frame = frame.loc[frame["combined"].ne("")].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")
    frame = frame.drop_duplicates(subset=["combined"]).reset_index(drop=True)
    exact_count = len(frame)

    if method == "semantic":
        from sentence_transformers import SentenceTransformer
        from sklearn.neighbors import NearestNeighbors

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.encode(
            frame["combined"].tolist(), batch_size=64, normalize_embeddings=True,
            show_progress_bar=True,
        )
        neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        neighbors.fit(vectors)
        groups = neighbors.radius_neighbors(vectors, radius=1 - threshold, return_distance=False)
        parent = list(range(len(frame)))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for index, matches in enumerate(groups):
            for match in matches:
                left, right = root(index), root(int(match))
                if left != right:
                    parent[max(left, right)] = min(left, right)
        keep = sorted({root(index) for index in range(len(frame))})
        frame = frame.iloc[keep].reset_index(drop=True)
    elif method != "exact":
        raise ValueError("method must be exact or semantic")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Exact dedup: {exact_count:,} rows; final: {len(frame):,} rows")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/WELFake_Dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/final_dataset.csv")
    parser.add_argument("--method", choices=["exact", "semantic"], default="semantic")
    parser.add_argument("--threshold", type=float, default=0.98)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Missing input: {args.input}")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    prepare(args.input, args.output, method=args.method, threshold=args.threshold)
