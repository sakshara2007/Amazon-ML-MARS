import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from data import read_records, combine_sources, lookup
from text import make_name, make_address, make_combined
from embeddings import Embedder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--embedding-model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = Path(args.train_dir)
    s1 = read_records(d / "train_source1.tsv")
    s2 = read_records(d / "train_source2.tsv")
    s3 = read_records(d / "train_source3.tsv")
    all_df = pd.concat([s1, s2, s3], ignore_index=True)

    records = all_df.to_dict("records")
    embedder = Embedder(args.embedding_model, args.device, args.batch_size)

    for field, fn in [
        ("name", make_name),
        ("address", make_address),
        ("combined", make_combined),
    ]:
        texts = [fn(r) for r in records]
        idx, emb = embedder.encode_unique(texts)
        np.save(out / f"{field}_embeddings.npy", emb)
        pd.DataFrame(
            [{"text": text, "row": row} for text, row in idx.items()]
        ).to_csv(
            out / f"{field}_index.tsv",
            sep="\t",
            index=False
        )

    print("Embedding caches created.")

if __name__ == "__main__":
    main()
