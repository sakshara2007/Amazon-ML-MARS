import csv
from pathlib import Path

inp = Path(r"team_delivery\train\candidate_pairs.tsv")
out = Path(r"team_delivery\train\candidate_features.tsv")

print("Creating fast candidate_features.tsv...")
print("Input :", inp)
print("Output:", out)

with inp.open("r", encoding="utf-8", newline="") as fin, \
     out.open("w", encoding="utf-8", newline="", buffering=1024*1024) as fout:

    reader = csv.reader(fin, delimiter="\t")
    writer = csv.writer(fout, delimiter="\t", lineterminator="\n")

    header = next(reader)

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_id",
        "exact_name",
        "token_name_hits",
        "token_address_hits",
        "tfidf_name",
        "tfidf_address",
        "tfidf_combined",
        "bm25_name",
        "bm25_address"
    ])

    count = 0

    for row in reader:
        if len(row) < 2:
            continue

        s1 = row[0]
        candidates = row[1]

        if candidates:
            for cid in candidates.split(","):
                if not cid:
                    continue

                writer.writerow([
                    s1,
                    cid,
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0"
                ])

                count += 1

                if count % 1_000_000 == 0:
                    print(f"Written {count:,} feature rows...", flush=True)

print(f"COMPLETE: {count:,} rows")
print(f"Output: {out}")
