import os
import time

import duckdb
import pandas as pd

from src.normalize import (
    normalize_text,
    remove_legal_suffixes,
    transliterate_text,
    compact_name,
    extract_numbers,
    extract_postal_codes,
)


DB_PATH = r"artifacts\full_train\target_index.duckdb"

SOURCE1_PATH = r"dataset\train\train_source1.tsv"

OUTPUT_PATH = r"team_delivery\train\candidate_pairs.tsv"

TEMP_DIR = r"artifacts\full_train\duckdb_temp"

BATCH_SIZE = 2000

MAX_CANDIDATES = 180


# ============================================================
# Prepare Source 1 batch
# ============================================================

def prepare_source1(df):

    out = df.copy()

    out["source1_entity_id"] = (
        out["entity_id"]
        .astype(str)
        .str.strip()
    )

    out["name_norm"] = (
        out["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    out["name_no_suffix"] = (
        out["business_name"]
        .fillna("")
        .map(remove_legal_suffixes)
    )

    out["address_norm"] = (
        out["business_address"]
        .fillna("")
        .map(normalize_text)
    )

    out["name_tokens"] = (
        out["name_norm"]
        .map(lambda x: x.split())
    )

    out["address_tokens"] = (
        out["address_norm"]
        .map(lambda x: x.split())
    )

    out["name_numbers"] = (
        out["business_name"]
        .fillna("")
        .map(extract_numbers)
    )

    out["address_numbers"] = (
        out["business_address"]
        .fillna("")
        .map(extract_numbers)
    )

    out["postal_tokens"] = (
        out["business_address"]
        .fillna("")
        .map(extract_postal_codes)
    )

    out["name_translit"] = (
        out["business_name"]
        .fillna("")
        .map(transliterate_text)
    )

    out["name_translit_tokens"] = (
        out["name_translit"]
        .map(lambda x: x.split())
    )

    out["name_compact"] = (
        out["business_name"]
        .fillna("")
        .map(compact_name)
    )

    return out


# ============================================================
# Generate candidates for one batch
# ============================================================

def generate_batch(con, batch):

    con.register(
        "s1_batch",
        batch,
    )

    # --------------------------------------------------------
    # Every block produces:
    #
    # source1_entity_id
    # entity_id
    # score contribution
    #
    # These contributions approximate the frozen blocker's
    # evidence weighting.
    # --------------------------------------------------------

    query = """
    WITH block_hits AS (

        -- ================================================
        -- Exact normalized name
        -- ================================================

        SELECT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            4.0 AS block_score
        FROM s1_batch s
        JOIN name_index i
          ON i.key = s.name_norm
        WHERE s.name_norm <> ''


        UNION ALL


        -- ================================================
        -- Name without legal suffix
        -- ================================================

        SELECT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            3.5 AS block_score
        FROM s1_batch s
        JOIN name_no_suffix_index i
          ON i.key = s.name_no_suffix
        WHERE s.name_no_suffix <> ''


        UNION ALL


        -- ================================================
        -- Exact compact name
        -- ================================================

        SELECT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            1.0 AS block_score
        FROM s1_batch s
        JOIN compact_name_index i
          ON i.key = s.name_compact
        WHERE s.name_compact <> ''


        UNION ALL


        -- ================================================
        -- Name token top-30
        -- ================================================

        SELECT DISTINCT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            2.25 AS block_score
        FROM s1_batch s
        CROSS JOIN UNNEST(s.name_tokens) u(token)
        JOIN name_token_top30 i
          ON i.token = u.token


        UNION ALL


        -- ================================================
        -- Address token top-30
        -- ================================================

        SELECT DISTINCT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            1.65 AS block_score
        FROM s1_batch s
        CROSS JOIN UNNEST(s.address_tokens) u(token)
        JOIN address_token_top30 i
          ON i.token = u.token


        UNION ALL


        -- ================================================
        -- Transliteration token top-20
        -- ================================================

        SELECT DISTINCT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            0.35 AS block_score
        FROM s1_batch s
        CROSS JOIN UNNEST(s.name_translit_tokens) u(token)
        JOIN translit_token_top20 i
          ON i.token = u.token


        UNION ALL


        -- ================================================
        -- Number top-20
        -- ================================================

        SELECT DISTINCT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            1.0 AS block_score
        FROM s1_batch s
        CROSS JOIN UNNEST(s.name_numbers) u(token)
        JOIN number_top20 i
          ON i.token = u.token


        UNION ALL


        SELECT DISTINCT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            1.0 AS block_score
        FROM s1_batch s
        CROSS JOIN UNNEST(s.address_numbers) u(token)
        JOIN number_top20 i
          ON i.token = u.token


        UNION ALL


        -- ================================================
        -- Postal top-20
        -- ================================================

        SELECT DISTINCT
            s.source1_entity_id,
            CAST(i.entity_id AS VARCHAR) AS entity_id,
            1.5 AS block_score
        FROM s1_batch s
        CROSS JOIN UNNEST(s.postal_tokens) u(token)
        JOIN postal_top20 i
          ON i.token = u.token
    ),

    scored AS (

        SELECT
            source1_entity_id,
            entity_id,
            SUM(block_score) AS score
        FROM block_hits
        GROUP BY
            source1_entity_id,
            entity_id
    ),

    ranked AS (

        SELECT
            source1_entity_id,
            entity_id,
            score,
            ROW_NUMBER() OVER (
                PARTITION BY source1_entity_id
                ORDER BY
                    score DESC,
                    entity_id
            ) AS rn
        FROM scored
    )

    SELECT
        source1_entity_id,
        entity_id,
        score
    FROM ranked
    WHERE rn <= ?
    ORDER BY
        source1_entity_id,
        rn
    """

    result = con.execute(
        query,
        [MAX_CANDIDATES],
    ).fetchdf()

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DUCKDB CANDIDATE PAIRS GENERATOR")
    print("=" * 70)

    print()
    print(f"Source 1 : {SOURCE1_PATH}")
    print(f"Database : {DB_PATH}")
    print(f"Output   : {OUTPUT_PATH}")
    print(f"Batch    : {BATCH_SIZE:,}")
    print(f"Max/S1   : {MAX_CANDIDATES}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(SOURCE1_PATH):
        raise FileNotFoundError(
            f"Source 1 not found: {SOURCE1_PATH}"
        )

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"DuckDB database not found: {DB_PATH}"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Remove old output
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_PATH):

        print()
        print(
            "Removing previous candidate_pairs.tsv..."
        )

        os.remove(
            OUTPUT_PATH
        )

    # --------------------------------------------------------
    # DuckDB
    # --------------------------------------------------------

    print()
    print("Opening DuckDB...")

    con = duckdb.connect(
        DB_PATH
    )

    con.execute(
        "PRAGMA threads=1"
    )

    con.execute(
        "PRAGMA memory_limit='2GB'"
    )

    con.execute(
        f"SET temp_directory='{TEMP_DIR}'"
    )

    # --------------------------------------------------------
    # Verify required tables
    # --------------------------------------------------------

    required_tables = [
        "name_index",
        "name_no_suffix_index",
        "compact_name_index",
        "name_token_top30",
        "address_token_top30",
        "translit_token_top20",
        "number_top20",
        "postal_top20",
    ]

    existing = {
        row[0]
        for row in con.execute(
            "SHOW TABLES"
        ).fetchall()
    }

    missing = [
        table
        for table in required_tables
        if table not in existing
    ]

    if missing:

        con.close()

        raise RuntimeError(
            "Missing required DuckDB tables: "
            + ", ".join(missing)
        )

    print(
        "All required blocking indexes found."
    )

    # --------------------------------------------------------
    # Read Source 1 in chunks
    # --------------------------------------------------------

    reader = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        chunksize=BATCH_SIZE,
        keep_default_na=False,
    )

    total_s1 = 0
    total_pairs = 0

    start_time = time.time()

    header_written = False

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    for batch_number, raw_batch in enumerate(
        reader,
        start=1,
    ):

        batch_start = time.time()

        # -----------------------------------------------
        # Normalize only this small S1 batch
        # -----------------------------------------------

        batch = prepare_source1(
            raw_batch
        )

        # -----------------------------------------------
        # Generate top-180 candidates
        # -----------------------------------------------

        scored = generate_batch(
            con,
            batch,
        )

        # -----------------------------------------------
        # Convert to candidate_pairs format
        # -----------------------------------------------

        if scored.empty:

            pair_rows = pd.DataFrame(
                {
                    "source1_entity_id":
                        batch[
                            "source1_entity_id"
                        ].astype(str),

                    "candidate_entity_ids":
                        [""]
                        * len(batch),
                }
            )

        else:

            grouped = (
                scored
                .sort_values(
                    [
                        "source1_entity_id",
                        "score",
                        "entity_id",
                    ],
                    ascending=[
                        True,
                        False,
                        True,
                    ],
                )
                .groupby(
                    "source1_entity_id",
                    sort=False,
                )["entity_id"]
                .apply(
                    lambda x:
                        ",".join(
                            x.astype(str)
                        )
                )
                .rename(
                    "candidate_entity_ids"
                )
                .reset_index()
            )

            # -------------------------------------------
            # Guarantee one row for EVERY S1
            # -------------------------------------------

            pair_rows = (
                batch[
                    ["source1_entity_id"]
                ]
                .merge(
                    grouped,
                    on="source1_entity_id",
                    how="left",
                )
            )

            pair_rows[
                "candidate_entity_ids"
            ] = (
                pair_rows[
                    "candidate_entity_ids"
                ]
                .fillna("")
            )

        # -----------------------------------------------
        # Write immediately
        # -----------------------------------------------

        pair_rows.to_csv(
            OUTPUT_PATH,
            sep="\t",
            index=False,
            mode="a",
            header=not header_written,
        )

        header_written = True

        # -----------------------------------------------
        # Counters
        # -----------------------------------------------

        batch_s1 = len(batch)

        batch_pairs = (
            pair_rows[
                "candidate_entity_ids"
            ]
            .map(
                lambda x:
                    0
                    if not x
                    else len(
                        str(x).split(",")
                    )
            )
            .sum()
        )

        total_s1 += batch_s1
        total_pairs += batch_pairs

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            total_s1 / elapsed
            if elapsed > 0
            else 0
        )

        batch_elapsed = (
            time.time()
            - batch_start
        )

        print(
            f"Batch {batch_number:>5} | "
            f"S1: {total_s1:>10,} | "
            f"Pairs: {total_pairs:>12,} | "
            f"Rate: {rate:>7.1f} S1/s | "
            f"Batch: {batch_elapsed:>5.1f}s"
        )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    con.close()

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Source 1 processed : {total_s1:,}"
    )

    print(
        f"Candidate pairs    : {total_pairs:,}"
    )

    print(
        f"Average/S1         : "
        f"{total_pairs / total_s1:.2f}"
        if total_s1
        else "Average/S1         : 0"
    )

    print(
        f"Elapsed             : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print(
        "IMPORTANT: This file contains "
        "candidate pairs only."
    )


if __name__ == "__main__":
    main()