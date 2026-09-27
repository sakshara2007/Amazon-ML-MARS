from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd


class CandidateGenerator:
    """
    Memory-conscious candidate generator.

    Blocking signals:

    Existing:
      1. Exact normalized name
      2. Name without legal suffix
      3. Rare name tokens
      4. Rare address tokens
      5. Postal code
      6. Number

    Additional:
      7. Transliteration-aware name tokens
      8. Compact/domain-style name
      9. Compact character n-grams

    The final candidate list is still capped at
    max_candidates_per_s1.
    """

    def __init__(
        self,
        source23: pd.DataFrame,
        config,
    ):
        self.cfg = config
        self.df = source23.reset_index(drop=True)

        self.id_to_pos = {
            str(eid): i
            for i, eid in enumerate(
                self.df["entity_id"].astype(str)
            )
        }

        # -----------------------------------------------------
        # Existing indexes
        # -----------------------------------------------------

        self.name_exact: Dict[str, List[int]] = (
            defaultdict(list)
        )

        self.name_no_suffix: Dict[str, List[int]] = (
            defaultdict(list)
        )

        self.name_token: Dict[str, List[int]] = (
            defaultdict(list)
        )

        self.address_token: Dict[str, List[int]] = (
            defaultdict(list)
        )

        self.postal_index: Dict[str, List[int]] = (
            defaultdict(list)
        )

        self.number_index: Dict[str, List[int]] = (
            defaultdict(list)
        )

        # -----------------------------------------------------
        # New indexes
        # -----------------------------------------------------

        self.translit_name_token: Dict[str, List[int]] = (
            defaultdict(list)
        )

        self.compact_name_index: Dict[str, List[int]] = (
            defaultdict(list)
        )

        self.name_ngram_index: Dict[str, List[int]] = (
            defaultdict(list)
        )

        # -----------------------------------------------------
        # Document frequencies
        # -----------------------------------------------------

        name_token_df = Counter()
        address_token_df = Counter()
        translit_token_df = Counter()
        ngram_df = Counter()

        postal_df = Counter()
        number_df = Counter()

        # -----------------------------------------------------
        # First pass: document frequencies
        # -----------------------------------------------------

        for _, row in self.df.iterrows():

            # Existing name tokens
            for token in set(
                row["name_tokens"] or []
            ):
                token = str(token)

                if (
                    len(token)
                    >= self.cfg.min_token_length
                ):
                    name_token_df[token] += 1

            # Existing address tokens
            for token in set(
                row["address_tokens"] or []
            ):
                token = str(token)

                if (
                    len(token)
                    >= self.cfg.min_token_length
                ):
                    address_token_df[token] += 1

            # New transliterated tokens
            translit_name = str(
                row.get(
                    "name_translit",
                    "",
                )
                or ""
            )

            for token in set(
                translit_name.split()
            ):
                if (
                    len(token)
                    >= self.cfg.min_token_length
                ):
                    translit_token_df[token] += 1

            # New compact n-grams
            for gram in set(
                row.get(
                    "name_ngrams",
                    [],
                )
                or []
            ):
                gram = str(gram)

                if gram:
                    ngram_df[gram] += 1

            # Postal
            for postal in set(
                row["postal_tokens"] or []
            ):
                postal = str(postal)

                if postal:
                    postal_df[postal] += 1

            # Numbers
            numbers = set(
                (row["name_numbers"] or [])
                +
                (row["address_numbers"] or [])
            )

            for number in numbers:
                number = str(number)

                if number:
                    number_df[number] += 1

        n = max(len(self.df), 1)

        # -----------------------------------------------------
        # Rare-token thresholds
        # -----------------------------------------------------

        max_name_df = max(
            50,
            int(
                n
                * self.cfg.max_common_token_df_ratio
            ),
        )

        max_address_df = max(
            100,
            int(
                n
                * self.cfg.max_common_token_df_ratio
            ),
        )

        max_translit_df = max(
            50,
            int(
                n
                * self.cfg.max_common_token_df_ratio
            ),
        )

        max_ngram_df = max(
            100,
            int(
                n
                * 0.01
            ),
        )

        self.rare_name_tokens = {
            token
            for token, count
            in name_token_df.items()
            if count <= max_name_df
        }

        self.rare_address_tokens = {
            token
            for token, count
            in address_token_df.items()
            if count <= max_address_df
        }

        self.rare_translit_tokens = {
            token
            for token, count
            in translit_token_df.items()
            if count <= max_translit_df
        }

        self.useful_ngrams = {
            gram
            for gram, count
            in ngram_df.items()
            if count <= max_ngram_df
        }

        self.useful_postals = {
            token
            for token, count
            in postal_df.items()
            if count <= max(
                1000,
                int(n * 0.01),
            )
        }

        self.useful_numbers = {
            token
            for token, count
            in number_df.items()
            if count <= max(
                5000,
                int(n * 0.02),
            )
        }

        # -----------------------------------------------------
        # Second pass: build indexes
        # -----------------------------------------------------

        for i, row in self.df.iterrows():

            # ---------------------------------------------
            # Existing exact name
            # ---------------------------------------------

            name_norm = str(
                row["name_norm"] or ""
            )

            name_no_suffix = str(
                row["name_no_suffix"] or ""
            )

            if name_norm:
                self.name_exact[
                    name_norm
                ].append(i)

            if name_no_suffix:
                self.name_no_suffix[
                    name_no_suffix
                ].append(i)

            # ---------------------------------------------
            # Existing name tokens
            # ---------------------------------------------

            for token in set(
                row["name_tokens"] or []
            ):

                token = str(token)

                if (
                    len(token)
                    >= self.cfg.min_token_length
                    and token
                    in self.rare_name_tokens
                ):
                    self.name_token[
                        token
                    ].append(i)

            # ---------------------------------------------
            # Existing address tokens
            # ---------------------------------------------

            for token in set(
                row["address_tokens"] or []
            ):

                token = str(token)

                if (
                    len(token)
                    >= self.cfg.min_token_length
                    and token
                    in self.rare_address_tokens
                ):
                    self.address_token[
                        token
                    ].append(i)

            # ---------------------------------------------
            # New transliterated name tokens
            # ---------------------------------------------

            translit_name = str(
                row.get(
                    "name_translit",
                    "",
                )
                or ""
            )

            for token in set(
                translit_name.split()
            ):

                token = str(token)

                if (
                    len(token)
                    >= self.cfg.min_token_length
                    and token
                    in self.rare_translit_tokens
                ):
                    self.translit_name_token[
                        token
                    ].append(i)

            # ---------------------------------------------
            # New compact name
            # ---------------------------------------------

            compact = str(
                row.get(
                    "name_compact",
                    "",
                )
                or ""
            )

            if compact:
                self.compact_name_index[
                    compact
                ].append(i)

            # ---------------------------------------------
            # New character n-grams
            # ---------------------------------------------

            for gram in set(
                row.get(
                    "name_ngrams",
                    [],
                )
                or []
            ):

                gram = str(gram)

                if (
                    gram
                    and gram
                    in self.useful_ngrams
                ):
                    self.name_ngram_index[
                        gram
                    ].append(i)

            # ---------------------------------------------
            # Postal
            # ---------------------------------------------

            for postal in set(
                row["postal_tokens"] or []
            ):

                postal = str(postal)

                if postal in self.useful_postals:
                    self.postal_index[
                        postal
                    ].append(i)

            # ---------------------------------------------
            # Numbers
            # ---------------------------------------------

            numbers = set(
                (row["name_numbers"] or [])
                +
                (row["address_numbers"] or [])
            )

            for number in numbers:

                number = str(number)

                if number in self.useful_numbers:
                    self.number_index[
                        number
                    ].append(i)

    # =========================================================
    # Similarity helpers
    # =========================================================

    @staticmethod
    def _token_overlap(a, b) -> float:

        a = set(a or [])
        b = set(b or [])

        if not a or not b:
            return 0.0

        return len(a & b) / len(a | b)

    @staticmethod
    def _char_similarity(
        a: str,
        b: str,
    ) -> float:

        from difflib import SequenceMatcher

        a = str(a or "")
        b = str(b or "")

        if not a or not b:
            return 0.0

        return SequenceMatcher(
            None,
            a,
            b,
        ).ratio()

    @staticmethod
    def _number_overlap(a, b) -> float:

        a = set(a or [])
        b = set(b or [])

        if not a or not b:
            return 0.0

        return (
            1.0
            if a & b
            else 0.0
        )

    @staticmethod
    def _postal_overlap(a, b) -> float:

        a = set(a or [])
        b = set(b or [])

        if not a or not b:
            return 0.0

        return (
            1.0
            if a & b
            else 0.0
        )

    # =========================================================
    # Generate candidates for one S1
    # =========================================================

    def generate_one(
        self,
        s1row,
    ) -> List[Tuple[int, Dict]]:

        candidates = set()

        evidence = defaultdict(dict)

        # -----------------------------------------------------
        # 1. Exact normalized name
        # -----------------------------------------------------

        name_norm = str(
            s1row["name_norm"] or ""
        )

        for idx in self.name_exact.get(
            name_norm,
            [],
        ):

            candidates.add(idx)

            evidence[idx][
                "exact_name"
            ] = 1.0

        # -----------------------------------------------------
        # 2. Name without legal suffix
        # -----------------------------------------------------

        name_no_suffix = str(
            s1row["name_no_suffix"] or ""
        )

        for idx in self.name_no_suffix.get(
            name_no_suffix,
            [],
        ):

            candidates.add(idx)

            evidence[idx][
                "name_no_suffix"
            ] = 1.0

        # -----------------------------------------------------
        # 3. Rare name-token block
        # -----------------------------------------------------

        name_counts = Counter()

        for token in set(
            s1row["name_tokens"] or []
        ):

            token = str(token)

            if token in self.rare_name_tokens:

                for idx in self.name_token.get(
                    token,
                    [],
                ):
                    name_counts[idx] += 1

        for idx, hits in name_counts.most_common(
            self.cfg.token_top_k
        ):

            candidates.add(idx)

            evidence[idx][
                "name_token_hits"
            ] = hits

        # -----------------------------------------------------
        # 4. Rare address-token block
        # -----------------------------------------------------

        address_counts = Counter()

        for token in set(
            s1row["address_tokens"] or []
        ):

            token = str(token)

            if token in self.rare_address_tokens:

                for idx in self.address_token.get(
                    token,
                    [],
                ):
                    address_counts[idx] += 1

        for idx, hits in address_counts.most_common(
            self.cfg.token_top_k
        ):

            candidates.add(idx)

            evidence[idx][
                "address_token_hits"
            ] = hits

        # -----------------------------------------------------
        # 5. Transliteration-aware name block
        # -----------------------------------------------------

        translit_counts = Counter()

        translit_name = str(
            s1row.get(
                "name_translit",
                "",
            )
            or ""
        )

        for token in set(
            translit_name.split()
        ):

            token = str(token)

            if token in self.rare_translit_tokens:

                for idx in self.translit_name_token.get(
                    token,
                    [],
                ):

                    translit_counts[idx] += 1

        for idx, hits in translit_counts.most_common(
            self.cfg.translit_token_top_k
        ):

            candidates.add(idx)

            evidence[idx][
                "translit_token_hits"
            ] = hits

        # -----------------------------------------------------
        # 6. Compact-name block
        # -----------------------------------------------------

        compact = str(
            s1row.get(
                "name_compact",
                "",
            )
            or ""
        )

        if compact:

            for idx in self.compact_name_index.get(
                compact,
                [],
            ):

                candidates.add(idx)

                evidence[idx][
                    "compact_name"
                ] = 1.0

        # -----------------------------------------------------
        # 7. Character n-gram block
        # -----------------------------------------------------

        ngram_counts = Counter()

        for gram in set(
            s1row.get(
                "name_ngrams",
                [],
            )
            or []
        ):

            gram = str(gram)

            if gram in self.useful_ngrams:

                for idx in self.name_ngram_index.get(
                    gram,
                    [],
                ):

                    ngram_counts[idx] += 1

        for idx, hits in ngram_counts.most_common(
            self.cfg.ngram_top_k
        ):

            candidates.add(idx)

            evidence[idx][
                "ngram_hits"
            ] = hits

        # -----------------------------------------------------
        # 8. Postal-code block
        # -----------------------------------------------------

        for postal in set(
            s1row["postal_tokens"] or []
        ):

            postal = str(postal)

            if postal in self.useful_postals:

                for idx in self.postal_index.get(
                    postal,
                    [],
                ):

                    candidates.add(idx)

                    evidence[idx][
                        "postal_match"
                    ] = 1.0

        # -----------------------------------------------------
        # 9. Number block
        # -----------------------------------------------------

        numbers = set(
            (s1row["name_numbers"] or [])
            +
            (s1row["address_numbers"] or [])
        )

        number_counts = Counter()

        for number in numbers:

            number = str(number)

            if number in self.useful_numbers:

                for idx in self.number_index.get(
                    number,
                    [],
                ):

                    number_counts[idx] += 1

        for idx, hits in number_counts.most_common(
            self.cfg.number_top_k
        ):

            candidates.add(idx)

            evidence[idx][
                "number_hits"
            ] = hits

        # -----------------------------------------------------
        # 10. Local similarity scoring
        # -----------------------------------------------------

        scored = []

        source_country = str(
            s1row["country_norm"] or ""
        )

        for idx in candidates:

            row = self.df.iloc[idx]

            name_similarity = (
                self._char_similarity(
                    name_norm,
                    row["name_norm"],
                )
            )

            address_similarity = (
                self._char_similarity(
                    s1row["address_norm"],
                    row["address_norm"],
                )
            )

            name_token_similarity = (
                self._token_overlap(
                    s1row["name_tokens"],
                    row["name_tokens"],
                )
            )

            address_token_similarity = (
                self._token_overlap(
                    s1row["address_tokens"],
                    row["address_tokens"],
                )
            )

            number_match = (
                self._number_overlap(
                    numbers,
                    (
                        row["name_numbers"]
                        or []
                    )
                    +
                    (
                        row["address_numbers"]
                        or []
                    ),
                )
            )

            postal_match = (
                self._postal_overlap(
                    s1row["postal_tokens"],
                    row["postal_tokens"],
                )
            )

            country_match = (
                1.0
                if (
                    source_country
                    and
                    source_country
                    ==
                    str(
                        row["country_norm"]
                        or ""
                    )
                )
                else 0.0
            )

            e = evidence[idx]

            # Existing score
            final_score = (
                4.0
                * e.get(
                    "exact_name",
                    0.0,
                )
                +
                3.5
                * e.get(
                    "name_no_suffix",
                    0.0,
                )
                +
                2.0
                * name_token_similarity
                +
                1.5
                * address_token_similarity
                +
                2.0
                * name_similarity
                +
                1.5
                * address_similarity
                +
                1.5
                * postal_match
                +
                1.0
                * number_match
                +
                0.75
                * country_match
                +
                0.25
                * e.get(
                    "name_token_hits",
                    0,
                )
                +
                0.15
                * e.get(
                    "address_token_hits",
                    0,
                )
            )

            # New blocking evidence receives
            # small ranking bonuses.
            final_score += (
                0.35
                * e.get(
                    "translit_token_hits",
                    0,
                )
            )

            final_score += (
                1.0
                * e.get(
                    "compact_name",
                    0.0,
                )
            )

            final_score += (
                0.05
                * e.get(
                    "ngram_hits",
                    0,
                )
            )

            evidence[idx].update(
                {
                    "name_similarity":
                        round(
                            name_similarity,
                            6,
                        ),

                    "address_similarity":
                        round(
                            address_similarity,
                            6,
                        ),

                    "name_token_similarity":
                        round(
                            name_token_similarity,
                            6,
                        ),

                    "address_token_similarity":
                        round(
                            address_token_similarity,
                            6,
                        ),

                    "number_match":
                        number_match,

                    "postal_match":
                        postal_match,

                    "country_match":
                        country_match,

                    "blocking_score":
                        round(
                            final_score,
                            6,
                        ),
                }
            )

            scored.append(
                (
                    idx,
                    final_score,
                )
            )

        # -----------------------------------------------------
        # 11. Rank and cap
        # -----------------------------------------------------

        scored.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        scored = scored[
            : self.cfg.max_candidates_per_s1
        ]

        return [
            (
                idx,
                evidence[idx],
            )
            for idx, _ in scored
        ]

    # =========================================================
    # Full generation
    # =========================================================

    def generate(self, source1):

        all_rows = []
        feature_rows = []

        for _, s1row in source1.iterrows():

            pairs = self.generate_one(
                s1row
            )

            ids = [
                str(
                    self.df.iloc[idx][
                        "entity_id"
                    ]
                )
                for idx, _ in pairs
            ]

            all_rows.append(
                {
                    "source1_entity_id":
                        str(
                            s1row[
                                "entity_id"
                            ]
                        ),

                    "candidate_entity_ids":
                        ",".join(ids),
                }
            )

            for idx, features in pairs:

                feature_rows.append(
                    {
                        "source1_entity_id":
                            str(
                                s1row[
                                    "entity_id"
                                ]
                            ),

                        "candidate_entity_id":
                            str(
                                self.df.iloc[idx][
                                    "entity_id"
                                ]
                            ),

                        **features,
                    }
                )

        return (
            all_rows,
            feature_rows,
        )

    # =========================================================
    # Streaming generation
    # =========================================================

    def generate_iter(self, source1):

        for _, s1row in source1.iterrows():

            pairs = self.generate_one(
                s1row
            )

            ids = [
                str(
                    self.df.iloc[idx][
                        "entity_id"
                    ]
                )
                for idx, _ in pairs
            ]

            candidate_row = {
                "source1_entity_id":
                    str(
                        s1row[
                            "entity_id"
                        ]
                    ),

                "candidate_entity_ids":
                    ",".join(ids),
            }

            feature_rows = []

            for idx, features in pairs:

                feature_rows.append(
                    {
                        "source1_entity_id":
                            str(
                                s1row[
                                    "entity_id"
                                ]
                            ),

                        "candidate_entity_id":
                            str(
                                self.df.iloc[idx][
                                    "entity_id"
                                ]
                            ),

                        **features,
                    }
                )

            yield (
                candidate_row,
                feature_rows,
            )