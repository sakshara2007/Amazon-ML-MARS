from dataclasses import dataclass


@dataclass
class CandidateConfig:

    # Existing blocking
    token_top_k: int = 30
    number_top_k: int = 20

    # New blocking
    translit_token_top_k: int = 20
    compact_name_top_k: int = 30
    ngram_top_k: int = 20

    # Final candidate limit
    max_candidates_per_s1: int = 180

    # Rare-token controls
    max_common_token_df_ratio: float = 0.002
    min_token_length: int = 2

    # Streaming output
    output_chunk_size: int = 2000