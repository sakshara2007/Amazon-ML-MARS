import re
import unicodedata

from unidecode import unidecode


ABBREVIATIONS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "highway": "hwy",
    "lane": "ln",
    "drive": "dr",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "private": "pvt",
    "limited": "ltd",
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
}


LEGAL_SUFFIXES = {
    "pvt",
    "private",
    "ltd",
    "limited",
    "llp",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
}


def unicode_normalize(text):
    text = "" if text is None else str(text)
    return unicodedata.normalize("NFKC", text)


def basic_normalize(text):
    text = unicode_normalize(text).lower()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"_+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalize_tokens(text):
    s = basic_normalize(text)

    tokens = []

    for tok in s.split():

        tokens.append(
            ABBREVIATIONS.get(
                tok,
                tok,
            )
        )

    return tokens


def normalize_text(text):
    return " ".join(
        normalize_tokens(text)
    )


def transliterate_text(text):
    """
    Convert non-Latin text into an approximate
    Latin representation.

    Used only as an additional blocking signal.
    The original normalized text is preserved.
    """

    text = unicode_normalize(text)

    text = unidecode(text)

    return normalize_text(text)


def remove_legal_suffixes(text):

    tokens = normalize_text(text).split()

    while (
        tokens
        and tokens[-1] in LEGAL_SUFFIXES
    ):
        tokens.pop()

    return " ".join(tokens)


def compact_name(text):
    """
    Create a compact representation for
    approximate name blocking.

    Examples:

        Gulf Highland Bold LLC
        -> gulfhighlandbold

        gulfhighlandbold.com
        -> gulfhighlandbold
    """

    # Work from the original text so domain suffixes
    # can be detected before punctuation is removed.
    raw = unicode_normalize(text).lower()

    # Remove web/domain suffixes first.
    raw = re.sub(
        r"\.(?:com|net|org|biz|info|co|in)\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    # Then transliterate and normalize.
    s = transliterate_text(raw)

    # Remove legal suffixes.
    tokens = s.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    s = " ".join(tokens)

    # Remove any remaining non-alphanumeric characters.
    s = re.sub(
        r"[^a-z0-9]",
        "",
        s,
    )

    return s

def compact_name_tokens(text):
    """
    Return character n-grams for approximate
    name blocking.
    """

    s = compact_name(text)

    if not s:
        return []

    ngrams = set()

    # 4- and 5-character grams.
    for n in (4, 5):

        if len(s) < n:
            continue

        for i in range(
            len(s) - n + 1
        ):
            ngrams.add(
                s[i:i + n]
            )

    return list(ngrams)


def extract_numbers(text):
    return re.findall(
        r"\d+[A-Za-z]?",
        unicode_normalize(text).lower(),
    )


def extract_postal_codes(text):

    s = unicode_normalize(text).upper()

    tokens = re.findall(
        r"\b[A-Z0-9][A-Z0-9 -]{2,9}[A-Z0-9]\b",
        s,
    )

    return [
        re.sub(
            r"\s+",
            "",
            x,
        )
        for x in tokens
    ]


def prepare(df):

    out = df.copy()

    # Existing fields
    out["name_norm"] = (
        out["business_name"]
        .map(normalize_text)
    )

    out["name_no_suffix"] = (
        out["business_name"]
        .map(remove_legal_suffixes)
    )

    out["address_norm"] = (
        out["business_address"]
        .map(normalize_text)
    )

    # NEW: transliterated name
    out["name_translit"] = (
        out["business_name"]
        .map(transliterate_text)
    )

    # NEW: compact name
    out["name_compact"] = (
        out["business_name"]
        .map(compact_name)
    )

    # NEW: compact character n-grams
    out["name_ngrams"] = (
        out["business_name"]
        .map(compact_name_tokens)
    )

    # Existing token fields
    out["name_tokens"] = (
        out["name_norm"]
        .str.split()
    )

    out["address_tokens"] = (
        out["address_norm"]
        .str.split()
    )

    out["name_numbers"] = (
        out["business_name"]
        .map(extract_numbers)
    )

    out["address_numbers"] = (
        out["business_address"]
        .map(extract_numbers)
    )

    out["postal_tokens"] = (
        out["business_address"]
        .map(extract_postal_codes)
    )

    out["country_norm"] = (
        out["country"]
        .map(
            lambda x: basic_normalize(x)
        )
    )

    return out