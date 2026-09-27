import re
from difflib import SequenceMatcher

def safe_str(x):
    return "" if x is None else str(x)

def normalize_simple(text):
    text = safe_str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokens(text):
    return set(normalize_simple(text).split())

def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def overlap_min(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))

def containment(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))

def seq_ratio(a, b):
    a, b = normalize_simple(a), normalize_simple(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def length_ratio(a, b):
    la, lb = len(safe_str(a)), len(safe_str(b))
    if max(la, lb) == 0:
        return 1.0
    return min(la, lb) / max(la, lb)

def number_tokens(text):
    return set(re.findall(r"\d+[A-Za-z]?", safe_str(text).lower()))

def number_overlap(a, b):
    return jaccard(number_tokens(a), number_tokens(b))

def postal_tokens(text):
    # Generic, non-country-specific postal-like alphanumeric chunks.
    vals = re.findall(r"\b[A-Z0-9][A-Z0-9 -]{2,9}[A-Z0-9]\b",
                      safe_str(text).upper())
    return {re.sub(r"\s+", "", x) for x in vals}

def postal_overlap(a, b):
    return jaccard(postal_tokens(a), postal_tokens(b))
