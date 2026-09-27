import re

def clean(text):
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def make_name(record):
    return clean(record.get("business_name", ""))

def make_address(record):
    return clean(record.get("business_address", ""))

def make_combined(record):
    name = make_name(record)
    address = make_address(record)
    return f"{name} [SEP] {address}".strip()
