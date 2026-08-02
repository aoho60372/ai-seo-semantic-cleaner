"""Safe, shared input handling for SEO semantic-core files."""

from __future__ import annotations

import codecs
from pathlib import Path
from typing import Iterator

import pandas as pd


PHRASE_HEADERS = {
    "query",
    "keyword",
    "phrase",
    "search query",
    "поисковый запрос",
    "фраза",
    "запрос",
}


def detect_text_encoding(path: Path) -> str:
    """Return a practical encoding for Russian SEO exports without modifying them."""
    sample = path.read_bytes()[:262_144]
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for encoding in ("utf-8", "cp1251", "cp866"):
        try:
            # ``final=False`` accepts an incomplete multibyte sequence only at
            # the artificial end of our bounded sample.  A regular decode()
            # would misclassify a perfectly valid large UTF-8 file whenever
            # byte 262_144 falls inside a Cyrillic character.
            codecs.getincrementaldecoder(encoding)(errors="strict").decode(
                sample, final=False
            )
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "unknown", sample, 0, min(len(sample), 1),
        "CSV is neither UTF-8, CP1251, nor CP866",
    )


def _header(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, newline="") as handle:
        return handle.readline().rstrip("\r\n")


def delimited_read_options(path: Path) -> dict[str, object]:
    """Choose a deterministic separator; never let pandas guess from phrase text."""
    encoding = detect_text_encoding(path)
    header = _header(path, encoding).strip().lower()
    if path.suffix.lower() == ".tsv" or "\t" in header:
        separator = "\t"
    elif header in PHRASE_HEADERS:
        # A one-column semantic core may contain commas in the query itself.
        # Tab keeps every physical input line intact as one phrase.
        separator = "\t"
    elif ";" in header:
        separator = ";"
    elif "," in header:
        separator = ","
    else:
        separator = "\t"
    return {"sep": separator, "engine": "python", "encoding": encoding}


def read_delimited_table(path: Path, **kwargs: object) -> pd.DataFrame:
    options = delimited_read_options(path)
    options.update(kwargs)
    return pd.read_csv(path, **options)


def iter_delimited_chunks(path: Path, chunksize: int, **kwargs: object) -> Iterator[pd.DataFrame]:
    options = delimited_read_options(path)
    options.update(kwargs)
    return pd.read_csv(path, chunksize=chunksize, **options)
