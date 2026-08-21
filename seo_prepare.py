"""Prepare a representative semantic-core sample for model judgement."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from seo_io import iter_delimited_chunks, read_delimited_table
from seo_knowledge import (
    load_balanced_examples,
    load_topic_profile,
    topic_key,
)


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "files"
JOBS_DIR = PROJECT_DIR / "jobs"
DEFAULT_EMBEDDING_MODEL = "local:multilingual-e5-base"

FAMILY_PREFIX_LENGTH = 5
MAX_INTENT_FAMILY_CANDIDATES = 100
MAX_LEXICAL_INTENT_FAMILY_CANDIDATES = 60
MAX_STRUCTURAL_INTENT_FAMILY_CANDIDATES = 40
MAX_ANCHORED_STRUCTURAL_FAMILY_CANDIDATES = 20
MAX_DOCUMENT_STRUCTURAL_FAMILY_CANDIDATES = 10
LEXICAL_COVERAGE_RESCUE_CANDIDATES = 12
STRUCTURAL_COVERAGE_RESCUE_CANDIDATES = 8
INTENT_FAMILY_COVERAGE_SAMPLE_ROWS = 50
INTENT_FAMILY_EXAMPLES = 8
INTENT_FAMILY_EXAMPLE_CANDIDATES = 64
# Function words are not useful intent families. This is deliberately a small
# multilingual structural list, not a topic vocabulary.
FAMILY_STOP_WORDS = {
    "about", "after", "before", "from", "into", "near", "that", "the", "this", "with",
    "без", "был", "была", "были", "быть", "для", "его", "или", "как", "над", "она",
    "они", "под", "при", "про", "так", "что", "это", "этот", "этого", "эти", "этих",
}
# Domain-independent intent vocabulary is used only to prioritise which
# frequent families the main model reviews. It never assigns a class itself.
# The reviewed examples remain the source of the actual commercial,
# informational, or neutral decision.
INTENT_DISCOVERY_FAMILIES = {
    # Russian informational / commercial-investigation concepts.
    "отзыв*", "обзор*", "мнени*", "опыт", "плюс*", "минус*", "досто*", "недос*",
    "сравн*", "отлич*", "выбор*", "выбир*", "инстр*", "руков*", "причи*", "ошибк*",
    "неисп*", "ремон*", "замен*", "устан*", "сняти*", "схема*", "разме*", "харак*",
    "описа*", "соста*", "расхо*", "ресур*", "прове*", "рейти*", "надеж*", "пробл*",
    "жалоб*", "форум*", "видео*", "фото", "тест",
    # Russian transactional concepts.
    "купит*", "куплю*", "цена", "стоим*", "заказ*", "прода*", "магаз*", "доста*",
    "налич*", "аренд*", "услуг*", "прайс*", "скидк*", "акция*", "запис*",
    # Document/reference concepts are discovery priorities only. They do not
    # assign an intent: the main model still decides from diverse real phrases.
    "заявл*", "анкет*", "резюм*", "справ*", "догов*", "образ*", "бланк*",
    "форма*", "докум*", "шабло*",
    "appli*", "resum*", "form", "templ*", "contr*", "certi*", "docum*",
    # English equivalents for mixed and English-language cores.
    "revie*", "opini*", "exper*", "pros", "cons", "compa*", "guide*", "manua*",
    "specs*", "speci*", "repla*", "repai*", "insta*", "error*", "probl*", "ratin*",
    "price*", "cost", "order*", "sale", "shop", "deliv*", "rent",
}
DOCUMENT_DISCOVERY_FAMILIES = {
    "заявл*", "анкет*", "резюм*", "справ*", "догов*", "образ*", "бланк*",
    "форма*", "докум*", "шабло*",
    "appli*", "resum*", "form", "templ*", "contr*", "certi*", "docum*",
}
# Only unusually unambiguous lexical observations may become bounded family
# evidence.  Broader discovery stems (for example a stem that may mean either
# "sell" or "salesperson") are still shown to the model but cannot override a
# classifier by themselves.  Topic nouns, brands, products, and professions
# are deliberately absent.
SAFE_DECISIVE_LEXICAL_FAMILIES = {
    "отзыв*", "обзор*", "мнени*", "плюс*", "минус*", "досто*", "недос*",
    "сравн*", "отлич*", "инстр*", "причи*", "ошибк*", "неисп*", "схема*",
    "харак*", "описа*", "жалоб*", "рейти*", "купит*", "куплю*", "цена",
    "стоим*", "revie*", "opini*", "pros", "cons", "guide*", "manua*",
    "specs*", "price*", "cost", "order*",
}
# Language-level interrogative anchors only prioritise structures for model
# review; they never assign an intent.  This is intentionally independent of
# SEO topic vocabulary, so "where to buy" can still be commercial while
# "where is located" may be informational or irrelevant for the current job.
STRUCTURAL_DISCOVERY_ANCHORS = {
    "как", "где", "куда", "откуда", "почему", "зачем", "сколько", "что",
    "кто", "чей", "how", "where", "which", "what", "why", "when", "who",
}


def is_structural_discovery_anchor(token: str) -> bool:
    token = normalize(token)
    return token.startswith("как") or token in STRUCTURAL_DISCOVERY_ANCHORS


def structural_token_key(token: str) -> str | None:
    family = intent_family_key(token)
    if family:
        return family
    token = normalize(token)
    return token if is_structural_discovery_anchor(token) else None


def structural_family_is_prioritised(family: str) -> bool:
    return any(
        is_structural_discovery_anchor(token.rstrip("*"))
        for token in family.split()
    )


def structural_family_is_document_reference(family: str) -> bool:
    return any(token in DOCUMENT_DISCOVERY_FAMILIES for token in family.split())


def normalize(value: object) -> str:
    text = re.sub(r"[^a-zа-я0-9]+", " ", str(value).lower().replace("ё", "е"))
    return re.sub(r"\s+", " ", text).strip()


def intent_family_key(token: str) -> str | None:
    """Return a bounded lexical family used only for model-reviewed rules."""
    token = normalize(token)
    if (
        len(token) < 4
        or token in FAMILY_STOP_WORDS
        or token.isdigit()
        or not re.search(r"[a-zа-я]", token)
    ):
        return None
    if len(token) >= FAMILY_PREFIX_LENGTH:
        return token[:FAMILY_PREFIX_LENGTH] + "*"
    return token


def phrase_intent_families(value: object) -> set[str]:
    tokens = normalize(value).split()
    lexical = {
        family
        for token in tokens
        if (family := intent_family_key(token)) is not None
    }
    # Frequent two-token structures capture intent-bearing composition without
    # baking a topic or language vocabulary into the workflow.  For example,
    # the current model may review structures equivalent to "which vacancies"
    # separately from "what work does", while product/profession nouns remain
    # neutral lexical observations.  Limit the prefix to keep very long queries
    # and multi-million-row inputs bounded.
    structural: set[str] = set()
    bounded = tokens[:12]
    for left, right in zip(bounded, bounded[1:]):
        if left in FAMILY_STOP_WORDS and right in FAMILY_STOP_WORDS:
            continue
        left_family = structural_token_key(left)
        right_family = structural_token_key(right)
        if left_family and right_family:
            structural.add(f"{left_family} {right_family}")
    # Also bridge short/service words. Without this, high-value composition
    # such as "заявление на работу" loses the relationship between
    # "заявление" and "работа" merely because the preposition has no family.
    significant = [
        (index, family)
        for index, token in enumerate(bounded)
        if (family := structural_token_key(token)) is not None
    ]
    for (left_index, left_family), (right_index, right_family) in zip(
        significant, significant[1:]
    ):
        if right_index - left_index <= 3:
            structural.add(f"{left_family} {right_family}")
    for first, second, third in zip(bounded, bounded[1:], bounded[2:]):
        if not any(
            is_structural_discovery_anchor(token)
            for token in (first, second, third)
        ):
            continue
        parts = [structural_token_key(token) for token in (first, second, third)]
        if all(parts):
            structural.add(" ".join(str(part) for part in parts))
    return lexical | structural


def frequency_coverage_rescue(
    items: list[tuple[str, int]],
    excluded: set[str],
    limit: int,
) -> list[tuple[str, int]]:
    """Deterministically sample families below the head-frequency cutoff."""
    remaining = sorted(
        (item for item in items if item[0] not in excluded),
        key=lambda item: (-item[1], item[0]),
    )
    if limit <= 0 or not remaining:
        return []
    if len(remaining) <= limit:
        return remaining
    positions = np.linspace(0, len(remaining) - 1, num=limit, dtype=int)
    return [remaining[int(position)] for position in dict.fromkeys(positions)]


def document_structural_coverage(
    items: list[tuple[str, int]], limit: int
) -> list[tuple[str, int]]:
    """Reserve one strongest composition for each present document concept."""
    selected: list[tuple[str, int]] = []
    seen: set[str] = set()
    for document_family in sorted(DOCUMENT_DISCOVERY_FAMILIES):
        candidates = sorted(
            (
                item
                for item in items
                if document_family in item[0].split()
                and item[0] not in seen
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if candidates:
            selected.append(candidates[0])
            seen.add(candidates[0][0])
        if len(selected) >= limit:
            return selected
    for item in sorted(items, key=lambda value: (-value[1], value[0])):
        if item[0] not in seen:
            selected.append(item)
            seen.add(item[0])
        if len(selected) >= limit:
            break
    return selected


def select_intent_family_counts(
    counts: Counter[str], source_rows: int
) -> list[tuple[str, int]]:
    if source_rows <= 500:
        return []
    minimum = max(5, int(math.ceil(source_rows * 0.0002)))
    maximum = max(minimum, int(source_rows * 0.20))
    eligible = [
        (family, int(count))
        for family, count in counts.items()
        if minimum <= int(count) <= maximum
    ]
    lexical_items = [item for item in eligible if " " not in item[0]]
    prioritised_lexical = sorted(
        (item for item in lexical_items if item[0] in INTENT_DISCOVERY_FAMILIES),
        key=lambda item: (-item[1], item[0]),
    )
    ordinary_lexical = sorted(
        (item for item in lexical_items if item[0] not in INTENT_DISCOVERY_FAMILIES),
        key=lambda item: (-item[1], item[0]),
    )
    lexical_head_limit = (
        MAX_LEXICAL_INTENT_FAMILY_CANDIDATES
        - LEXICAL_COVERAGE_RESCUE_CANDIDATES
    )
    lexical_head = (prioritised_lexical + ordinary_lexical)[:lexical_head_limit]
    lexical_rescue = frequency_coverage_rescue(
        lexical_items,
        {item[0] for item in lexical_head},
        LEXICAL_COVERAGE_RESCUE_CANDIDATES,
    )
    lexical = lexical_head + lexical_rescue
    structural_items = [item for item in eligible if " " in item[0]]
    document_structural = document_structural_coverage(
        [
            item
            for item in structural_items
            if structural_family_is_document_reference(item[0])
        ],
        MAX_DOCUMENT_STRUCTURAL_FAMILY_CANDIDATES,
    )
    document_patterns = {item[0] for item in document_structural}
    question_structural = sorted(
        (
            item
            for item in structural_items
            if structural_family_is_prioritised(item[0])
            and item[0] not in document_patterns
        ),
        key=lambda item: (-item[1], item[0]),
    )[: MAX_ANCHORED_STRUCTURAL_FAMILY_CANDIDATES - len(document_structural)]
    anchored_structural = document_structural + question_structural
    general_structural_items = sorted(
        (item for item in structural_items if not structural_family_is_prioritised(item[0])),
        key=lambda item: (-item[1], item[0]),
    )
    general_head_limit = max(
        0,
        MAX_STRUCTURAL_INTENT_FAMILY_CANDIDATES
        - len(anchored_structural)
        - STRUCTURAL_COVERAGE_RESCUE_CANDIDATES,
    )
    general_structural = general_structural_items[:general_head_limit]
    structural_head = anchored_structural + general_structural
    structural_rescue = frequency_coverage_rescue(
        structural_items,
        {item[0] for item in structural_head},
        MAX_STRUCTURAL_INTENT_FAMILY_CANDIDATES - len(structural_head),
    )
    structural = structural_head + structural_rescue
    return (lexical + structural)[:MAX_INTENT_FAMILY_CANDIDATES]


def intent_family_payload(
    counts: Counter[str],
    examples: dict[str, list[str]],
    source_rows: int,
) -> dict[str, object]:
    selected = select_intent_family_counts(counts, source_rows)
    return {
        "version": 3,
        "source_rows": int(source_rows),
        "minimum_occurrences": max(5, int(math.ceil(source_rows * 0.0002))),
        "families": [
            {
                "id": f"IF{index:04d}",
                "pattern": family,
                "kind": "structural" if " " in family else "lexical",
                "safe_decisive_lexical": bool(
                    " " in family or family in SAFE_DECISIVE_LEXICAL_FAMILIES
                ),
                "discovery_priority": (
                    "document_reference"
                    if family in DOCUMENT_DISCOVERY_FAMILIES
                    else "document_reference_structure"
                    if " " in family and structural_family_is_document_reference(family)
                    else "intent_vocabulary"
                    if family in INTENT_DISCOVERY_FAMILIES
                    else "frequency_coverage"
                ),
                "occurrences": count,
                "share": round(count / max(source_rows, 1), 6),
                "examples": diverse_family_examples(
                    examples.get(family, []), family, INTENT_FAMILY_EXAMPLES
                ),
            }
            for index, (family, count) in enumerate(selected, start=1)
        ],
    }


def intent_family_coverage_rows(
    grouped: pd.DataFrame,
    family_payload: dict[str, object],
    limit: int = INTENT_FAMILY_COVERAGE_SAMPLE_ROWS,
) -> pd.DataFrame:
    """Choose real phrases that make important family coverage explicit."""
    if limit <= 0 or grouped.empty:
        return grouped.head(0).copy()
    families = [
        value
        for value in family_payload.get("families", [])
        if isinstance(value, dict)
    ]
    families.sort(
        key=lambda value: (
            0 if str(value.get("discovery_priority", "")).startswith("document_reference") else 1,
            0 if value.get("discovery_priority") == "intent_vocabulary" else 1,
            -int(value.get("occurrences", 0) or 0),
            str(value.get("pattern", "")),
        )
    )
    by_phrase = {
        normalize(phrase): index
        for index, phrase in grouped["Phrase"].items()
        if normalize(phrase)
    }
    chosen: list[int] = []
    chosen_set: set[int] = set()
    family_for_index: dict[int, str] = {}
    for family in families:
        if family.get("discovery_priority") == "document_reference":
            per_family = 4
        elif family.get("discovery_priority") == "document_reference_structure":
            per_family = 2
        else:
            per_family = 1
        added = 0
        for phrase in family.get("examples", []):
            index = by_phrase.get(normalize(phrase))
            if index is None or index in chosen_set:
                continue
            chosen.append(index)
            chosen_set.add(index)
            family_for_index[index] = str(family.get("pattern", ""))
            added += 1
            if added >= per_family or len(chosen) >= limit:
                break
        if len(chosen) >= limit:
            break
    result = grouped.loc[chosen].copy() if chosen else grouped.head(0).copy()
    result["Coverage Family"] = [family_for_index[index] for index in result.index]
    return result


def family_example_score(family: str, phrase: str) -> int:
    """Stable pseudo-random score used for bounded sampling across a full file."""
    digest = hashlib.blake2b(
        f"{family}\0{normalize(phrase)}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def add_family_example_candidates(
    phrases: object,
    selected: set[str],
    heaps: dict[str, list[tuple[int, str]]],
    seen: dict[str, set[str]],
) -> None:
    for value in phrases:
        phrase = str(value).strip()
        if not phrase:
            continue
        for family in phrase_intent_families(phrase) & selected:
            normalized_phrase = normalize(phrase)
            if normalized_phrase in seen[family]:
                continue
            score = family_example_score(family, normalized_phrase)
            heap = heaps[family]
            entry = (-score, phrase)
            if len(heap) < INTENT_FAMILY_EXAMPLE_CANDIDATES:
                heapq.heappush(heap, entry)
                seen[family].add(normalized_phrase)
            elif score < -heap[0][0]:
                _, removed = heapq.heapreplace(heap, entry)
                seen[family].discard(normalize(removed))
                seen[family].add(normalized_phrase)


def diverse_family_examples(
    phrases: object, family: str, limit: int = INTENT_FAMILY_EXAMPLES
) -> list[str]:
    """Choose lexically diverse contexts instead of the first matching rows."""
    unique = {normalize(value): str(value).strip() for value in phrases if str(value).strip()}
    candidates = list(unique.values())
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda value: family_example_score(family, value))

    stem = normalize(family).rstrip("*")

    def features(value: str) -> set[str]:
        tokens = normalize(value).split()
        context = {token for token in tokens if not token.startswith(stem)}
        context.add(f"__length_{min(len(tokens) // 3, 3)}")
        return context

    feature_map = {value: features(value) for value in candidates}
    ordered = sorted(candidates, key=lambda value: family_example_score(family, value))
    chosen = [ordered.pop(0)]
    while ordered and len(chosen) < limit:
        def diversity(value: str) -> tuple[float, int]:
            current = feature_map[value]
            nearest_similarity = max(
                len(current & feature_map[other]) / max(len(current | feature_map[other]), 1)
                for other in chosen
            )
            return (1.0 - nearest_similarity, -family_example_score(family, value))

        best = max(ordered, key=diversity)
        ordered.remove(best)
        chosen.append(best)
    return chosen


def collect_family_examples(
    phrases: object, selected: set[str]
) -> dict[str, list[str]]:
    heaps = {family: [] for family in selected}
    seen = {family: set() for family in selected}
    add_family_example_candidates(phrases, selected, heaps, seen)
    return {
        family: [phrase for _, phrase in heap]
        for family, heap in heaps.items()
    }


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zа-я0-9]+", "-", normalize(value)).strip("-")
    return slug[:80] or "seo-job"


def resolve_input(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    candidate = INPUT_DIR / Path(value).name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Input not found: {value}. Expected it in {INPUT_DIR}.")


def read_table(path: Path, phrase_column: str | None = None) -> tuple[pd.DataFrame, str | int]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return read_delimited_table(path), 0
    requested = normalize(phrase_column) if phrase_column else ""
    candidates = {"поисковый запрос", "фраза", "keyword", "query", "запрос"}
    with pd.ExcelFile(path) as workbook:
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
            columns = {normalize(column) for column in frame.columns}
            if (requested and requested in columns) or (not requested and columns & candidates):
                return frame, sheet_name
        available = "; ".join(workbook.sheet_names)
    raise ValueError(
        "No worksheet contains the requested phrase column. "
        f"Available worksheets: {available}. Use --phrase-column if needed."
    )


def streaming_csv_candidates(
    path: Path,
    phrase_column: str | None,
    frequency_column: str | None,
    sample_size: int,
) -> tuple[pd.DataFrame, str, str | None, dict[str, int], dict[str, object]]:
    """Read a delimited semantic core in chunks and retain a bounded sample only."""
    header = read_delimited_table(path, nrows=0)
    phrase_name = choose_phrase_column(header, phrase_column)
    frequency_name = choose_frequency_column(header, frequency_column)
    reservoir_size = max(5_000, sample_size * 12)
    rng = np.random.default_rng(42)
    reservoir: list[dict[str, object]] = []
    top: list[tuple[float, int, dict[str, object]]] = []
    seen_rows = 0
    non_empty = 0
    source_rows = 0
    family_counts: Counter[str] = Counter()

    for chunk in iter_delimited_chunks(path, chunksize=50_000):
        source_rows += len(chunk)
        raw = chunk[phrase_name].dropna().astype(str).str.strip()
        raw = raw[raw.ne("")]
        volumes = (
            pd.to_numeric(chunk.loc[raw.index, frequency_name], errors="coerce").fillna(0)
            if frequency_name
            else pd.Series(0.0, index=raw.index)
        )
        for row_index, phrase in raw.items():
            non_empty += 1
            seen_rows += 1
            family_counts.update(phrase_intent_families(phrase))
            record = {
                phrase_name: phrase,
                "__source_row": int(row_index) + 2,
                "__volume": float(volumes.loc[row_index]),
            }
            if len(reservoir) < reservoir_size:
                reservoir.append(record)
            else:
                replacement = int(rng.integers(0, seen_rows))
                if replacement < reservoir_size:
                    reservoir[replacement] = record
            priority = float(np.log1p(max(record["__volume"], 0)))
            entry = (priority, seen_rows, record)
            if len(top) < max(100, sample_size // 5):
                heapq.heappush(top, entry)
            elif entry[:2] > top[0][:2]:
                heapq.heapreplace(top, entry)

    if not reservoir:
        raise ValueError("The CSV/TSV file has no non-empty phrases.")
    candidates = pd.DataFrame(reservoir + [entry[2] for entry in top])
    candidates = candidates.drop_duplicates(subset=[phrase_name], keep="first")
    if frequency_name:
        candidates[frequency_name] = candidates["__volume"]
    candidates = candidates.drop(columns=["__source_row", "__volume"], errors="ignore")
    selected_families = {
        family for family, _ in select_intent_family_counts(family_counts, non_empty)
    }
    family_examples = {family: [] for family in selected_families}
    if selected_families:
        example_heaps = {family: [] for family in selected_families}
        example_seen = {family: set() for family in selected_families}
        for chunk in iter_delimited_chunks(path, chunksize=50_000, usecols=[phrase_name]):
            phrases = chunk[phrase_name].dropna().astype(str).str.strip()
            add_family_example_candidates(
                phrases[phrases.ne("")],
                selected_families,
                example_heaps,
                example_seen,
            )
        family_examples = {
            family: [phrase for _, phrase in heap]
            for family, heap in example_heaps.items()
        }
    family_payload = intent_family_payload(family_counts, family_examples, non_empty)
    return candidates, phrase_name, frequency_name, {
        "rows_in_source": source_rows,
        "non_empty_phrases": non_empty,
        # It is an upper bound, deliberately used to select the safe large path.
        "unique_phrase_upper_bound": non_empty,
    }, family_payload


def choose_phrase_column(frame: pd.DataFrame, requested: str | None) -> str:
    normalized = {normalize(column): column for column in frame.columns}
    if requested:
        matched = normalized.get(normalize(requested))
        if matched is None:
            raise ValueError(f"Column not found: {requested}. Available: {list(frame.columns)}")
        return matched
    for candidate in ("поисковый запрос", "фраза", "keyword", "query", "запрос"):
        if candidate in normalized:
            return normalized[candidate]
    text_columns = [column for column in frame.columns if frame[column].dtype == "object"]
    if not text_columns:
        raise ValueError("No text column found. Specify --phrase-column.")
    return text_columns[0]


def choose_frequency_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Frequency column not found: {requested}")
        return requested
    for column in frame.columns:
        name = normalize(column)
        if any(token in name for token in ("частот", "показы", "frequency", "volume")):
            return column
    return None


def representative_sample(table: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    if len(table) <= sample_size:
        return table.copy()

    chosen: set[int] = set()
    top_count = max(20, sample_size // 5)
    chosen.update(table.nlargest(top_count, "Priority").index.tolist())
    chosen.update(table.nlargest(max(10, sample_size // 10), "Length").index.tolist())

    remaining_target = sample_size - len(chosen)
    lexical_target = max(0, int(remaining_target * 0.65))
    candidates = table.drop(index=list(chosen))
    if lexical_target and len(candidates) > lexical_target:
        try:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=6000
            )
            matrix = vectorizer.fit_transform(candidates["Normalized"])
            cluster_count = min(lexical_target, max(20, sample_size // 2), len(candidates))
            model = MiniBatchKMeans(
                n_clusters=cluster_count,
                random_state=random_state,
                batch_size=2048,
                n_init="auto",
            )
            labels = model.fit_predict(matrix)
            distances = model.transform(matrix)
            local_indices = np.arange(len(candidates))
            for cluster_id in range(cluster_count):
                members = local_indices[labels == cluster_id]
                if len(members):
                    nearest = members[np.argmin(distances[members, cluster_id])]
                    chosen.add(int(candidates.index[nearest]))
        except ValueError:
            pass

    remaining = sample_size - len(chosen)
    if remaining > 0:
        pool = table.drop(index=list(chosen))
        chosen.update(pool.sample(n=min(remaining, len(pool)), random_state=random_state).index.tolist())
    return table.loc[sorted(chosen)].head(sample_size).copy()


def prepare_job(
    input_value: str,
    topic: str,
    sample_size: int = 800,
    phrase_column: str | None = None,
    frequency_column: str | None = None,
    job_name: str | None = None,
    reuse_topic: str | None = "auto",
    prior_limit: int = 150,
) -> Path:
    input_path = resolve_input(input_value)
    stream_stats: dict[str, int] | None = None
    family_payload: dict[str, object] | None = None
    if input_path.suffix.lower() in {".csv", ".tsv"}:
        source, phrase_column, frequency_column, stream_stats, family_payload = streaming_csv_candidates(
            input_path, phrase_column, frequency_column, sample_size
        )
        source_sheet: str | int = 0
    else:
        source, source_sheet = read_table(input_path, phrase_column)
        phrase_column = choose_phrase_column(source, phrase_column)
        frequency_column = choose_frequency_column(source, frequency_column)

    raw = source[phrase_column].dropna().astype(str).str.strip()
    raw = raw[raw.ne("")]
    working = pd.DataFrame({"Phrase": raw, "Source Row": raw.index + 2})
    working["Normalized"] = working["Phrase"].map(normalize)
    if frequency_column:
        frequency = pd.to_numeric(source.loc[raw.index, frequency_column], errors="coerce").fillna(0)
        working["Search Volume"] = frequency.to_numpy()
    else:
        working["Search Volume"] = 0.0

    grouped = (
        working.groupby("Normalized", as_index=False)
        .agg(
            Phrase=("Phrase", "first"),
            Occurrences=("Phrase", "size"),
            **{"Search Volume": ("Search Volume", "max"), "Source Row": ("Source Row", "first")},
        )
    )
    grouped["Length"] = grouped["Normalized"].str.len()
    grouped["Priority"] = np.log1p(grouped["Search Volume"].clip(lower=0)) + np.log1p(grouped["Occurrences"])
    unique_source_count = int(
        stream_stats["unique_phrase_upper_bound"] if stream_stats else len(grouped)
    )
    if family_payload is None:
        family_counts: Counter[str] = Counter()
        for phrase in raw:
            family_counts.update(phrase_intent_families(phrase))
        selected = {
            family
            for family, _ in select_intent_family_counts(
                family_counts, unique_source_count
            )
        }
        family_payload = intent_family_payload(
            family_counts,
            collect_family_examples(raw, selected),
            unique_source_count,
        )
    direct_model_labeling = unique_source_count <= 500
    reused_topic_key = None
    prior = pd.DataFrame()
    if not direct_model_labeling and reuse_topic and reuse_topic.lower() != "none":
        reused_topic_key = (
            topic_key(topic) if reuse_topic.lower() == "auto" else topic_key(reuse_topic)
        )
        prior = load_balanced_examples(
            reused_topic_key, min(prior_limit, max(0, sample_size // 3))
        )
    current_target = max(50, sample_size - len(prior))
    coverage = intent_family_coverage_rows(
        grouped,
        family_payload,
        min(INTENT_FAMILY_COVERAGE_SAMPLE_ROWS, current_target),
    )
    coverage_indices = set(coverage.index)
    ordinary_pool = grouped.drop(index=list(coverage_indices), errors="ignore")
    ordinary_target = max(0, min(current_target, len(grouped)) - len(coverage))
    ordinary_sample = representative_sample(
        ordinary_pool,
        min(ordinary_target, len(ordinary_pool)),
    )
    sample = pd.concat([coverage, ordinary_sample], axis=0)
    sample = sample.sort_values(["Priority", "Occurrences"], ascending=False)
    sample["Model Label"] = ""
    sample["Model Confidence"] = ""
    sample["Model Notes"] = ""
    sample["Knowledge Source"] = "current representative sample"
    if "Coverage Family" in sample.columns:
        family_coverage = sample["Coverage Family"].fillna("").astype(str).str.strip().ne("")
        sample.loc[family_coverage, "Knowledge Source"] = "current family coverage sample"
    else:
        sample["Coverage Family"] = ""
    sample["Source Job"] = ""
    sample = sample[
        [
            "Phrase",
            "Occurrences",
            "Search Volume",
            "Source Row",
            "Model Label",
            "Model Confidence",
            "Model Notes",
            "Knowledge Source",
            "Source Job",
            "Coverage Family",
        ]
    ]
    if not prior.empty:
        prior["Occurrences"] = 0
        prior["Search Volume"] = 0
        prior["Source Row"] = ""
        prior["Coverage Family"] = ""
        prior = prior[sample.columns]
        sample = pd.concat([prior, sample], ignore_index=True)
        sample["_phrase_key"] = sample["Phrase"].map(normalize)
        sample["_prior"] = sample["Model Label"].astype(str).str.strip().ne("")
        sample = (
            sample.sort_values("_prior", ascending=False)
            .drop_duplicates("_phrase_key", keep="first")
            .drop(columns=["_phrase_key", "_prior"])
            .head(sample_size)
        )

    # Reused knowledge predates this job and has no job-local identifier.
    # Create IDs only after combining both sources, otherwise selecting the
    # prior table with the current sample columns raises "Sample ID not in index".
    sample.insert(0, "Sample ID", [f"row-{index:06d}" for index in range(1, len(sample) + 1)])

    job_name_slug = slugify(job_name or f"{input_path.stem}-{topic}")
    workflow_job_id = f"seo-{uuid.uuid4().hex[:16]}"
    job_dir = JOBS_DIR / job_name_slug
    job_dir.mkdir(parents=True, exist_ok=True)
    sample_path = job_dir / "model_labels.xlsx"
    sample.to_excel(sample_path, sheet_name="Model labels", index=False)

    prior_profile = load_topic_profile(reused_topic_key) if reused_topic_key else None
    config = prior_profile or {
        "job_name": job_name_slug,
        "workflow_job_id": workflow_job_id,
        "topic": {
            "description": topic,
            "relevant_seeds": [],
            "garbage_seeds": [],
            "positive_markers": [],
            "garbage_markers": [],
            "minimum_relevance": 0.35,
            "garbage_margin": 0.05,
        },
        "intent": {
            "commercial_seeds": [],
            "informational_seeds": [],
            "commercial_markers": [],
            "informational_markers": [],
            "weak_question_markers": [],
            "family_rules": {
                "commercial": [],
                "informational": [],
                "neutral": [],
            },
            "semantic_margin": 0.03,
            "review_margin": 0.04,
            "informational_decision_margin": 0.12,
            "strong_informational_decision_margin": 0.02,
            "weak_question_informational_margin": 0.02,
            "family_override_tolerance": 0.05,
            "default": "commercial",
        },
        "intent_policy": {
            "commercial_prototypes": [],
            "implicit_commercial_prototypes": [],
            "informational_prototypes": [],
            "synthetic_weight": 0.75,
            "informational_evidence_margin": 0.005,
            "strength": 0.12,
            "minimum_similarity": 0.55,
        },
        "relevance": {
            "relevant_prototypes": [],
            "garbage_prototypes": [],
            "synthetic_weight": 1.0,
            "garbage_threshold": 0.45,
            "quarantine_margin": 0.10,
            "outlier_garbage_threshold": 0.80,
            "minimum_garbage_precision": 0.80,
            "maximum_relevant_false_positive_rate": 0.05,
            "tfidf_weight": 0.35,
            "intent_tfidf_weight": 1.25,
            "max_garbage_class_weight": 8.0,
            "audit_batch_size": 50,
            "min_garbage_examples": 30,
            "max_audit_batches": 6,
        },
        "clustering": {
            "batch_size": 256,
            "n_neighbors": 30,
            "min_cluster_size": 20,
            "umap_dimensions": 20,
            "review_threshold": 0.35,
        },
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_device": "auto",
        "embedding_batch_size": 256,
        "embedding_prefix": "query: ",
        "phrase_column": phrase_column,
        "frequency_column": frequency_column,
        "source_sheet": source_sheet,
        "cluster_stop_words": [],
    }
    config["job_name"] = job_name_slug
    config["workflow_job_id"] = workflow_job_id
    # A saved topic profile carries semantic labels and policy, not a pinned
    # model version. New jobs use the current project default; a user may still
    # override this job-local field after preparation.
    config["embedding_model"] = DEFAULT_EMBEDDING_MODEL
    config["topic"]["description"] = topic
    config["phrase_column"] = phrase_column
    config["frequency_column"] = frequency_column
    config["knowledge_reuse"] = {
        "topic_key": reused_topic_key,
        "prior_examples": int(len(prior)),
        "policy": "Auxiliary prior only. Revalidate against the current sample and scope.",
    }
    (job_dir / "job_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (job_dir / "intent_family_candidates.json").write_text(
        json.dumps(family_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inspection = {
        "job_id": job_name_slug,
        "workflow_job_id": workflow_job_id,
        "topic": topic,
        "input_file": str(input_path),
        "rows_in_source": (
            stream_stats["rows_in_source"] if stream_stats else int(len(source))
        ),
        "non_empty_phrases": (
            stream_stats["non_empty_phrases"] if stream_stats else int(len(raw))
        ),
        "unique_normalized_phrases": (
            unique_source_count
        ),
        "exact_or_normalized_duplicates": (
            None if stream_stats else int(len(raw) - len(grouped))
        ),
        "streaming_prepare": bool(stream_stats),
        "phrase_column": phrase_column,
        "frequency_column": frequency_column,
        "source_sheet": source_sheet,
        "sample_rows": int(len(sample)),
        "intent_family_candidates": int(len(family_payload.get("families", []))),
        "prior_knowledge_rows": int(len(prior)),
        "reused_topic_key": reused_topic_key,
        "execution_strategy": (
            "direct_model_labeling" if direct_model_labeling else "active_learning"
        ),
        "next_step": "The model must label the compact workflow batches; seeds are derived automatically from real labels.",
    }
    (job_dir / "inspection.json").write_text(
        json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return job_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an SEO job and representative model-labeling sample.")
    parser.add_argument("input")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--sample-size", type=int, default=800)
    parser.add_argument("--phrase-column")
    parser.add_argument("--frequency-column")
    parser.add_argument("--job-name")
    parser.add_argument(
        "--reuse-topic",
        default="auto",
        help="Knowledge topic key/description, auto for the current topic, or none",
    )
    parser.add_argument("--prior-limit", type=int, default=150)
    args = parser.parse_args()
    job_dir = prepare_job(
        args.input,
        args.topic,
        args.sample_size,
        args.phrase_column,
        args.frequency_column,
        args.job_name,
        args.reuse_topic,
        args.prior_limit,
    )
    print(f"Job directory: {job_dir}")
    print(f"Model labeling workbook: {job_dir / 'model_labels.xlsx'}")
    print(f"Job config: {job_dir / 'job_config.json'}")


if __name__ == "__main__":
    main()
