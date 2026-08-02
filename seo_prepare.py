"""Prepare a representative semantic-core sample for model judgement."""

from __future__ import annotations

import argparse
import heapq
import json
import re
import uuid
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


def normalize(value: object) -> str:
    text = re.sub(r"[^a-zа-я0-9]+", " ", str(value).lower().replace("ё", "е"))
    return re.sub(r"\s+", " ", text).strip()


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
    workbook = pd.ExcelFile(path)
    requested = normalize(phrase_column) if phrase_column else ""
    candidates = {"поисковый запрос", "фраза", "keyword", "query", "запрос"}
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
) -> tuple[pd.DataFrame, str, str | None, dict[str, int]]:
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
    return candidates, phrase_name, frequency_name, {
        "rows_in_source": source_rows,
        "non_empty_phrases": non_empty,
        # It is an upper bound, deliberately used to select the safe large path.
        "unique_phrase_upper_bound": non_empty,
    }


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
    sample_size: int = 500,
    phrase_column: str | None = None,
    frequency_column: str | None = None,
    job_name: str | None = None,
    reuse_topic: str | None = "auto",
    prior_limit: int = 150,
) -> Path:
    input_path = resolve_input(input_value)
    stream_stats: dict[str, int] | None = None
    if input_path.suffix.lower() in {".csv", ".tsv"}:
        source, phrase_column, frequency_column, stream_stats = streaming_csv_candidates(
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
    reused_topic_key = None
    prior = pd.DataFrame()
    if reuse_topic and reuse_topic.lower() != "none":
        reused_topic_key = (
            topic_key(topic) if reuse_topic.lower() == "auto" else topic_key(reuse_topic)
        )
        prior = load_balanced_examples(
            reused_topic_key, min(prior_limit, max(0, sample_size // 3))
        )
    current_target = max(50, sample_size - len(prior))
    sample = representative_sample(grouped, min(current_target, len(grouped)))
    sample = sample.sort_values(["Priority", "Occurrences"], ascending=False)
    sample["Model Label"] = ""
    sample["Model Confidence"] = ""
    sample["Model Notes"] = ""
    sample["Knowledge Source"] = "current representative sample"
    sample["Source Job"] = ""
    sample.insert(0, "Sample ID", [f"row-{index:06d}" for index in range(1, len(sample) + 1)])
    sample = sample[
        [
            "Sample ID",
            "Phrase",
            "Occurrences",
            "Search Volume",
            "Source Row",
            "Model Label",
            "Model Confidence",
            "Model Notes",
            "Knowledge Source",
            "Source Job",
        ]
    ]
    if not prior.empty:
        prior["Occurrences"] = 0
        prior["Search Volume"] = 0
        prior["Source Row"] = ""
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
            "garbage_probability": 0.75,
        },
        "intent": {
            "commercial_seeds": [],
            "informational_seeds": [],
            "commercial_markers": [],
            "informational_markers": [],
            "semantic_margin": 0.03,
            "review_margin": 0.04,
            "default": "commercial",
        },
        "clustering": {
            "batch_size": 64,
            "n_neighbors": 30,
            "min_cluster_size": 20,
            "umap_dimensions": 20,
            "review_threshold": 0.35,
        },
        "embedding_model": "intfloat/multilingual-e5-small",
        "phrase_column": phrase_column,
        "frequency_column": frequency_column,
        "source_sheet": source_sheet,
        "cluster_stop_words": [],
    }
    config["job_name"] = job_name_slug
    config["workflow_job_id"] = workflow_job_id
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
            stream_stats["unique_phrase_upper_bound"] if stream_stats else int(len(grouped))
        ),
        "exact_or_normalized_duplicates": (
            None if stream_stats else int(len(raw) - len(grouped))
        ),
        "streaming_prepare": bool(stream_stats),
        "phrase_column": phrase_column,
        "frequency_column": frequency_column,
        "source_sheet": source_sheet,
        "sample_rows": int(len(sample)),
        "prior_knowledge_rows": int(len(prior)),
        "reused_topic_key": reused_topic_key,
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
    parser.add_argument("--sample-size", type=int, default=500)
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
