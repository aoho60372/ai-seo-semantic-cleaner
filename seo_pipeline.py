"""Run model-directed SEO classification, validation, and semantic clustering."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from seo_embeddings import encode_queries, load_local_embedding_model
from seo_io import iter_delimited_chunks, read_delimited_table

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "files"
OUTPUT_DIR = PROJECT_DIR / "outputs"
VALID_LABELS = {"commercial", "informational", "garbage"}
EXCEL_MAX_DATA_ROWS = 1_048_575
LABEL_ALIASES = {
    "коммерческий": "commercial",
    "коммерческие": "commercial",
    "commercial": "commercial",
    "transactional": "commercial",
    "информационный": "informational",
    "информационные": "informational",
    "informational": "informational",
    "information": "informational",
    "мусор": "garbage",
    "garbage": "garbage",
    "irrelevant": "garbage",
}


def normalize(value: object) -> str:
    text = re.sub(r"[^a-zа-я0-9]+", " ", str(value).lower().replace("ё", "е"))
    return re.sub(r"\s+", " ", text).strip()


def read_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return read_delimited_table(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def resolve_input(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    candidate = INPUT_DIR / Path(value).name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Input not found: {value}. Expected it in {INPUT_DIR}.")


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
    text = [column for column in frame.columns if frame[column].dtype == "object"]
    if not text:
        raise ValueError("No text phrase column found.")
    return text[0]


def build_phrase_table(
    source: pd.DataFrame,
    phrase_column: str,
    frequency_column: str | None,
) -> pd.DataFrame:
    raw = source[phrase_column].dropna().astype(str).str.strip()
    raw = raw[raw.ne("")]
    table = pd.DataFrame({"Phrase": raw, "Source Row": raw.index + 2})
    table["Normalized"] = table["Phrase"].map(normalize)
    if frequency_column and frequency_column in source.columns:
        table["Search Volume"] = pd.to_numeric(
            source.loc[raw.index, frequency_column], errors="coerce"
        ).fillna(0).to_numpy()
    else:
        table["Search Volume"] = 0.0
    return (
        table.groupby("Normalized", as_index=False)
        .agg(
            Phrase=("Phrase", "first"),
            Occurrences=("Phrase", "size"),
            **{"Search Volume": ("Search Volume", "max"), "Source Row": ("Source Row", "first")},
        )
    )


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for section in ("topic", "intent", "clustering"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    if config["intent"].get("default") not in {"commercial", "informational"}:
        raise ValueError("intent.default must be commercial or informational")
    return config


def load_model_labels(path: Path | None) -> pd.DataFrame:
    if not path or not path.is_file():
        return pd.DataFrame(columns=["Phrase", "Label", "Model Notes"])
    frame = read_table(path, "Model labels" if path.suffix.lower() == ".xlsx" else 0)
    column_lookup = {normalize(column): column for column in frame.columns}
    phrase_column = column_lookup.get("phrase") or column_lookup.get("фраза")
    label_column = column_lookup.get("model label") or column_lookup.get("label") or column_lookup.get("метка")
    notes_column = column_lookup.get("model notes") or column_lookup.get("notes")
    if not phrase_column or not label_column:
        raise ValueError("Label file must contain Phrase and Model Label columns.")
    labels = pd.DataFrame(
        {
            "Phrase": frame[phrase_column].astype(str).str.strip(),
            "Raw Label": frame[label_column].astype(str).str.strip().str.lower(),
            "Model Notes": frame[notes_column].fillna("").astype(str) if notes_column else "",
        }
    )
    labels["Label"] = labels["Raw Label"].map(LABEL_ALIASES)
    labels = labels[labels["Label"].isin(VALID_LABELS) & labels["Phrase"].ne("")]
    labels["Normalized"] = labels["Phrase"].map(normalize)
    return labels.drop_duplicates("Normalized", keep="last")


def marker_pattern(marker: str) -> re.Pattern[str]:
    escaped = re.escape(normalize(marker)).replace(r"\*", r"[a-zа-я0-9]*")
    return re.compile(rf"(?:^|\s){escaped}(?:$|\s)", re.IGNORECASE)


def marker_hits(texts: list[str], markers: list[str] | None) -> np.ndarray:
    patterns = [marker_pattern(marker) for marker in (markers or []) if normalize(marker)]
    return np.asarray([any(pattern.search(text) for pattern in patterns) for text in texts])


def seed_scores(
    vectors: np.ndarray,
    seeds: list[str],
    model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    if not seeds:
        return np.zeros(len(vectors), dtype=float)
    seed_vectors = encode_queries(model, [normalize(seed) for seed in seeds], batch_size)
    return np.max(vectors @ np.asarray(seed_vectors).T, axis=1)


def seed_classification(
    phrase_table: pd.DataFrame,
    vectors: np.ndarray,
    model: SentenceTransformer,
    config: dict[str, Any],
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic, intent = config["topic"], config["intent"]
    required = {
        "topic.relevant_seeds": topic.get("relevant_seeds"),
        "intent.commercial_seeds": intent.get("commercial_seeds"),
        "intent.informational_seeds": intent.get("informational_seeds"),
    }
    missing = [name for name, values in required.items() if not values]
    if missing:
        raise ValueError(
            "Not enough labeled examples and missing seed lists: " + ", ".join(missing)
        )

    texts = phrase_table["Normalized"].tolist()
    relevant = seed_scores(vectors, topic["relevant_seeds"], model, batch_size)
    garbage = seed_scores(vectors, topic.get("garbage_seeds", []), model, batch_size)
    commercial = seed_scores(vectors, intent["commercial_seeds"], model, batch_size)
    informational = seed_scores(vectors, intent["informational_seeds"], model, batch_size)
    positive_hits = marker_hits(texts, topic.get("positive_markers"))
    garbage_hits = marker_hits(texts, topic.get("garbage_markers"))
    commercial_hits = marker_hits(texts, intent.get("commercial_markers"))
    informational_hits = marker_hits(texts, intent.get("informational_markers"))
    minimum_relevance = float(topic.get("minimum_relevance", 0.35))
    garbage_margin = float(topic.get("garbage_margin", 0.05))
    intent_margin = float(intent.get("semantic_margin", 0.03))
    labels, bases, confidences = [], [], []

    for index in range(len(phrase_table)):
        if garbage_hits[index] and not positive_hits[index]:
            label, basis, confidence = "garbage", "Task garbage marker", 1.0
        elif (
            relevant[index] < minimum_relevance
            and garbage[index] >= relevant[index] + garbage_margin
            and not positive_hits[index]
        ):
            label, basis = "garbage", "Seed-based topic relevance"
            confidence = min(1.0, garbage[index] - relevant[index] + 0.5)
        elif commercial_hits[index] and not informational_hits[index]:
            label, basis, confidence = "commercial", "Task commercial marker", 1.0
        elif informational_hits[index] and not commercial_hits[index]:
            label, basis, confidence = "informational", "Task informational marker", 1.0
        elif commercial[index] >= informational[index] + intent_margin:
            label, basis = "commercial", "Commercial seed similarity"
            confidence = min(1.0, commercial[index] - informational[index] + 0.5)
        elif informational[index] >= commercial[index] + intent_margin:
            label, basis = "informational", "Informational seed similarity"
            confidence = min(1.0, informational[index] - commercial[index] + 0.5)
        else:
            label, basis, confidence = intent["default"], "Ambiguous seed result", 0.5
        labels.append(label)
        bases.append(basis)
        confidences.append(confidence)

    result = phrase_table.copy()
    result["Intent"] = labels
    result["Classification confidence"] = confidences
    result["Decision basis"] = bases
    result["Topic relevance"] = relevant
    return result, pd.DataFrame([("Mode", "Seed fallback")], columns=["Metric", "Value"])


def supervised_classification(
    phrase_table: pd.DataFrame,
    vectors: np.ndarray,
    labels: pd.DataFrame,
    config: dict[str, Any],
    model: SentenceTransformer,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = labels["Label"].value_counts()
    required_intents = {"commercial", "informational"}
    if not required_intents.issubset(set(counts.index)) or len(labels) < 30:
        raise ValueError(
            "At least 30 valid model labels covering commercial and informational are required. "
            "Garbage is optional and must never be invented."
        )

    train_vectors = encode_queries(model, labels["Normalized"], batch_size)
    train_labels = labels["Label"].to_numpy()
    validation_rows: list[tuple[str, Any]] = [
        ("Mode", "Supervised model labels"),
        ("Labeled phrases", len(labels)),
    ]
    can_validate = len(labels) >= 40 and counts.min() >= 4
    if can_validate:
        x_train, x_test, y_train, y_test = train_test_split(
            train_vectors,
            train_labels,
            test_size=0.25,
            random_state=42,
            stratify=train_labels,
        )
        validation_model = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ).fit(x_train, y_train)
        prediction = validation_model.predict(x_test)
        report = classification_report(y_test, prediction, output_dict=True, zero_division=0)
        validation_rows.append(("Validation samples", len(y_test)))
        validation_rows.append(("Validation accuracy", report["accuracy"]))
        for label in sorted(set(y_test)):
            validation_rows.extend(
                [
                    (f"{label} precision", report[label]["precision"]),
                    (f"{label} recall", report[label]["recall"]),
                    (f"{label} f1", report[label]["f1-score"]),
                ]
            )
        ordered = sorted(set(y_test))
        matrix = confusion_matrix(y_test, prediction, labels=ordered)
        for row_index, actual in enumerate(ordered):
            for column_index, predicted in enumerate(ordered):
                validation_rows.append(
                    (f"Confusion actual={actual} predicted={predicted}", int(matrix[row_index, column_index]))
                )
    else:
        validation_rows.append(("Validation status", "Insufficient per-class labels for holdout validation"))

    classifier = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ).fit(train_vectors, train_labels)
    probabilities = classifier.predict_proba(vectors)
    classes = classifier.classes_
    class_index = {label: index for index, label in enumerate(classes)}
    prediction = classes[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    garbage_probability = float(config["topic"].get("garbage_probability", 0.75))

    if "garbage" in class_index:
        garbage_scores = probabilities[:, class_index["garbage"]]
        for index in range(len(prediction)):
            if prediction[index] == "garbage" and garbage_scores[index] < garbage_probability:
                alternatives = [
                    label for label in ("commercial", "informational") if label in class_index
                ]
                if alternatives:
                    prediction[index] = max(
                        alternatives, key=lambda label: probabilities[index, class_index[label]]
                    )
                    confidence[index] = probabilities[index, class_index[prediction[index]]]

    exact_labels = labels.set_index("Normalized")["Label"].to_dict()
    bases = ["Classifier trained on model labels"] * len(phrase_table)
    for index, normalized in enumerate(phrase_table["Normalized"]):
        if normalized in exact_labels:
            prediction[index] = exact_labels[normalized]
            confidence[index] = 1.0
            bases[index] = "Exact model-reviewed label"

    result = phrase_table.copy()
    result["Intent"] = prediction
    result["Classification confidence"] = confidence
    result["Decision basis"] = bases
    for label in ("commercial", "informational", "garbage"):
        result[f"P({label})"] = (
            probabilities[:, class_index[label]] if label in class_index else 0.0
        )
    result["Topic relevance"] = 1.0 - result["P(garbage)"]
    return result, pd.DataFrame(validation_rows, columns=["Metric", "Value"])


def cluster_labels(phrases: list[str], labels: np.ndarray, stop_words: list[str]) -> list[str]:
    result = ["Unclustered" if label == -1 else "" for label in labels]
    for label in sorted(set(labels)):
        if label == -1:
            continue
        positions = np.flatnonzero(labels == label)
        corpus = [normalize(phrases[position]) for position in positions]
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=[normalize(word) for word in stop_words],
                max_features=300,
            )
            matrix = vectorizer.fit_transform(corpus)
            terms = vectorizer.get_feature_names_out()
            scores = np.asarray(matrix.mean(axis=0)).ravel()
            best = [terms[position] for position in scores.argsort()[::-1] if len(terms[position]) > 2][:3]
            name = " / ".join(best) or f"Cluster {label}"
        except ValueError:
            name = f"Cluster {label}"
        for position in positions:
            result[position] = name
    return result


def cluster_subset(
    subset: pd.DataFrame,
    vectors: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = subset.copy().reset_index(drop=True)
    if len(result) < 8:
        result["Cluster"] = "Small cluster"
        result["Cluster probability"] = 1.0
        return result
    # UMAP and HDBSCAN are needed only for the final clustering operation.
    # Delaying their imports keeps workflow-control commands fast.
    import hdbscan
    import umap

    clustering = config["clustering"]
    neighbors = min(int(clustering.get("n_neighbors", 30)), len(result) - 1)
    dimensions = min(int(clustering.get("umap_dimensions", 20)), len(result) - 2)
    minimum_size = min(
        int(clustering.get("min_cluster_size", 20)), max(2, len(result) // 2)
    )
    reduced = umap.UMAP(
        n_neighbors=max(2, neighbors),
        n_components=max(2, dimensions),
        metric="cosine",
        random_state=42,
        low_memory=True,
    ).fit_transform(vectors)
    fitted = hdbscan.HDBSCAN(
        min_cluster_size=max(2, minimum_size),
        min_samples=max(2, minimum_size // 2),
        cluster_selection_method="eom",
    ).fit(reduced)
    result["Cluster"] = cluster_labels(
        result["Phrase"].tolist(), fitted.labels_, config.get("cluster_stop_words", [])
    )
    result["Cluster probability"] = fitted.probabilities_
    return result


def build_cluster_summary(frame: pd.DataFrame, intent: str) -> pd.DataFrame:
    rows = []
    for cluster, group in frame.groupby("Cluster", dropna=False):
        ranked = group.sort_values(
            ["Search Volume", "Cluster probability", "Occurrences"],
            ascending=False,
        )
        examples = "\n".join(ranked["Phrase"].head(8).tolist())
        rows.append(
            {
                "Intent": intent,
                "Cluster": cluster,
                "Phrases": len(group),
                "Total Search Volume": float(group["Search Volume"].sum()),
                "Average Classification Confidence": float(
                    group["Classification confidence"].mean()
                ),
                "Average Cluster Probability": float(group["Cluster probability"].mean()),
                "Representative Phrases": examples,
                "Model Cluster Name": "",
                "Model Action": "",
                "Model Notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Total Search Volume", "Phrases"], ascending=False
    )


def style_workbook(writer: pd.ExcelWriter) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for cells in worksheet.columns:
            values = [len(str(cell.value or "")) for cell in cells[:200]]
            width = min(max(values, default=10) + 2, 60)
            worksheet.column_dimensions[cells[0].column_letter].width = width
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)


def flatten_config(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_config(child, child_prefix))
    elif isinstance(value, list):
        rows.append((prefix, " | ".join(str(item) for item in value)))
    else:
        rows.append((prefix, str(value)))
    return rows


def build_human_review_queue(
    combined: pd.DataFrame,
    garbage: pd.DataFrame,
    classification_threshold: float,
    cluster_threshold: float,
) -> pd.DataFrame:
    """Build a bounded review queue with coverage, not only uncertainty.

    A classifier can be confidently wrong.  The queue therefore reserves space
    for high-value phrases, cluster representatives and class-boundary rows in
    addition to low-confidence predictions.
    """
    pieces: list[pd.DataFrame] = []

    def with_reason(frame: pd.DataFrame, reason: str, order: int, limit: int) -> None:
        if frame.empty:
            return
        selected = frame.head(limit).copy()
        selected["Review reason"] = reason
        selected["_review_order"] = order
        pieces.append(selected)

    def ranked(frame: pd.DataFrame, columns: list[str], ascending: list[bool]) -> pd.DataFrame:
        result = frame.copy()
        result["_review_volume"] = pd.to_numeric(
            result.get("Search Volume", 0), errors="coerce"
        ).fillna(0)
        return result.sort_values(columns, ascending=ascending, kind="stable")

    confidence = pd.to_numeric(combined.get("Classification confidence", 0), errors="coerce").fillna(0)
    cluster_probability = pd.to_numeric(combined.get("Cluster probability", 0), errors="coerce").fillna(0)
    uncertain = combined[
        confidence.lt(classification_threshold)
        | combined.get("Cluster", pd.Series("Unclustered", index=combined.index)).eq("Unclustered")
        | cluster_probability.lt(cluster_threshold)
    ].copy()
    uncertain["_confidence"] = confidence.loc[uncertain.index]
    uncertain["_cluster_probability"] = cluster_probability.loc[uncertain.index]
    uncertain["_review_volume"] = pd.to_numeric(uncertain.get("Search Volume", 0), errors="coerce").fillna(0)
    uncertain = uncertain.sort_values(
        ["_confidence", "_cluster_probability", "_review_volume"],
        ascending=[True, True, False],
        kind="stable",
    )
    with_reason(uncertain, "Low confidence, unclustered, or weak cluster", 1, 175)

    high_risk_garbage = ranked(garbage, ["Topic relevance", "_review_volume"], [False, False])
    with_reason(high_risk_garbage, "Garbage near the topic-relevance boundary", 2, 100)

    if {"P(commercial)", "P(informational)"}.issubset(combined.columns):
        boundary = combined.copy()
        boundary["_intent_margin"] = (
            pd.to_numeric(boundary["P(commercial)"], errors="coerce").fillna(0)
            - pd.to_numeric(boundary["P(informational)"], errors="coerce").fillna(0)
        ).abs()
        boundary = ranked(boundary, ["_intent_margin", "_review_volume"], [True, False])
        with_reason(boundary, "Commercial/informational probability boundary", 3, 75)

    high_value = ranked(combined, ["_review_volume"], [False])
    with_reason(high_value, "High-priority phrase", 4, 75)

    if not combined.empty and {"Intent", "Cluster"}.issubset(combined.columns):
        cluster_sizes = (
            combined[combined["Cluster"].ne("Unclustered")]
            .groupby(["Intent", "Cluster"], dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        representatives: list[pd.DataFrame] = []
        for (intent, cluster), _ in cluster_sizes.items():
            candidates = combined[combined["Intent"].eq(intent) & combined["Cluster"].eq(cluster)].copy()
            candidates["_review_volume"] = pd.to_numeric(candidates.get("Search Volume", 0), errors="coerce").fillna(0)
            representatives.append(candidates.nlargest(2, "_review_volume"))
            if sum(len(part) for part in representatives) >= 75:
                break
        if representatives:
            with_reason(pd.concat(representatives, ignore_index=True), "Representative high-priority phrase in cluster", 5, 75)

    if not pieces:
        return pd.DataFrame()
    review = pd.concat(pieces, ignore_index=True, sort=False)
    key = review.get("Normalized", review.get("Phrase", pd.Series("", index=review.index))).astype(str)
    reason_map = (
        pd.DataFrame({"key": key, "reason": review["Review reason"]})
        .groupby("key", sort=False)["reason"]
        .agg(lambda values: " | ".join(dict.fromkeys(values)))
    )
    review["_review_key"] = key
    review = review.sort_values(["_review_order", "_review_volume"], ascending=[True, False], kind="stable")
    review = review.drop_duplicates("_review_key", keep="first").head(500).copy()
    review["Review reason"] = review["_review_key"].map(reason_map)
    review = review.drop(columns=[column for column in review.columns if column.startswith("_review_") or column in {"_confidence", "_cluster_probability", "_intent_margin"}])
    for column in ("Correct Intent", "Correct Cluster", "Reviewer Notes"):
        review[column] = ""
    return review


def run_pipeline(
    input_value: str,
    config_path: Path,
    labels_path: Path | None,
    output_path: Path | None,
) -> Path:
    input_path = resolve_input(input_value)
    config = load_config(config_path)
    source = read_table(input_path, config.get("source_sheet", 0))
    phrase_column = choose_phrase_column(source, config.get("phrase_column"))
    phrase_table = build_phrase_table(source, phrase_column, config.get("frequency_column"))
    batch_size = int(config.get("embedding_batch_size", config["clustering"].get("batch_size", 256)))
    model, execution_device, model_path = load_local_embedding_model(config)
    vectors = encode_queries(model, phrase_table["Normalized"], batch_size, str(config.get("embedding_prefix", "query: ")))

    model_labels = load_model_labels(labels_path)
    try:
        classified, validation = supervised_classification(
            phrase_table, vectors, model_labels, config, model, batch_size
        )
    except ValueError as error:
        classified, validation = seed_classification(
            phrase_table, vectors, model, config, batch_size
        )
        validation = pd.concat(
            [
                validation,
                pd.DataFrame([("Supervised fallback reason", str(error))], columns=["Metric", "Value"]),
            ],
            ignore_index=True,
        )

    intent_frames: dict[str, pd.DataFrame] = {}
    summaries = []
    for intent, title in (
        ("commercial", "Commercial clusters"),
        ("informational", "Informational clusters"),
    ):
        positions = np.flatnonzero(classified["Intent"].to_numpy() == intent)
        clustered = cluster_subset(classified.iloc[positions], vectors[positions], config)
        intent_frames[title] = clustered
        if len(clustered):
            summaries.append(build_cluster_summary(clustered, intent))
    garbage = classified[classified["Intent"].eq("garbage")].copy()
    combined = pd.concat(intent_frames.values(), ignore_index=True)
    review_threshold = float(config["clustering"].get("review_threshold", 0.35))
    classification_threshold = float(config.get("classification_review_threshold", 0.65))
    review = build_human_review_queue(
        combined, garbage, classification_threshold, review_threshold
    )

    cluster_summary = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(columns=["Intent", "Cluster", "Phrases"])
    )
    report = pd.DataFrame(
        [
            ("Job", config.get("job_name", config_path.stem)),
            ("Workflow Job ID", config.get("workflow_job_id", "")),
            ("Topic", config["topic"].get("description", "")),
            ("Input file", input_path.name),
            ("Execution device", execution_device),
            ("Embedding model", str(model_path)),
            ("Unique phrases", len(phrase_table)),
            ("Commercial phrases", len(intent_frames["Commercial clusters"])),
            ("Informational phrases", len(intent_frames["Informational clusters"])),
            ("Garbage phrases", len(garbage)),
            ("Garbage share", len(garbage) / max(len(phrase_table), 1)),
            ("Unclustered phrases", int(combined["Cluster"].eq("Unclustered").sum())),
            ("Human review candidates", len(review)),
            ("Model-labeled examples", len(model_labels)),
        ],
        columns=["Metric", "Value"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or OUTPUT_DIR / f"{input_path.stem}_clustered.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        intent_frames["Commercial clusters"].to_excel(
            writer, sheet_name="Commercial clusters", index=False
        )
        intent_frames["Informational clusters"].to_excel(
            writer, sheet_name="Informational clusters", index=False
        )
        garbage.to_excel(writer, sheet_name="Garbage", index=False)
        cluster_summary.to_excel(writer, sheet_name="Cluster review", index=False)
        review.to_excel(writer, sheet_name="Human review", index=False)
        report.to_excel(writer, sheet_name="Quality report", index=False)
        validation.to_excel(writer, sheet_name="Validation metrics", index=False)
        pd.DataFrame(flatten_config(config), columns=["Parameter", "Value"]).to_excel(
            writer, sheet_name="Run configuration", index=False
        )
        style_workbook(writer)
    return output_path


def fit_large_classifier(
    labels: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int,
) -> LogisticRegression:
    counts = labels["Label"].value_counts()
    if len(labels) < 30 or not {"commercial", "informational"}.issubset(counts.index):
        raise ValueError(
            "run-large requires at least 30 real model labels, including commercial and informational."
        )
    vectors = encode_queries(model, labels["Normalized"], batch_size)
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(
        vectors, labels["Label"].to_numpy()
    )


def large_part_path(parts_dir: Path, number: int) -> Path:
    return parts_dir / f"part-{number:06d}.csv.gz"


def stable_sample_mask(values: pd.Series, divisor: int = 40) -> np.ndarray:
    return np.asarray(
        [int(hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).hexdigest(), 16) % divisor == 0 for value in values]
    )


def _excel_value(value: object) -> object:
    """Convert pandas missing values to cells that Excel can store."""
    return None if pd.isna(value) else value


def export_large_result_workbook(
    parts_dir: Path,
    report_path: Path,
    cluster_summary: pd.DataFrame,
    uncertain_review: pd.DataFrame,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, int]:
    """Create the same review workbook contract as the standard pipeline.

    Classified rows are streamed from the compressed parts, so exporting a
    large core does not require loading the whole result into RAM.  Excel has a
    hard per-sheet row limit; exceptionally large intent groups continue on
    numbered sheets while preserving the familiar first-sheet names.
    """
    parts = sorted(parts_dir.glob("part-*.csv.gz"))
    if not parts:
        raise FileNotFoundError(f"No classified parts found in {parts_dir}")
    columns = list(pd.read_csv(parts[0], compression="gzip", nrows=0).columns)
    workbook = Workbook(write_only=True)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    sheet_states: dict[str, tuple[object, int, int]] = {}
    total_by_intent = {"commercial": 0, "informational": 0, "garbage": 0}
    sheet_names = {
        "commercial": "Commercial clusters",
        "informational": "Informational clusters",
        "garbage": "Garbage",
    }

    def create_sheet(base_name: str, part_number: int, headers: list[str]) -> object:
        title = base_name if part_number == 1 else f"{base_name} {part_number}"
        sheet = workbook.create_sheet(title)
        header_cells = []
        for header in headers:
            cell = WriteOnlyCell(sheet, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            header_cells.append(cell)
        sheet.append(header_cells)
        sheet.freeze_panes = "A2"
        for index, header in enumerate(headers, start=1):
            sheet.column_dimensions[get_column_letter(index)].width = min(max(len(header) + 2, 14), 42)
        return sheet

    def target_sheet(intent: str) -> tuple[object, int, int]:
        state = sheet_states.get(intent)
        if state is None or state[1] >= EXCEL_MAX_DATA_ROWS:
            part_number = 1 if state is None else state[2] + 1
            state = (create_sheet(sheet_names[intent], part_number, columns), 0, part_number)
            sheet_states[intent] = state
        return state

    for part in parts:
        frame = pd.read_csv(part, compression="gzip")
        for intent in total_by_intent:
            subset = frame[frame["Intent"].eq(intent)]
            if subset.empty:
                continue
            for row in subset[columns].itertuples(index=False, name=None):
                sheet, written, sheet_number = target_sheet(intent)
                sheet.append([_excel_value(value) for value in row])
                sheet_states[intent] = (sheet, written + 1, sheet_number)
                total_by_intent[intent] += 1

    for intent in total_by_intent:
        if intent not in sheet_states:
            sheet_states[intent] = (create_sheet(sheet_names[intent], 1, columns), 0, 1)

    review = uncertain_review.copy()
    for column in ("Correct Intent", "Correct Cluster", "Reviewer Notes"):
        if column not in review.columns:
            review[column] = ""
    review_columns = list(review.columns)

    def append_small_sheet(name: str, frame: pd.DataFrame) -> None:
        sheet = create_sheet(name, 1, list(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            sheet.append([_excel_value(value) for value in row])
        sheet.auto_filter.ref = f"A1:{get_column_letter(max(len(frame.columns), 1))}{len(frame) + 1}"

    append_small_sheet("Cluster review", cluster_summary)
    append_small_sheet("Human review", review[review_columns])
    quality = pd.DataFrame(
        [
            ("Job", config.get("job_name", "")),
            ("Workflow Job ID", config.get("workflow_job_id", "")),
            ("Topic", config.get("topic", {}).get("description", "")),
            ("Input file", metadata.get("input", "")),
            ("Execution device", metadata.get("execution_device", "CPU")),
            ("Processing mode", "Large / streamed"),
            ("Unique phrases", metadata.get("classified_rows_per_chunk_deduplicated", 0)),
            ("Commercial phrases", total_by_intent["commercial"]),
            ("Informational phrases", total_by_intent["informational"]),
            ("Garbage phrases", total_by_intent["garbage"]),
            ("Human review candidates", len(review)),
        ],
        columns=["Metric", "Value"],
    )
    append_small_sheet("Quality report", quality)
    validation = pd.DataFrame(
        [
            ("Classifier", "LogisticRegression trained on model-reviewed labels"),
            ("Clustering", metadata.get("clustering", "")),
            ("Parts", metadata.get("parts", len(parts))),
        ],
        columns=["Metric", "Value"],
    )
    append_small_sheet("Validation metrics", validation)
    append_small_sheet("Run configuration", pd.DataFrame(flatten_config(config), columns=["Parameter", "Value"]))

    for intent, (sheet, written, _) in sheet_states.items():
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{written + 1}"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(report_path)
    return total_by_intent


def run_large_pipeline(
    input_value: str,
    config_path: Path,
    labels_path: Path,
    output_dir: Path,
    chunk_size: int = 50000,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Chunked classification plus sample-based centroid clustering for large cores."""

    input_path = resolve_input(input_value)
    if input_path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("run-large accepts CSV or TSV only. Export Excel first; do not create a converted copy beside the input.")
    if chunk_size < 1000:
        raise ValueError("chunk-size must be at least 1000.")
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    header = read_delimited_table(input_path, nrows=0)
    phrase_column = choose_phrase_column(header, config.get("phrase_column"))
    frequency_column = config.get("frequency_column")
    if frequency_column not in header.columns:
        frequency_column = None
    batch_size = int(config.get("embedding_batch_size", config["clustering"].get("batch_size", 256)))
    model, execution_device, model_path = load_local_embedding_model(config)
    embedding_prefix = str(config.get("embedding_prefix", "query: "))
    cache_dir = output_dir / ".embedding-cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    classifier = fit_large_classifier(load_model_labels(labels_path), model, batch_size)
    classes = classifier.classes_
    class_index = {label: index for index, label in enumerate(classes)}
    total_rows = 0
    representatives: dict[str, list[pd.DataFrame]] = {"commercial": [], "informational": []}
    use_columns = [phrase_column] + ([frequency_column] if frequency_column else [])
    reader = iter_delimited_chunks(input_path, chunksize=chunk_size, usecols=use_columns)
    for part_number, chunk in enumerate(reader, start=1):
        raw = chunk[phrase_column].dropna().astype(str).str.strip()
        raw = raw[raw.ne("")]
        table = pd.DataFrame({"Phrase": raw, "Source Row": raw.index + 2})
        table["Normalized"] = table["Phrase"].map(normalize)
        if frequency_column:
            table["Search Volume"] = pd.to_numeric(chunk.loc[raw.index, frequency_column], errors="coerce").fillna(0).to_numpy()
        else:
            table["Search Volume"] = 0.0
        table = table.drop_duplicates("Normalized", keep="first")
        if table.empty:
            continue
        vectors = encode_queries(model, table["Normalized"], batch_size, embedding_prefix)
        np.save(cache_dir / f"part-{part_number:06d}.npy", vectors)
        probabilities = classifier.predict_proba(vectors)
        prediction = classes[np.argmax(probabilities, axis=1)]
        table["Intent"] = prediction
        table["Classification confidence"] = np.max(probabilities, axis=1)
        for label in ("commercial", "informational", "garbage"):
            table[f"P({label})"] = probabilities[:, class_index[label]] if label in class_index else 0.0
        table["Topic relevance"] = 1.0 - table["P(garbage)"]
        table["Part"] = part_number
        table.to_csv(large_part_path(parts_dir, part_number), index=False, compression="gzip", encoding="utf-8")
        total_rows += len(table)
        mask = stable_sample_mask(table["Normalized"])
        for intent in representatives:
            chosen = table.loc[mask & table["Intent"].eq(intent), ["Phrase", "Normalized", "Search Volume", "Intent"]]
            if not chosen.empty:
                representatives[intent].append(chosen.head(1000))
        print(json.dumps({"event": "part_completed", "part": part_number, "rows": int(len(table))}, ensure_ascii=False))

    centroids: dict[str, tuple[np.ndarray, list[str]]] = {}
    cluster_rows: list[pd.DataFrame] = []
    for intent, samples in representatives.items():
        sample = pd.concat(samples, ignore_index=True).drop_duplicates("Normalized").head(10000) if samples else pd.DataFrame()
        if len(sample) < 8:
            continue
        vectors = encode_queries(model, sample["Normalized"], batch_size, embedding_prefix)
        cluster_count = min(max(2, int(np.sqrt(len(sample) / 2))), 80, len(sample))
        fitted = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, batch_size=min(2048, len(sample)), n_init="auto").fit(vectors)
        names = cluster_labels(sample["Phrase"].tolist(), fitted.labels_, config.get("cluster_stop_words", []))
        centroid_names = []
        for cluster_id in range(cluster_count):
            first = next((names[index] for index, value in enumerate(fitted.labels_) if value == cluster_id), f"Cluster {cluster_id}")
            centroid_names.append(first)
        centroids[intent] = (np.asarray(fitted.cluster_centers_), centroid_names)
        sample["Cluster"] = names
        sample["Cluster probability"] = np.max(vectors @ fitted.cluster_centers_.T, axis=1)
        cluster_rows.append(sample)

    review_candidates: dict[str, list[pd.DataFrame]] = {
        "uncertain": [],
        "garbage_boundary": [],
        "intent_boundary": [],
        "high_value": [],
        "cluster_representative": [],
    }
    summaries: list[dict[str, Any]] = []
    for part in sorted(parts_dir.glob("part-*.csv.gz")):
        frame = pd.read_csv(part, compression="gzip")
        cache_path = cache_dir / f"{part.stem.replace('.csv', '')}.npy"
        vectors = np.load(cache_path, mmap_mode="r") if cache_path.is_file() else None
        for intent, (centroid_vectors, names) in centroids.items():
            mask = frame["Intent"].eq(intent)
            if not mask.any():
                continue
            if vectors is None:
                vectors = encode_queries(model, frame["Normalized"], batch_size, embedding_prefix)
            similarity = vectors[mask.to_numpy()] @ centroid_vectors.T
            indexes = np.argmax(similarity, axis=1)
            frame.loc[mask, "Cluster"] = [names[index] for index in indexes]
            frame.loc[mask, "Cluster probability"] = np.max(similarity, axis=1)
        if "Cluster" not in frame.columns:
            frame["Cluster"] = "Unclustered"
        else:
            frame["Cluster"] = frame["Cluster"].fillna("Unclustered")
        if "Cluster probability" not in frame.columns:
            frame["Cluster probability"] = 0.0
        else:
            frame["Cluster probability"] = pd.to_numeric(frame["Cluster probability"], errors="coerce").fillna(0.0)
        classification_threshold = float(config.get("classification_review_threshold", 0.65))
        cluster_threshold = float(config.get("clustering", {}).get("review_threshold", 0.35))
        low = frame[
            frame["Classification confidence"].lt(classification_threshold)
            | frame["Cluster"].eq("Unclustered")
            | frame["Cluster probability"].lt(cluster_threshold)
        ].copy()
        if not low.empty:
            low["Review reason"] = "Low confidence, unclustered, or weak cluster"
            review_candidates["uncertain"].append(
                low.sort_values(["Classification confidence", "Search Volume"], ascending=[True, False]).head(200)
            )
        garbage_boundary = frame[frame["Intent"].eq("garbage")].nlargest(100, "Topic relevance").copy()
        if not garbage_boundary.empty:
            garbage_boundary["Review reason"] = "Garbage near the topic-relevance boundary"
            review_candidates["garbage_boundary"].append(garbage_boundary)
        if {"P(commercial)", "P(informational)"}.issubset(frame.columns):
            intent_boundary = frame.copy()
            intent_boundary["_intent_margin"] = (intent_boundary["P(commercial)"] - intent_boundary["P(informational)"]).abs()
            intent_boundary = intent_boundary.nsmallest(75, "_intent_margin")
            intent_boundary["Review reason"] = "Commercial/informational probability boundary"
            review_candidates["intent_boundary"].append(intent_boundary)
        high_value = frame.nlargest(75, "Search Volume").copy()
        if not high_value.empty:
            high_value["Review reason"] = "High-priority phrase"
            review_candidates["high_value"].append(high_value)
        clustered = frame[frame["Cluster"].ne("Unclustered")]
        if not clustered.empty:
            cluster_rows = (
                clustered.sort_values("Search Volume", ascending=False)
                .groupby(["Intent", "Cluster"], dropna=False, group_keys=False)
                .head(1)
                .head(75)
                .copy()
            )
            cluster_rows["Review reason"] = "Representative high-priority phrase in cluster"
            review_candidates["cluster_representative"].append(cluster_rows)
        frame.to_csv(part, index=False, compression="gzip", encoding="utf-8")
        grouped = frame.groupby(["Intent", "Cluster"], dropna=False).agg(Phrases=("Phrase", "size"), Total_Search_Volume=("Search Volume", "sum"), Average_Confidence=("Classification confidence", "mean")).reset_index()
        summaries.extend(grouped.to_dict("records"))

    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.groupby(["Intent", "Cluster"], dropna=False).agg(Phrases=("Phrases", "sum"), Total_Search_Volume=("Total_Search_Volume", "sum"), Average_Confidence=("Average_Confidence", "mean")).reset_index().sort_values("Total_Search_Volume", ascending=False)
    review_parts: list[pd.DataFrame] = []
    queue_limits = {
        "uncertain": 175,
        "garbage_boundary": 100,
        "intent_boundary": 75,
        "high_value": 75,
        "cluster_representative": 75,
    }
    for kind, limit in queue_limits.items():
        batches = review_candidates[kind]
        if not batches:
            continue
        candidates = pd.concat(batches, ignore_index=True)
        if kind == "uncertain":
            candidates = candidates.sort_values(["Classification confidence", "Search Volume"], ascending=[True, False])
        elif kind == "intent_boundary":
            candidates = candidates.sort_values(["_intent_margin", "Search Volume"], ascending=[True, False])
        elif kind == "garbage_boundary":
            candidates = candidates.sort_values(["Topic relevance", "Search Volume"], ascending=[False, False])
        else:
            candidates = candidates.sort_values("Search Volume", ascending=False)
        review_parts.append(candidates.head(limit))
    uncertain_frame = pd.concat(review_parts, ignore_index=True, sort=False) if review_parts else pd.DataFrame()
    if not uncertain_frame.empty:
        review_key = uncertain_frame.get("Normalized", uncertain_frame["Phrase"]).astype(str)
        reason_map = (
            pd.DataFrame({"key": review_key, "reason": uncertain_frame["Review reason"]})
            .groupby("key", sort=False)["reason"]
            .agg(lambda values: " | ".join(dict.fromkeys(values)))
        )
        uncertain_frame["_review_key"] = review_key
        uncertain_frame = uncertain_frame.drop_duplicates("_review_key", keep="first").head(500).copy()
        uncertain_frame["Review reason"] = uncertain_frame["_review_key"].map(reason_map)
        uncertain_frame = uncertain_frame.drop(columns=[column for column in ("_review_key", "_intent_margin") if column in uncertain_frame])
    uncertain_path = output_dir / "uncertain_review.csv"
    # These two exports are intended for people as well as scripts.  The BOM
    # lets desktop Excel on Windows recognise Russian UTF-8 text correctly.
    uncertain_frame.to_csv(uncertain_path, index=False, encoding="utf-8-sig")
    summary_path = output_dir / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    source_stem = Path(source_name).stem if source_name else input_path.stem
    report_path = output_dir / f"{source_stem}_clustered.xlsx"
    manifest = {
        "status": "completed",
        "workflow_job_id": config.get("workflow_job_id", ""),
        "input": source_name or str(input_path.resolve()),
        "format": "partitioned_csv_gzip",
        "parts_directory": str(parts_dir.resolve()),
        "parts": len(list(parts_dir.glob("part-*.csv.gz"))),
        "classified_rows_per_chunk_deduplicated": total_rows,
        "chunk_size": chunk_size,
        "embedding_model": str(model_path),
        "execution_device": execution_device,
        "embedding_batch_size": batch_size,
        "clustering": "MiniBatchKMeans on deterministic representative samples; all rows assigned to nearest sample centroid.",
        "uncertain_review": str(uncertain_path.resolve()),
        "cluster_summary": str(summary_path.resolve()),
        "excel_result": str(report_path.resolve()),
        # Kept for compatibility with existing workflow state readers.
        "excel_summary": str(report_path.resolve()),
    }
    manifest["intent_counts"] = export_large_result_workbook(
        parts_dir, report_path, summary, uncertain_frame, config, manifest
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised SEO classification and clustering.")
    parser.add_argument("input")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_pipeline(args.input, args.config, args.labels, args.output)
    print(f"Output workbook: {result.resolve()}")


if __name__ == "__main__":
    main()
