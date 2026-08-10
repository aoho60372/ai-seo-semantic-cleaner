"""Single entry point for the model-directed SEO workflow."""

from __future__ import annotations

import argparse
import csv
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from seo_knowledge import (
    forget_topic,
    knowledge_status,
    save_topic_profile,
    store_examples,
    topic_key,
)


PROJECT_DIR = Path(__file__).resolve().parent
JOBS_DIR = PROJECT_DIR / "jobs"
OUTPUT_DIR = PROJECT_DIR / "outputs"
# Keep light workflow commands independent from seo_pipeline.  The agent sends
# only these canonical English labels, but accepting common aliases preserves
# compatibility with previously labelled workbooks.
LABEL_ALIASES = {
    "commercial": "commercial",
    "transactional": "commercial",
    "\u043a\u043e\u043c\u043c\u0435\u0440\u0447\u0435\u0441\u043a\u0438\u0439": "commercial",
    "\u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u0439": "informational",
    "\u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u0435": "informational",
    "informational": "informational",
    "information": "informational",
    "\u043c\u0443\u0441\u043e\u0440": "garbage",
    "garbage": "garbage",
    "irrelevant": "garbage",
}
REQUIRED_RESULT_SHEETS = {
    "Commercial clusters",
    "Informational clusters",
    "Garbage",
    "Cluster review",
    "Human review",
    "Quality report",
    "Validation metrics",
    "Run configuration",
}
ALLOWED_ROOT_PYTHON_FILES = {
    "seo_io.py",
    "seo_embeddings.py",
    "seo_knowledge.py",
    "seo_pipeline.py",
    "seo_prepare.py",
    "seo_workflow.py",
}
VALID_LABELS = {"commercial", "informational", "garbage"}
MIN_REVIEWED_LABELS = 100
MIN_EXAMPLES_PER_CLASS = 10
MIN_SEEDS_PER_CLASS = 5
HIGH_PRIORITY_REVIEW_ROWS = 50
GARBAGE_SHARE_WARNING = 0.35
HIGH_GARBAGE_ADDITIONAL_LABELS = 400
INITIAL_LABEL_BATCH_SIZE = 50
ACTIVE_REVIEW_BATCH_SIZE = 50
DEFAULT_LARGE_THRESHOLD = 100_000
# Hermes invokes terminal commands through a Bash-compatible transport before
# Windows PowerShell receives them.  The quotes preserve the leading backslash:
# without them Bash turns ``.\seo.ps1`` into ``.seo.ps1`` and PowerShell then
# reports a misleading, mojibake-prone "file not found" error.
HERMES_COMMAND = r'powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ".\seo.ps1"'
RUN_LOCK_NAME = ".seo_run.lock.json"


def resolve_input(value: str) -> Path:
    """Resolve an input without loading the ML pipeline."""
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    candidate = PROJECT_DIR / "files" / Path(value).name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Input not found: {value}. Expected it in {PROJECT_DIR / 'files'}.")


def state_file(job_dir: Path) -> Path:
    return job_dir / "state.json"


def read_state(job_dir: Path) -> dict[str, object]:
    path = state_file(job_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(job_dir: Path, stage: str, **updates: object) -> dict[str, object]:
    state = read_state(job_dir)
    state.update(updates)
    state["stage"] = stage
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = state_file(job_dir).with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(state_file(job_dir))
    return state


def ensure_workflow_job_id(job_dir: Path) -> str:
    """Return the immutable ASCII ID that links a result workbook to its job."""
    config_path = job_dir / "job_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read job_config.json for workflow identity: {exc}") from exc
    current = str(config.get("workflow_job_id", "")).strip()
    if not re.fullmatch(r"seo-[a-f0-9]{16}", current):
        current = f"seo-{uuid.uuid4().hex[:16]}"
        config["workflow_job_id"] = current
        temporary = config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(config_path)
    state = read_state(job_dir)
    if state.get("workflow_job_id") != current:
        write_state(job_dir, str(state.get("stage", "unknown")), workflow_job_id=current)
    return current


def find_job_by_workflow_id(workflow_job_id: str) -> Path:
    matches: list[Path] = []
    for candidate in JOBS_DIR.iterdir() if JOBS_DIR.is_dir() else []:
        config_path = candidate / "job_config.json"
        if not candidate.is_dir() or not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(config.get("workflow_job_id", "")) == workflow_job_id:
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No SEO job matches Workflow Job ID {workflow_job_id}.")
    raise ValueError(f"Workflow Job ID {workflow_job_id} is ambiguous; no import was made.")


def compact_status(status: dict[str, object]) -> dict[str, object]:
    quality = status.get("label_quality", {})
    state = status.get("state", {})
    return {
        "job_directory": status.get("job_directory"),
        "stage": state.get("stage", "unknown") if isinstance(state, dict) else "unknown",
        "labels_reviewed": quality.get("valid_labeled_rows", 0) if isinstance(quality, dict) else 0,
        "labeled_counts": status.get("labeled_counts", {}),
        "ready_for_supervised_run": status.get("ready_for_supervised_run", False),
        "blocking_errors": list(status.get("blocking_errors", []))[:3],
        "warnings": list(status.get("warnings", []))[:3],
    }


def next_action(job_dir: Path) -> dict[str, object]:
    """Return the sole safe workflow decision point for a Bionic agent."""
    status = job_status(job_dir)
    state = read_state(job_dir)
    stage = str(state.get("stage", "unknown"))
    lock_path = job_dir / RUN_LOCK_NAME
    active_lock = read_run_lock(lock_path) if lock_path.is_file() else {}
    if active_lock and lock_process_is_active(active_lock):
        # A run may outlive the terminal tool's foreground timeout.  Returning
        # a command here would make an agent start a duplicate process or burn
        # its context on polling an already-running job.
        return {
            "status": "running",
            "stage": "run",
            "pid": active_lock.get("pid"),
            "started_at": active_lock.get("started_at"),
            "instruction": "A workflow run is already active. Do not run, retry, wait, poll, or call next. Return immediately and rely on the terminal completion notification.",
        }
    inspection = status.get("inspection", {})
    quality = status.get("label_quality", {})
    reviewed = int(quality.get("valid_labeled_rows", 0)) if isinstance(quality, dict) else 0
    total = int(quality.get("sample_rows", 0)) if isinstance(quality, dict) else 0
    garbage_share = float(quality.get("garbage_share", 0.0)) if isinstance(quality, dict) else 0.0
    errors = list(status.get("blocking_errors", []))
    root_files = list(status.get("unexpected_python_files", []))
    if root_files:
        return {
            "status": "blocked",
            "stage": stage,
            "reason": "Unexpected root Python files block the workflow.",
            "files": root_files,
            "instruction": "Report this block to the user. Do not rename, run, or bypass these files.",
        }
    if stage == "quality_review":
        return {
            "status": "stop",
            "stage": stage,
            "reason": "A run already completed. Do not search for another output or rerun cleaning.",
            "last_run": state.get("last_run_path") or state.get("last_large_run"),
            "instruction": "Report the stored result path and request user review or a reviewed workbook for feedback.",
        }
    if stage in {"knowledge_saved", "learned"}:
        return {
            "status": "stop",
            "stage": stage,
            "reason": "Reviewed labels and topic knowledge were saved. Do not rerun cleaning unless the user explicitly requests recalculation.",
            "knowledge_topic_key": state.get("knowledge_topic_key"),
        }
    if stage == "blocked":
        return {
            "status": "blocked",
            "stage": stage,
            "reason": state.get("run_error") or state.get("blocked_reason") or "The workflow recorded a blocking failure.",
            "instruction": "Report this block once. Do not retry the same command.",
        }
    if stage == "finalized":
        return {
            "status": "stop",
            "stage": stage,
            "final": state.get("final"),
            "instruction": "Report the final path. No further processing is required.",
        }
    if not inspection:
        return {
            "status": "blocked",
            "stage": stage,
            "reason": "Job has no valid inspection.json.",
            "instruction": "Report the block; do not create a replacement script.",
        }
    if reviewed < min(100, total):
        return {
            "status": "continue",
            "stage": "initial_labeling",
            "action": "label_initial_batch",
            "labels_reviewed": reviewed,
            "command": f'{HERMES_COMMAND} sample --job "{job_dir.name}" --offset 0 --limit {min(INITIAL_LABEL_BATCH_SIZE, total - reviewed)} --quiet',
            "after_labels": f'{HERMES_COMMAND} apply-labels-inline --job "{job_dir.name}" --labels "SampleID|label|confidence;..." --quiet',
        }
    non_label_errors = [error for error in errors if not error.startswith("Only ") and "Class " not in error and "priority rows" not in error]
    if non_label_errors:
        return {
            "status": "continue",
            "stage": "configuration",
            "action": "configure_job",
            "job_config": str((job_dir / "job_config.json").resolve()),
            "blocking_errors": non_label_errors[:5],
            "instruction": "Update only the requested topic-specific fields in job_config.json, then call next again.",
        }
    normal_target = min(200, total)
    adaptive_target = (
        min(HIGH_GARBAGE_ADDITIONAL_LABELS, total)
        if garbage_share > GARBAGE_SHARE_WARNING
        else normal_target
    )
    if reviewed < adaptive_target:
        return {
            "status": "continue",
            "stage": "active_review",
            "action": (
                "label_high_garbage_boundary_batch"
                if adaptive_target > normal_target
                else "label_uncertain_batch"
            ),
            "labels_reviewed": reviewed,
            "target_labels": adaptive_target,
            "command": f'{HERMES_COMMAND} review --job "{job_dir.name}" --limit {min(ACTIVE_REVIEW_BATCH_SIZE, adaptive_target - reviewed)} --quiet',
            "after_labels": f'{HERMES_COMMAND} apply-labels-inline --job "{job_dir.name}" --labels "SampleID|label|confidence;..." --quiet',
        }
    if not status.get("ready_for_supervised_run", False):
        return {
            "status": "blocked",
            "stage": stage,
            "reason": "Job readiness checks still fail.",
            "blocking_errors": errors[:5],
        }
    return {
        "status": "continue",
        "stage": "run",
        "action": "run_auto",
        "command": f'{HERMES_COMMAND} run-auto "{Path(str(inspection.get("input_file", ""))).name}" --job "{job_dir.name}" --quiet',
    }


def recorded_next_action(job_dir: Path) -> dict[str, object]:
    """Persist the terminal state of `next` for Hermes' SEO autopilot guard."""
    action = next_action(job_dir)
    status = str(action.get("status", "blocked"))
    state = read_state(job_dir)
    persisted_stage = "run" if status == "running" else str(state.get("stage", action.get("stage", "unknown")))
    write_state(
        job_dir,
        persisted_stage,
        autopilot_last_status=status,
    )
    return action


def print_result(value: dict[str, object], quiet: bool = False) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=None if quiet else 2))


def quiet_library_logs() -> None:
    for name in ("huggingface_hub", "transformers", "sentence_transformers", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)


def resolve_job(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()
    candidate = JOBS_DIR / value
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Job directory not found: {value}")


def find_job_for_input(input_value: str) -> Path | None:
    wanted = Path(input_value).name.lower()
    matches: list[tuple[float, Path]] = []
    for candidate in (JOBS_DIR.iterdir() if JOBS_DIR.is_dir() else []):
        inspection = candidate / "inspection.json"
        if not candidate.is_dir() or not inspection.is_file():
            continue
        try:
            data = json.loads(inspection.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if Path(str(data.get("input_file", ""))).name.lower() == wanted:
            matches.append((inspection.stat().st_mtime, candidate))
    return max(matches, default=(0.0, None), key=lambda item: item[0])[1]


def normalized_column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {
        re.sub(r"\s+", " ", str(column).strip().lower()): column
        for column in frame.columns
    }


def configured_values(config: dict[str, object], section: str, key: str) -> list[str]:
    raw_section = config.get(section, {})
    if not isinstance(raw_section, dict):
        return []
    values = raw_section.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def label_quality_status(
    labels: pd.DataFrame,
    config: dict[str, object],
) -> tuple[dict[str, object], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    lookup = normalized_column_lookup(labels)
    required_columns = {"phrase", "model label", "model confidence"}
    missing_columns = sorted(required_columns - set(lookup))
    if missing_columns:
        errors.append(
            "model_labels.xlsx is missing required columns: "
            + ", ".join(missing_columns)
        )
        return {}, errors, warnings

    phrase_column = lookup["phrase"]
    label_column = lookup["model label"]
    confidence_column = lookup["model confidence"]
    notes_column = lookup.get("model notes")

    phrases = labels[phrase_column].fillna("").astype(str).str.strip()
    raw_labels = labels[label_column].fillna("").astype(str).str.strip().str.lower()
    normalized_labels = raw_labels.map(LABEL_ALIASES)
    labeled_mask = normalized_labels.notna() & phrases.ne("")
    invalid_mask = raw_labels.ne("") & normalized_labels.isna()
    confidence = pd.to_numeric(labels[confidence_column], errors="coerce")
    valid_confidence = confidence.between(0.0, 1.0, inclusive="both")
    confidence_complete = labeled_mask & valid_confidence

    labeled_count = int(labeled_mask.sum())
    sample_rows = int(phrases.ne("").sum())
    required_labeled = min(MIN_REVIEWED_LABELS, sample_rows)
    invalid_count = int(invalid_mask.sum())
    missing_or_invalid_confidence = int((labeled_mask & ~valid_confidence).sum())
    counts = {
        label: int((normalized_labels == label).sum())
        for label in sorted(VALID_LABELS)
    }

    if labeled_count < required_labeled:
        errors.append(
            f"Only {labeled_count} valid labels; at least {required_labeled} are required."
        )
    if invalid_count:
        errors.append(
            f"{invalid_count} labels are invalid; use only commercial, informational, or garbage."
        )
    if missing_or_invalid_confidence:
        errors.append(
            f"{missing_or_invalid_confidence} labeled rows have missing or invalid Model Confidence; "
            "use a numeric value from 0 to 1."
        )
    for label in ("commercial", "informational"):
        if counts[label] < MIN_EXAMPLES_PER_CLASS:
            errors.append(
                f"Class {label} has {counts[label]} examples; at least "
                f"{MIN_EXAMPLES_PER_CLASS} model-reviewed examples are required."
            )
    if counts["garbage"] and counts["garbage"] < MIN_EXAMPLES_PER_CLASS:
        warnings.append(
            f"Class garbage has only {counts['garbage']} real examples. It is optional; "
            "do not invent labels merely to meet a quota."
        )

    priority_columns = [
        column
        for column in (lookup.get("search volume"), lookup.get("occurrences"))
        if column is not None
    ]
    high_priority_unreviewed = 0
    if priority_columns:
        priority = pd.Series(0.0, index=labels.index)
        for column in priority_columns:
            priority += pd.to_numeric(labels[column], errors="coerce").fillna(0)
        high_priority_index = priority.nlargest(
            min(HIGH_PRIORITY_REVIEW_ROWS, len(labels))
        ).index
        high_priority_unreviewed = int(
            (~(labeled_mask & valid_confidence).loc[high_priority_index]).sum()
        )
        if high_priority_unreviewed:
            errors.append(
                f"{high_priority_unreviewed} of the top "
                f"{min(HIGH_PRIORITY_REVIEW_ROWS, len(labels))} priority rows "
                "lack a valid label or confidence."
            )

    garbage_share = counts["garbage"] / max(labeled_count, 1)
    if garbage_share > GARBAGE_SHARE_WARNING:
        warnings.append(
            f"Garbage share in reviewed labels is {garbage_share:.1%}. "
            "A dirty semantic core is allowed, but manually audit false garbage before saving knowledge."
        )

    positive_markers = configured_values(config, "topic", "positive_markers")
    positive_markers += configured_values(config, "topic", "relevant_seeds")
    normalized_markers = {
        re.sub(r"\s+", " ", marker.lower().replace("ё", "е")).strip()
        for marker in positive_markers
        if marker.strip()
    }
    garbage_phrases = (
        phrases[normalized_labels.eq("garbage")]
        .str.lower()
        .str.replace("ё", "е", regex=False)
    )
    conflicting_garbage: list[str] = []
    for phrase in garbage_phrases:
        if any(marker in phrase for marker in normalized_markers):
            conflicting_garbage.append(phrase)
        if len(conflicting_garbage) >= 10:
            break
    if conflicting_garbage:
        errors.append(
            f"{len(conflicting_garbage)} sampled garbage labels conflict with configured "
            "positive markers or relevant seeds. Examples: "
            + " | ".join(conflicting_garbage[:5])
        )

    low_confidence_without_notes = 0
    if notes_column:
        notes = labels[notes_column].fillna("").astype(str).str.strip()
        low_confidence_without_notes = int(
            (labeled_mask & valid_confidence & confidence.lt(0.7) & notes.eq("")).sum()
        )
        if low_confidence_without_notes:
            warnings.append(
                f"{low_confidence_without_notes} low-confidence labels have no Model Notes."
            )

    metrics = {
        "sample_rows": sample_rows,
        "valid_labeled_rows": labeled_count,
        "required_labeled_rows": required_labeled,
        "labeled_counts": counts,
        "invalid_labels": invalid_count,
        "confidence_complete_rows": int(confidence_complete.sum()),
        "missing_or_invalid_confidence": missing_or_invalid_confidence,
        "high_priority_unreviewed": high_priority_unreviewed,
        "garbage_share": round(garbage_share, 6),
        "positive_marker_garbage_conflicts": len(conflicting_garbage),
        "low_confidence_without_notes": low_confidence_without_notes,
    }
    return metrics, errors, warnings


def job_status(job_dir: Path) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    inspection_path = job_dir / "inspection.json"
    inspection = (
        json.loads(inspection_path.read_text(encoding="utf-8"))
        if inspection_path.is_file()
        else {}
    )
    if not inspection:
        errors.append("inspection.json is missing or empty.")

    config_path = job_dir / "job_config.json"
    config: dict[str, object] = {}
    if config_path.is_file():
        try:
            loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded_config, dict):
                config = loaded_config
            else:
                errors.append("job_config.json must contain a JSON object.")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"job_config.json cannot be read: {exc}")
    else:
        errors.append("job_config.json is missing.")

    seed_requirements = (
        ("topic", "relevant_seeds"),
        ("intent", "commercial_seeds"),
        ("intent", "informational_seeds"),
    )
    seed_counts: dict[str, int] = {}
    for section, key in seed_requirements:
        count = len(configured_values(config, section, key))
        seed_counts[f"{section}.{key}"] = count
        if count < MIN_SEEDS_PER_CLASS:
            errors.append(
                f"{section}.{key} has {count} values; at least "
                f"{MIN_SEEDS_PER_CLASS} diverse seeds are required."
            )

    garbage_seed_count = len(configured_values(config, "topic", "garbage_seeds"))
    seed_counts["topic.garbage_seeds"] = garbage_seed_count

    labels_path = job_dir / "model_labels.xlsx"
    label_metrics: dict[str, object] = {}
    if labels_path.is_file():
        try:
            labels = pd.read_excel(labels_path, sheet_name="Model labels")
            label_metrics, label_errors, label_warnings = label_quality_status(
                labels, config
            )
            errors.extend(label_errors)
            warnings.extend(label_warnings)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"model_labels.xlsx cannot be validated: {exc}")
    else:
        errors.append("model_labels.xlsx is missing.")

    unexpected_python_files = sorted(
        path.name
        for path in PROJECT_DIR.glob("*.py")
        if path.name not in ALLOWED_ROOT_PYTHON_FILES
    )
    if unexpected_python_files:
        errors.append(
            "Unexpected Python files exist in the project root: "
            + ", ".join(unexpected_python_files)
            + ". Use only the maintained workflow scripts; put disposable data under the job tmp directory."
        )

    return {
        "job_directory": str(job_dir),
        "inspection": inspection,
        "labeled_counts": label_metrics.get("labeled_counts", {}),
        "config_exists": config_path.is_file(),
        "seed_counts": seed_counts,
        "label_quality": label_metrics,
        "unexpected_python_files": unexpected_python_files,
        "blocking_errors": errors,
        "warnings": warnings,
        "ready_for_supervised_run": not errors,
        "state": read_state(job_dir),
    }


def load_label_sheet(job_dir: Path) -> pd.DataFrame:
    path = job_dir / "model_labels.xlsx"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path, sheet_name="Model labels")
    if "Sample ID" not in frame.columns:
        frame.insert(0, "Sample ID", [f"row-{index:06d}" for index in range(1, len(frame) + 1)])
    required = {"Sample ID", "Phrase", "Model Label", "Model Confidence"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("model_labels.xlsx is missing columns: " + ", ".join(missing))
    for column in ("Sample ID", "Phrase", "Model Label", "Model Confidence", "Model Notes"):
        if column in frame.columns:
            frame[column] = frame[column].astype(object)
    return frame


def label_summary(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    labels = frame["Model Label"].fillna("").astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    phrases = frame["Phrase"].fillna("").astype(str).str.strip()
    return labels, phrases


def bootstrap_seeds_from_real_labels(job_dir: Path, frame: pd.DataFrame) -> dict[str, int]:
    """Fill only missing seed slots from model-reviewed, real source phrases.

    Seeds are an aid for the CPU pipeline, not input the user must provide.
    They must never be invented before the model has seen the semantic core.
    Existing manually configured values stay first and are never replaced.
    """
    config_path = job_dir / "job_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(config, dict):
        return {}

    labels, phrases = label_summary(frame)

    def unique_phrases(mask: pd.Series) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for phrase in phrases[mask]:
            normalized = re.sub(r"\s+", " ", str(phrase).lower().replace("ё", "е")).strip()
            if normalized and normalized not in seen:
                result.append(str(phrase).strip())
                seen.add(normalized)
        return result

    candidates = {
        ("topic", "relevant_seeds"): unique_phrases(labels.isin(["commercial", "informational"])),
        ("intent", "commercial_seeds"): unique_phrases(labels.eq("commercial")),
        ("intent", "informational_seeds"): unique_phrases(labels.eq("informational")),
        ("topic", "garbage_seeds"): unique_phrases(labels.eq("garbage")),
    }
    changed = False
    counts: dict[str, int] = {}
    for (section, key), values in candidates.items():
        section_data = config.setdefault(section, {})
        if not isinstance(section_data, dict):
            continue
        current = configured_values(config, section, key)
        seen = {
            re.sub(r"\s+", " ", value.lower().replace("ё", "е")).strip()
            for value in current
        }
        merged = list(current)
        for value in values:
            normalized = re.sub(r"\s+", " ", value.lower().replace("ё", "е")).strip()
            if normalized not in seen:
                merged.append(value)
                seen.add(normalized)
            if len(merged) >= MIN_SEEDS_PER_CLASS:
                break
        if merged != current:
            section_data[key] = merged
            changed = True
        counts[f"{section}.{key}"] = len(merged)

    if changed:
        temporary = config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(config_path)
    return counts


def sample_rows(job_dir: Path, offset: int, limit: int, only_unlabeled: bool = True) -> dict[str, object]:
    if offset < 0 or limit < 1 or limit > 50:
        raise ValueError("offset must be non-negative and limit must be from 1 to 50.")
    frame = load_label_sheet(job_dir)
    labels, phrases = label_summary(frame)
    candidates = frame[phrases.ne("") & (labels.isna() if only_unlabeled else pd.Series(True, index=frame.index))].copy()
    if "Search Volume" in candidates:
        candidates["_priority"] = pd.to_numeric(candidates["Search Volume"], errors="coerce").fillna(0)
    else:
        candidates["_priority"] = 0.0
    candidates = candidates.sort_values(["_priority", "Sample ID"], ascending=[False, True])
    view = candidates.iloc[offset : offset + limit]
    rows = [
        {
            "id": str(row["Sample ID"]),
            "phrase": str(row["Phrase"]),
            "priority": round(float(row["_priority"]), 3),
        }
        for _, row in view.iterrows()
    ]
    return {
        "stage": read_state(job_dir).get("stage", "initial_labeling"),
        "offset": offset,
        "limit": limit,
        "total_unlabeled": int(len(candidates)),
        "rows": rows,
    }


def parse_inline_labels(value: str) -> list[dict[str, object]]:
    """Parse a deliberately small, phrase-free label transport for Bionic."""
    payload: list[dict[str, object]] = []
    for raw_item in value.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        fields = [field.strip() for field in item.split("|")]
        if len(fields) != 3:
            raise ValueError("Each inline label must be SampleID|label|confidence, separated by semicolons.")
        payload.append({"id": fields[0], "label": fields[1], "confidence": fields[2], "notes": ""})
    if not payload:
        raise ValueError("At least one inline label is required.")
    return payload


def apply_label_payload(job_dir: Path, payload: object) -> dict[str, object]:
    if not isinstance(payload, list):
        raise ValueError("Label JSON must be an array of {id, label, confidence, notes?} objects.")
    frame = load_label_sheet(job_dir)
    id_index = {str(value): index for index, value in frame["Sample ID"].items()}
    seen: set[str] = set()
    new_labels = 0
    updated_labels = 0
    unchanged_labels = 0
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Every label item must be an object.")
        sample_id = str(item.get("id", "")).strip()
        raw_label = str(item.get("label", "")).strip().lower()
        label = LABEL_ALIASES.get(raw_label)
        if not sample_id or sample_id not in id_index:
            raise ValueError(f"Unknown Sample ID: {sample_id or '<empty>'}")
        if sample_id in seen:
            raise ValueError(f"Duplicate Sample ID in label JSON: {sample_id}")
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label for {sample_id}: {raw_label}")
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid confidence for {sample_id}") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence for {sample_id} must be from 0 to 1.")
        row = id_index[sample_id]
        raw_previous_label = frame.loc[row, "Model Label"]
        previous_label = LABEL_ALIASES.get("" if pd.isna(raw_previous_label) else str(raw_previous_label).strip().lower())
        previous_confidence = pd.to_numeric(pd.Series([frame.loc[row, "Model Confidence"]]), errors="coerce").iloc[0]
        raw_previous_notes = frame.loc[row, "Model Notes"]
        previous_notes = "" if pd.isna(raw_previous_notes) else str(raw_previous_notes).strip()
        notes = str(item.get("notes", "")).strip()
        if previous_label == label and pd.notna(previous_confidence) and abs(float(previous_confidence) - confidence) < 1e-9 and previous_notes == notes:
            unchanged_labels += 1
            seen.add(sample_id)
            continue
        if previous_label is None:
            new_labels += 1
        else:
            updated_labels += 1
        frame.loc[row, "Model Label"] = label
        frame.loc[row, "Model Confidence"] = confidence
        frame.loc[row, "Model Notes"] = notes
        seen.add(sample_id)
    applied = new_labels + updated_labels
    if applied:
        temporary = job_dir / "tmp" / "model_labels.updated.xlsx"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        frame.to_excel(temporary, sheet_name="Model labels", index=False)
        temporary.replace(job_dir / "model_labels.xlsx")
    labels, _ = label_summary(frame)
    seed_counts = bootstrap_seeds_from_real_labels(job_dir, frame)
    stage = "active_review" if int(labels.notna().sum()) >= 100 else "initial_labeling"
    state = write_state(
        job_dir,
        stage,
        labels_reviewed=int(labels.notna().sum()),
        remaining_review_rows=int(labels.isna().sum()),
    )
    return {
        "status": "labels_applied" if applied else "no_progress",
        "new": new_labels,
        "updated": updated_labels,
        "unchanged": unchanged_labels,
        "applied": applied,
        "seed_counts": seed_counts,
        "state": state,
    }


def apply_labels(job_dir: Path, source: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Label JSON cannot be read: {exc}") from exc
    return apply_label_payload(job_dir, payload)


def select_active_review(job_dir: Path, limit: int) -> dict[str, object]:
    if limit < 1 or limit > 50:
        raise ValueError("limit must be from 1 to 50.")
    frame = load_label_sheet(job_dir)
    labels, phrases = label_summary(frame)
    labeled = frame[labels.notna() & phrases.ne("")].copy()
    labeled_labels = labels.loc[labeled.index]
    if len(labeled) < 20 or labeled_labels.nunique() < 2:
        return sample_rows(job_dir, 0, limit, only_unlabeled=True) | {
            "selection": "initial_priority",
            "reason": "Need at least 20 labels across two real classes for active review.",
        }
    unlabeled = frame[labels.isna() & phrases.ne("")].copy()
    if unlabeled.empty:
        return {"stage": "ready_to_run", "selection": "complete", "rows": []}
    # This classifier is used only after the initial labels exist.  Import it
    # here so `next`, `sample`, and label import do not pay scikit-learn's
    # startup cost on every agent turn.
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=12000)
    train = vectorizer.fit_transform(labeled["Phrase"].astype(str))
    candidate = vectorizer.transform(unlabeled["Phrase"].astype(str))
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(train, labeled_labels)
    probabilities = classifier.predict_proba(candidate)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if probabilities.shape[1] > 1 else np.ones(len(unlabeled))
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    priority = pd.to_numeric(unlabeled.get("Search Volume", 0), errors="coerce").fillna(0).to_numpy()
    ranking = np.lexsort((-priority, -entropy, margin))[:limit]
    selected = unlabeled.iloc[ranking].copy()
    rows = [
        {
            "id": str(row["Sample ID"]),
            "phrase": str(row["Phrase"]),
            "reason": "low_margin",
            "priority": round(float(pd.to_numeric(row.get("Search Volume", 0), errors="coerce") if pd.notna(pd.to_numeric(row.get("Search Volume", 0), errors="coerce")) else 0), 3),
        }
        for _, row in selected.iterrows()
    ]
    write_state(job_dir, "active_review", labels_reviewed=int(labels.notna().sum()), remaining_review_rows=int(labels.isna().sum()))
    return {"stage": "active_review", "selection": "uncertain_tfidf", "rows": rows, "remaining_unlabeled": int(len(unlabeled))}


def import_feedback(job_dir: Path, reviewed_workbook: Path) -> int:
    if not reviewed_workbook.is_file():
        raise FileNotFoundError(reviewed_workbook)
    review = pd.read_excel(reviewed_workbook, sheet_name="Human review")
    lookup = normalized_column_lookup(review)
    phrase_column = lookup.get("phrase")
    intent_column = lookup.get("correct intent")
    notes_column = lookup.get("reviewer notes")
    confidence_column = lookup.get("classification confidence")
    if not phrase_column or not intent_column:
        raise ValueError("Human review must contain Phrase and Correct Intent columns.")

    corrections = pd.DataFrame(
        {
            "Phrase": review[phrase_column].astype(str).str.strip(),
            "Model Label": review[intent_column].fillna("").astype(str).str.strip(),
            "Model Confidence": (
                review[confidence_column] if confidence_column else ""
            ),
            "Model Notes": review[notes_column].fillna("").astype(str) if notes_column else "",
        }
    )
    corrections["_normalized_label"] = (
        corrections["Model Label"].str.lower().map(LABEL_ALIASES)
    )
    corrections = corrections[
        corrections["_normalized_label"].notna() & corrections["Phrase"].ne("")
    ].copy()
    corrections["Model Label"] = corrections["_normalized_label"]
    corrections = corrections.drop(columns="_normalized_label")
    if corrections.empty:
        return 0

    labels_path = job_dir / "model_labels.xlsx"
    existing = load_label_sheet(job_dir)
    for column in ("Model Label", "Model Confidence", "Model Notes"):
        if column not in existing.columns:
            existing[column] = ""
    combined = pd.concat([existing, corrections], ignore_index=True)
    combined["Sample ID"] = [f"row-{index:06d}" for index in range(1, len(combined) + 1)]
    combined["_phrase_key"] = combined["Phrase"].astype(str).str.strip().str.lower()
    combined = combined.drop_duplicates("_phrase_key", keep="last").drop(columns="_phrase_key")
    combined.to_excel(labels_path, sheet_name="Model labels", index=False)
    return len(corrections)


def resolve_reviewed_workbook(value: Path) -> Path:
    """Accept an explicit review path or a filename in files/ or outputs/."""
    if value.is_file():
        return value.resolve()
    for directory in (PROJECT_DIR / "files", OUTPUT_DIR):
        candidate = directory / value.name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Reviewed workbook not found: {value}. Put it in {PROJECT_DIR / 'files'} or provide its full path."
    )


def workflow_job_id_from_workbook(reviewed_workbook: Path) -> str:
    try:
        configuration = pd.read_excel(reviewed_workbook, sheet_name="Run configuration")
    except ValueError as exc:
        raise ValueError("Reviewed workbook has no Run configuration sheet with Workflow Job ID.") from exc
    lookup = normalized_column_lookup(configuration)
    parameter_column = lookup.get("parameter")
    value_column = lookup.get("value")
    if not parameter_column or not value_column:
        raise ValueError("Run configuration must contain Parameter and Value columns.")
    # The generated configuration sheet uses the Python key spelling
    # ``workflow_job_id``. Treat underscores and spaces alike so this stays
    # compatible with both the machine-readable key and the human label.
    parameters = configuration[parameter_column].fillna("").astype(str).map(
        lambda value: re.sub(r"[_\s]+", " ", value).strip().lower()
    )
    rows = configuration[parameters.eq("workflow job id")]
    if len(rows) != 1:
        raise ValueError("Reviewed workbook must contain exactly one Workflow Job ID.")
    workflow_job_id = str(rows.iloc[0][value_column]).strip()
    if not re.fullmatch(r"seo-[a-f0-9]{16}", workflow_job_id):
        raise ValueError("Workflow Job ID is missing or invalid; no import was made.")
    return workflow_job_id


def reviewed_corrections(reviewed_workbook: Path) -> pd.DataFrame:
    if not reviewed_workbook.is_file():
        return pd.DataFrame(columns=["Phrase", "Label", "Confidence"])
    review = pd.read_excel(reviewed_workbook, sheet_name="Human review")
    lookup = normalized_column_lookup(review)
    phrase_column = lookup.get("phrase")
    intent_column = lookup.get("correct intent")
    if not phrase_column or not intent_column:
        return pd.DataFrame(columns=["Phrase", "Label", "Confidence"])
    result = pd.DataFrame(
        {
            "Phrase": review[phrase_column].astype(str).str.strip(),
            "Label": review[intent_column].fillna("").astype(str).str.strip().str.lower().map(LABEL_ALIASES),
            "Confidence": 1.0,
        }
    )
    return result[result["Label"].notna() & result["Phrase"].ne("")]


def learn_job(job_dir: Path, reviewed_workbook: Path | None) -> dict[str, object]:
    config_path = job_dir / "job_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    description = str(config["topic"]["description"])
    reused = config.get("knowledge_reuse", {}).get("topic_key")
    key = reused or topic_key(description)
    save_topic_profile(key, description, config)

    labels_path = job_dir / "model_labels.xlsx"
    labels = pd.read_excel(labels_path, sheet_name="Model labels")
    lookup = normalized_column_lookup(labels)
    phrase_column = lookup.get("phrase")
    label_column = lookup.get("model label")
    confidence_column = lookup.get("model confidence")
    if not phrase_column or not label_column:
        raise ValueError("model_labels.xlsx must contain Phrase and Model Label.")
    examples = pd.DataFrame(
        {
            "Phrase": labels[phrase_column].astype(str).str.strip(),
            "Label": labels[label_column].fillna("").astype(str).str.strip(),
            "Confidence": (
                labels[confidence_column] if confidence_column else 0.8
            ),
        }
    )
    learned = store_examples(
        key, examples, job_dir.name, "model_reviewed", default_confidence=0.8
    )
    corrected = 0
    if reviewed_workbook:
        corrections = reviewed_corrections(reviewed_workbook)
        corrected = store_examples(
            key,
            corrections,
            job_dir.name,
            "review_corrected",
            default_confidence=1.0,
        )
    return {
        "topic_key": key,
        "model_reviewed_examples_processed": learned,
        "review_corrected_examples_processed": corrected,
        "knowledge": knowledge_status(key),
    }


def apply_review_and_learn(reviewed_value: Path) -> dict[str, object]:
    """Import a human review and persist knowledge without recalculating SEO output."""
    reviewed_workbook = resolve_reviewed_workbook(reviewed_value)
    workflow_job_id = workflow_job_id_from_workbook(reviewed_workbook)
    job_dir = find_job_by_workflow_id(workflow_job_id)
    imported = import_feedback(job_dir, reviewed_workbook)
    learned = learn_job(job_dir, reviewed_workbook)
    state = write_state(
        job_dir,
        "knowledge_saved",
        feedback_imported=imported,
        knowledge_topic_key=learned["topic_key"],
        reviewed_workbook=str(reviewed_workbook),
    )
    return {
        "status": "knowledge_saved",
        "recalculated": False,
        "workflow_job_id": workflow_job_id,
        "feedback_imported": imported,
        "reviewed_workbook": str(reviewed_workbook),
        "model_reviewed_examples_processed": learned["model_reviewed_examples_processed"],
        "review_corrected_examples_processed": learned["review_corrected_examples_processed"],
        # Keep the terminal response ASCII and compact. The complete topic
        # metadata remains in the local knowledge database and state.json.
        "stage": state["stage"],
    }


def cleanup_job(job_dir: Path) -> list[str]:
    removed: list[str] = []
    safe_targets = [job_dir / "tmp", job_dir / "cache"]
    for target in safe_targets:
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
    cache_targets = [PROJECT_DIR / "__pycache__", *job_dir.rglob("__pycache__")]
    for cache in cache_targets:
        if cache.is_dir():
            shutil.rmtree(cache)
            removed.append(str(cache))
    for temporary in job_dir.rglob("*"):
        if temporary.is_file() and (
            temporary.suffix.lower() in {".tmp", ".part"}
            or temporary.name.startswith("~$")
        ):
            temporary.unlink()
            removed.append(str(temporary))
    lock_path = job_dir / RUN_LOCK_NAME
    if lock_path.is_file():
        lock_data = read_run_lock(lock_path)
        if not lock_process_is_active(lock_data):
            lock_path.unlink()
            removed.append(str(lock_path))
    return removed


def read_run_lock(lock_path: Path) -> dict[str, object]:
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def lock_process_is_active(lock_data: dict[str, object]) -> bool:
    """Return True only when a lock owner still appears to be running."""
    try:
        process_id = int(lock_data.get("pid", 0))
    except (TypeError, ValueError):
        return False
    if process_id <= 0:
        return False
    if os.name == "nt":
        # Python 3.14 on Windows may turn ``os.kill(pid, 0)`` with access
        # denied into a SystemError.  Query the process handle directly so a
        # lock check never breaks `next` while a worker is running.
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(process_query_limited_information, False, process_id)
        if not handle:
            error_code = ctypes.get_last_error()
            # Access denied means that Windows knows the process exists but
            # this interpreter may not inspect it (for example, an elevated
            # worker launched by Hermes).
            return error_code == 5
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_run_lock(job_dir: Path) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Atomically claim the one permitted final run for a job."""
    lock_path = job_dir / RUN_LOCK_NAME
    token = uuid.uuid4().hex
    mine = {
        "token": token,
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for _ in range(2):
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = read_run_lock(lock_path)
            if lock_process_is_active(existing):
                return None, existing
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            continue
        try:
            os.write(descriptor, json.dumps(mine, ensure_ascii=False).encode("utf-8"))
        finally:
            os.close(descriptor)
        return mine, None
    return None, read_run_lock(lock_path)


def release_run_lock(job_dir: Path, token: str) -> None:
    """Release only the lock created by this exact workflow process."""
    lock_path = job_dir / RUN_LOCK_NAME
    if read_run_lock(lock_path).get("token") == token:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def next_output_path(job_dir: Path, input_value: str) -> Path:
    input_path = resolve_input(input_value)
    job_output_dir = OUTPUT_DIR / job_dir.name
    job_output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(job_output_dir.glob(f"{input_path.stem}_run_*.xlsx"))
    numbers = []
    for path in existing:
        match = re.search(r"_run_(\d+)$", path.stem)
        if match:
            numbers.append(int(match.group(1)))
    return job_output_dir / f"{input_path.stem}_run_{max(numbers, default=0) + 1:03d}.xlsx"


def latest_run_path(job_dir: Path, input_value: str) -> Path:
    input_path = resolve_input(input_value)
    job_output_dir = OUTPUT_DIR / job_dir.name
    candidates: list[tuple[int, Path]] = []
    for path in job_output_dir.glob(f"{input_path.stem}_run_*.xlsx"):
        match = re.search(r"_run_(\d+)$", path.stem)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"No versioned run found for {input_path.name} in {job_output_dir}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def finalize_workbook(
    job_dir: Path,
    input_value: str,
    source: Path | None = None,
) -> dict[str, object]:
    input_path = resolve_input(input_value)
    job_output_dir = OUTPUT_DIR / job_dir.name
    job_output_dir.mkdir(parents=True, exist_ok=True)

    source_path = source.resolve() if source else latest_run_path(job_dir, input_value)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".xlsx":
        raise ValueError("The final source must be an .xlsx workbook.")

    with pd.ExcelFile(source_path) as workbook:
        available_sheets = set(workbook.sheet_names)
    missing_sheets = sorted(REQUIRED_RESULT_SHEETS - available_sheets)
    if missing_sheets:
        raise ValueError(
            "Cannot finalize an incomplete result workbook. Missing sheets: "
            + ", ".join(missing_sheets)
        )

    final_path = job_output_dir / f"{input_path.stem}_FINAL.xlsx"
    temporary_path = job_output_dir / f".{input_path.stem}_FINAL.tmp"
    replaced_previous_final = final_path.exists()
    shutil.copy2(source_path, temporary_path)
    temporary_path.replace(final_path)
    return {
        "input": str(input_path.resolve()),
        "source": str(source_path.resolve()),
        "final": str(final_path.resolve()),
        "source_preserved": True,
        "replaced_previous_final": replaced_previous_final,
    }


def record_run_failure(job_dir: Path, error: Exception) -> None:
    """Make deterministic pipeline failures terminal for the Hermes autopilot."""
    write_state(
        job_dir,
        "blocked",
        autopilot_last_status="blocked",
        run_error=str(error),
    )


def run_with_log(job_dir: Path, input_value: str, output_path: Path | None) -> dict[str, object]:
    lock, existing_lock = acquire_run_lock(job_dir)
    if lock is None:
        return {
            "status": "already_running",
            "job": str(job_dir.resolve()),
            "pid": existing_lock.get("pid") if existing_lock else None,
            "started_at": existing_lock.get("started_at") if existing_lock else None,
        }
    logs = job_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "run.log"
    started = time.perf_counter()
    write_state(
        job_dir,
        "run",
        autopilot_last_status="running",
        run_pid=lock["pid"],
        run_started_at=lock["started_at"],
        run_mode="standard",
    )
    try:
        try:
            ensure_workflow_job_id(job_dir)
            # Importing the pipeline initializes torch/transformers/UMAP.  Keep it
            # out of all short control commands used during agent labelling.
            from seo_pipeline import run_pipeline

            quiet_library_logs()
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] run started")
                output = run_pipeline(
                    input_value,
                    job_dir / "job_config.json",
                    job_dir / "model_labels.xlsx",
                    output_path,
                )
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] run completed")
        except Exception:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            raise
    except Exception as error:
        record_run_failure(job_dir, error)
        release_run_lock(job_dir, str(lock["token"]))
        raise
    duration = round(time.perf_counter() - started, 3)
    inspection = json.loads((job_dir / "inspection.json").read_text(encoding="utf-8"))
    state = write_state(
        job_dir,
        "quality_review",
        last_run=output.name,
        last_run_path=str(output.resolve()),
        duration_seconds=duration,
        labels_reviewed=job_status(job_dir).get("label_quality", {}).get("valid_labeled_rows", 0),
    )
    result = {
        "status": "completed",
        "output": str(output.resolve()),
        "rows": int(inspection.get("unique_normalized_phrases", 0)),
        "duration_seconds": duration,
        "log": str(log_path.resolve()),
        "stage": state["stage"],
    }
    release_run_lock(job_dir, str(lock["token"]))
    return result


def stage_large_input(job_dir: Path, input_value: str) -> tuple[str, Path | None]:
    """Create a workflow-owned TSV only when a large XLSX needs streaming.

    The language model never converts user files.  This short-lived staging
    file contains only the configured phrase and optional frequency columns
    and is removed by ``run_large_with_log`` in every exit path.
    """
    input_path = resolve_input(input_value)
    if input_path.suffix.lower() in {".csv", ".tsv"}:
        return str(input_path), None
    if input_path.suffix.lower() != ".xlsx":
        raise ValueError("Large mode supports CSV, TSV, and XLSX inputs.")
    config = json.loads((job_dir / "job_config.json").read_text(encoding="utf-8"))
    requested_phrase = str(config.get("phrase_column") or "").strip()
    requested_frequency = str(config.get("frequency_column") or "").strip()
    sheet_ref = config.get("source_sheet", 0)
    workbook = load_workbook(input_path, read_only=True, data_only=True)
    try:
        sheet = workbook[sheet_ref] if isinstance(sheet_ref, str) else workbook.worksheets[int(sheet_ref)]
        rows = sheet.iter_rows(values_only=True)
        header_values = next(rows, None)
        if not header_values:
            raise ValueError("The XLSX source sheet has no header row.")
        headers = ["" if value is None else str(value).strip() for value in header_values]
        normalized = {re.sub(r"\s+", " ", value.lower()): index for index, value in enumerate(headers)}
        phrase_index = normalized.get(re.sub(r"\s+", " ", requested_phrase.lower()))
        if phrase_index is None:
            raise ValueError(f"Phrase column not found in XLSX staging: {requested_phrase}")
        frequency_index = normalized.get(re.sub(r"\s+", " ", requested_frequency.lower())) if requested_frequency else None
        staging = job_dir / "tmp" / f"large-input-{uuid.uuid4().hex}.tsv"
        staging.parent.mkdir(parents=True, exist_ok=True)
        with staging.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            output_header = [headers[phrase_index]]
            if frequency_index is not None:
                output_header.append(headers[frequency_index])
            writer.writerow(output_header)
            for values in rows:
                phrase = values[phrase_index] if phrase_index < len(values) else None
                if phrase is None or not str(phrase).strip():
                    continue
                output_row = [phrase]
                if frequency_index is not None:
                    output_row.append(values[frequency_index] if frequency_index < len(values) else "")
                writer.writerow(output_row)
        return str(staging), staging
    finally:
        workbook.close()


def run_large_with_log(job_dir: Path, input_value: str, chunk_size: int) -> dict[str, object]:
    lock, existing_lock = acquire_run_lock(job_dir)
    if lock is None:
        return {
            "status": "already_running",
            "job": str(job_dir.resolve()),
            "pid": existing_lock.get("pid") if existing_lock else None,
            "started_at": existing_lock.get("started_at") if existing_lock else None,
        }
    logs = job_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "run_large.log"
    output_dir = OUTPUT_DIR / job_dir.name / "large" / Path(input_value).stem
    started = time.perf_counter()
    staged_path: Path | None = None
    write_state(
        job_dir,
        "run",
        autopilot_last_status="running",
        run_pid=lock["pid"],
        run_started_at=lock["started_at"],
        run_mode="large",
    )
    try:
        try:
            ensure_workflow_job_id(job_dir)
            from seo_pipeline import run_large_pipeline

            staged_input, staged_path = stage_large_input(job_dir, input_value)
            quiet_library_logs()
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] large run started")
                manifest = run_large_pipeline(staged_input, job_dir / "job_config.json", job_dir / "model_labels.xlsx", output_dir, chunk_size, source_name=Path(input_value).name)
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] large run completed")
        except Exception:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            raise
    except Exception as error:
        record_run_failure(job_dir, error)
        release_run_lock(job_dir, str(lock["token"]))
        raise
    finally:
        if staged_path and staged_path.is_file():
            staged_path.unlink()
    duration = round(time.perf_counter() - started, 3)
    state = write_state(job_dir, "quality_review", last_large_run=manifest["excel_summary"], duration_seconds=duration)
    result = {"status": "completed", "output": manifest["excel_summary"], "rows": manifest["classified_rows_per_chunk_deduplicated"], "duration_seconds": duration, "log": str(log_path.resolve()), "stage": state["stage"]}
    release_run_lock(job_dir, str(lock["token"]))
    return result


def run_auto(job_dir: Path, input_value: str, threshold: int, chunk_size: int) -> dict[str, object]:
    if threshold < 1:
        raise ValueError("large-threshold must be at least 1.")
    inspection_path = job_dir / "inspection.json"
    if not inspection_path.is_file():
        raise FileNotFoundError("inspection.json is required for run-auto.")
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    rows = int(inspection.get("unique_normalized_phrases", 0))
    input_path = resolve_input(input_value)
    if rows < threshold:
        result = run_with_log(job_dir, input_value, None)
        result["mode"] = "standard"
        result["large_threshold"] = threshold
        return result
    if input_path.suffix.lower() not in {".csv", ".tsv", ".xlsx"}:
        raise ValueError(
            f"run-auto selected large mode for {rows} phrases, but the input format "
            f"{input_path.suffix or '<none>'} is unsupported. Use CSV, TSV, or XLSX."
        )
    result = run_large_with_log(job_dir, input_value, chunk_size)
    result["mode"] = "large"
    result["large_threshold"] = threshold
    return result


def export_completed_large_result(job_dir: Path, input_value: str) -> dict[str, object]:
    """Build the human review workbook from an already completed large run.

    This deliberately reads only the classified part files.  It never starts
    embeddings, classification, or clustering again.
    """
    input_path = resolve_input(input_value)
    output_dir = OUTPUT_DIR / job_dir.name / "large" / input_path.stem
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed large-run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Large run is not completed; export was not started.")
    parts_dir = output_dir / "parts"
    if not list(parts_dir.glob("part-*.csv.gz")):
        raise FileNotFoundError("Completed large-run parts are missing.")
    from seo_pipeline import export_large_result_workbook

    summary_path = output_dir / "cluster_summary.csv"
    uncertain_path = output_dir / "uncertain_review.csv"
    summary = pd.read_csv(summary_path, encoding="utf-8-sig") if summary_path.is_file() else pd.DataFrame()
    uncertain = pd.read_csv(uncertain_path, encoding="utf-8-sig") if uncertain_path.is_file() else pd.DataFrame()
    config = json.loads((job_dir / "job_config.json").read_text(encoding="utf-8"))
    result_path = output_dir / f"{input_path.stem}_clustered.xlsx"
    counts = export_large_result_workbook(parts_dir, result_path, summary, uncertain, config, manifest)
    manifest["excel_result"] = str(result_path.resolve())
    manifest["excel_summary"] = str(result_path.resolve())
    manifest["intent_counts"] = counts
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_state(job_dir, "quality_review", last_large_run=str(result_path.resolve()))
    return {
        "status": "completed",
        "recalculated": False,
        "output": str(result_path.resolve()),
        "rows": int(manifest.get("classified_rows_per_chunk_deduplicated", 0)),
        "intent_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate the model-directed SEO workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Inspect input and create a representative labeling job.")
    prepare.add_argument("input")
    prepare.add_argument("--topic", required=True)
    prepare.add_argument("--sample-size", type=int, default=500)
    prepare.add_argument("--phrase-column")
    prepare.add_argument("--frequency-column")
    prepare.add_argument("--job-name")
    prepare.add_argument(
        "--reuse-topic",
        default="auto",
        help="Prior knowledge topic, auto for current topic, or none",
    )
    prepare.add_argument("--prior-limit", type=int, default=150)
    prepare.add_argument("--quiet", action="store_true")

    status = subparsers.add_parser("status", help="Show labeling and configuration readiness.")
    status.add_argument("--job", required=True)
    status.add_argument("--compact", action="store_true")
    status.add_argument("--quiet", action="store_true")

    next_step = subparsers.add_parser("next", help="Return the next safe action from the persistent job state.")
    next_step.add_argument("--job")
    next_step.add_argument("--input", help="Input filename used to safely locate an existing job or create the first prepare action.")
    next_step.add_argument("--topic", help="Required with a new input when no prepared job exists.")
    next_step.add_argument("--quiet", action="store_true")

    sample = subparsers.add_parser("sample", help="Return a small UTF-8 JSON labeling batch; never print XLSX rows.")
    sample.add_argument("--job", required=True)
    sample.add_argument("--offset", type=int, default=0)
    sample.add_argument("--limit", type=int, default=25)
    sample.add_argument("--include-labeled", action="store_true")
    sample.add_argument("--quiet", action="store_true")

    labels = subparsers.add_parser("apply-labels", help="Atomically import model labels from a job-local JSON batch.")
    labels.add_argument("--job", required=True)
    labels.add_argument("--input", type=Path, required=True)
    labels.add_argument("--quiet", action="store_true")

    inline_labels = subparsers.add_parser("apply-labels-inline", help="Import a compact phrase-free batch: SampleID|label|confidence;...")
    inline_labels.add_argument("--job", required=True)
    inline_labels.add_argument("--labels", required=True)
    inline_labels.add_argument("--quiet", action="store_true")

    review = subparsers.add_parser("review", help="Select an uncertainty-focused active-learning labeling batch.")
    review.add_argument("--job", required=True)
    review.add_argument("--limit", type=int, default=20)
    review.add_argument("--quiet", action="store_true")

    run = subparsers.add_parser("run", help="Run classification and clustering for a prepared job.")
    run.add_argument("input")
    run.add_argument("--job", required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--quiet", action="store_true")

    run_large = subparsers.add_parser("run-large", help="Chunked local GPU/CPU mode for CSV/TSV semantic cores; full data is partitioned CSV, not Excel.")
    run_large.add_argument("input")
    run_large.add_argument("--job", required=True)
    run_large.add_argument("--chunk-size", type=int, default=50000)
    run_large.add_argument("--quiet", action="store_true")

    run_auto_parser = subparsers.add_parser("run-auto", help="Choose standard or large local GPU/CPU mode from the prepared unique-phrase count.")
    run_auto_parser.add_argument("input")
    run_auto_parser.add_argument("--job", required=True)
    run_auto_parser.add_argument("--large-threshold", type=int, default=DEFAULT_LARGE_THRESHOLD)
    run_auto_parser.add_argument("--chunk-size", type=int, default=50000)
    run_auto_parser.add_argument("--quiet", action="store_true")

    export_large = subparsers.add_parser(
        "export-large", help="Create the full review workbook from completed large-run parts without recalculation."
    )
    export_large.add_argument("input")
    export_large.add_argument("--job", required=True)
    export_large.add_argument("--quiet", action="store_true")

    finalize = subparsers.add_parser(
        "finalize",
        help="Create a stable *_FINAL.xlsx copy from the latest or specified reviewed run.",
    )
    finalize.add_argument("input")
    finalize.add_argument("--job", required=True)
    finalize.add_argument(
        "--source",
        type=Path,
        help="Reviewed result workbook; defaults to the highest numbered run.",
    )
    finalize.add_argument("--quiet", action="store_true")

    feedback = subparsers.add_parser(
        "feedback", help="Import Correct Intent values from a reviewed output workbook."
    )
    feedback.add_argument("reviewed_workbook", type=Path)
    feedback.add_argument("--job", required=True)
    feedback.add_argument("--quiet", action="store_true")

    apply_review = subparsers.add_parser(
        "apply-review",
        help="Import Human review corrections and save topic knowledge; never reruns cleaning.",
    )
    apply_review.add_argument("reviewed_workbook", type=Path)
    apply_review.add_argument("--quiet", action="store_true")

    provenance = subparsers.add_parser(
        "provenance",
        help="Assign or show the permanent ASCII Workflow Job ID for one job.",
    )
    provenance.add_argument("--job", required=True)
    provenance.add_argument("--quiet", action="store_true")

    learn = subparsers.add_parser(
        "learn", help="Persist model-reviewed labels and optional corrected feedback."
    )
    learn.add_argument("--job", required=True)
    learn.add_argument("--reviewed-workbook", type=Path)
    learn.add_argument("--quiet", action="store_true")

    knowledge = subparsers.add_parser(
        "knowledge", help="Show persistent topic knowledge."
    )
    knowledge.add_argument("--topic")
    knowledge.add_argument("--quiet", action="store_true")

    cleanup = subparsers.add_parser(
        "cleanup", help="Remove only known temporary files and Python caches."
    )
    cleanup.add_argument("--job", required=True)
    cleanup.add_argument("--quiet", action="store_true")

    forget = subparsers.add_parser(
        "forget", help="Delete one topic's persistent knowledge after explicit confirmation."
    )
    forget.add_argument("--topic", required=True)
    forget.add_argument(
        "--confirm",
        required=True,
        help="Must exactly match the normalized topic key shown by knowledge status",
    )
    forget.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        # Preparation uses scikit-learn only once; do not import it while the
        # agent merely asks for `next`, samples a batch, or imports labels.
        from seo_prepare import prepare_job

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
        inspection = json.loads((job_dir / "inspection.json").read_text(encoding="utf-8"))
        workflow_job_id = ensure_workflow_job_id(job_dir)
        write_state(job_dir, "initial_labeling", input=inspection["input_file"], topic=inspection["topic"], workflow_job_id=workflow_job_id, labels_reviewed=0, remaining_review_rows=inspection["sample_rows"])
        # The empty seed lists are expected before the first model labels.  Do
        # not expose readiness checks here: Hermes must receive its first
        # compact labeling command, not a misleading configuration block.
        print_result(recorded_next_action(job_dir), args.quiet)
    elif args.command == "status":
        result = job_status(resolve_job(args.job))
        print_result(compact_status(result) if args.compact else result, args.quiet)
    elif args.command == "next":
        if args.job:
            job_dir = resolve_job(args.job)
            print_result(recorded_next_action(job_dir), args.quiet)
        elif args.input:
            # Validate the requested source before looking up an existing job.
            # This keeps slash-command entry points from silently accepting a
            # misspelled or missing filename.
            input_path = resolve_input(args.input)
            job_dir = find_job_for_input(input_path.name)
            if job_dir:
                print_result(recorded_next_action(job_dir), args.quiet)
            elif args.topic:
                print_result(
                    {
                        "status": "continue",
                        "stage": "prepare",
                        "action": "prepare_job",
                        "command": f'{HERMES_COMMAND} prepare "{input_path.name}" --topic "{args.topic}" --quiet',
                    },
                    args.quiet,
                )
            else:
                raise ValueError("next needs --job, or both --input and --topic for a new job.")
        else:
            raise ValueError("next needs --job, or both --input and --topic for a new job.")
    elif args.command == "sample":
        print_result(sample_rows(resolve_job(args.job), args.offset, args.limit, not args.include_labeled), args.quiet)
    elif args.command == "apply-labels":
        print_result(apply_labels(resolve_job(args.job), args.input), args.quiet)
    elif args.command == "apply-labels-inline":
        print_result(
            apply_label_payload(resolve_job(args.job), parse_inline_labels(args.labels)),
            args.quiet,
        )
    elif args.command == "review":
        print_result(select_active_review(resolve_job(args.job), args.limit), args.quiet)
    elif args.command == "run":
        job_dir = resolve_job(args.job)
        readiness = job_status(job_dir)
        if not readiness["ready_for_supervised_run"]:
            details = "\n- ".join(readiness["blocking_errors"])
            raise ValueError(
                "Job is not ready for a supervised run. Fix these blocking errors:\n- "
                + details
            )
        output_path = args.output or next_output_path(job_dir, args.input)
        print_result(run_with_log(job_dir, args.input, output_path), args.quiet)
    elif args.command == "run-large":
        job_dir = resolve_job(args.job)
        readiness = job_status(job_dir)
        if not readiness["ready_for_supervised_run"]:
            details = "\n- ".join(readiness["blocking_errors"])
            raise ValueError(
                "Job is not ready for a supervised large run. Fix these blocking errors:\n- " + details
            )
        print_result(run_large_with_log(job_dir, args.input, args.chunk_size), args.quiet)
    elif args.command == "run-auto":
        job_dir = resolve_job(args.job)
        readiness = job_status(job_dir)
        if not readiness["ready_for_supervised_run"]:
            details = "\n- ".join(readiness["blocking_errors"])
            raise ValueError(
                "Job is not ready for a supervised run. Fix these blocking errors:\n- " + details
            )
        print_result(run_auto(job_dir, args.input, args.large_threshold, args.chunk_size), args.quiet)
    elif args.command == "export-large":
        print_result(export_completed_large_result(resolve_job(args.job), args.input), args.quiet)
    elif args.command == "finalize":
        job_dir = resolve_job(args.job)
        result = finalize_workbook(job_dir, args.input, args.source)
        write_state(job_dir, "finalized", final=result["final"])
        print_result(result, args.quiet)
    elif args.command == "feedback":
        job_dir = resolve_job(args.job)
        reviewed_workbook = resolve_reviewed_workbook(args.reviewed_workbook)
        imported = import_feedback(job_dir, reviewed_workbook)
        write_state(job_dir, "quality_review", feedback_imported=imported)
        print_result({"status": "feedback_imported", "imported": imported, "job": str(job_dir)}, args.quiet)
    elif args.command == "apply-review":
        print_result(
            apply_review_and_learn(args.reviewed_workbook),
            args.quiet,
        )
    elif args.command == "provenance":
        job_dir = resolve_job(args.job)
        print_result(
            {
                "status": "provenance_ready",
                "workflow_job_id": ensure_workflow_job_id(job_dir),
            },
            args.quiet,
        )
    elif args.command == "learn":
        job_dir = resolve_job(args.job)
        result = learn_job(job_dir, args.reviewed_workbook)
        write_state(job_dir, "learned", knowledge_topic_key=result["topic_key"])
        print_result(result, args.quiet)
    elif args.command == "knowledge":
        key = topic_key(args.topic) if args.topic else None
        print_result(knowledge_status(key), args.quiet)
    elif args.command == "cleanup":
        job_dir = resolve_job(args.job)
        removed = cleanup_job(job_dir)
        print_result({"removed": removed, "count": len(removed)}, args.quiet)
    elif args.command == "forget":
        key = topic_key(args.topic)
        if args.confirm != key:
            raise ValueError(f"Confirmation mismatch. Required: --confirm {key}")
        print_result({"topic_key": key, **forget_topic(key)}, args.quiet)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
