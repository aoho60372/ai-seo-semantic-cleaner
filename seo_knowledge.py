"""Persistent, reviewed topic knowledge for SEO jobs."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
DATABASE_PATH = KNOWLEDGE_DIR / "seo_knowledge.sqlite3"
VALID_LABELS = {"commercial", "informational", "garbage"}
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


def topic_key(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "-", normalize(value)).strip("-")[:100] or "topic"


def connect() -> sqlite3.Connection:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS topics (
            topic_key TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS examples (
            topic_key TEXT NOT NULL,
            phrase_key TEXT NOT NULL,
            phrase TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            verification TEXT NOT NULL,
            source_job TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (topic_key, phrase_key)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS examples_topic_label ON examples(topic_key, label)"
    )
    return connection


def save_topic_profile(key: str, description: str, config: dict[str, Any]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO topics(topic_key, description, profile_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(topic_key) DO UPDATE SET
                description=excluded.description,
                profile_json=excluded.profile_json,
                updated_at=excluded.updated_at
            """,
            (key, description, json.dumps(config, ensure_ascii=False), timestamp),
        )


def load_topic_profile(key: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT profile_json FROM topics WHERE topic_key=?", (key,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def store_examples(
    key: str,
    examples: pd.DataFrame,
    source_job: str,
    verification: str,
    default_confidence: float,
) -> int:
    if examples.empty:
        return 0
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for record in examples.to_dict("records"):
        phrase = str(record.get("Phrase", "")).strip()
        label = LABEL_ALIASES.get(str(record.get("Label", "")).strip().lower())
        if not phrase or label not in VALID_LABELS:
            continue
        confidence = pd.to_numeric(record.get("Confidence"), errors="coerce")
        confidence = (
            float(confidence)
            if pd.notna(confidence)
            else float(default_confidence)
        )
        rows.append(
            (
                key,
                normalize(phrase),
                phrase,
                label,
                max(0.0, min(1.0, confidence)),
                verification,
                source_job,
                timestamp,
            )
        )
    with connect() as connection:
        connection.executemany(
            """
            INSERT INTO examples(
                topic_key, phrase_key, phrase, label, confidence,
                verification, source_job, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_key, phrase_key) DO UPDATE SET
                phrase=excluded.phrase,
                label=excluded.label,
                confidence=MAX(examples.confidence, excluded.confidence),
                verification=CASE
                    WHEN excluded.verification='review_corrected'
                    THEN excluded.verification
                    ELSE examples.verification
                END,
                source_job=excluded.source_job,
                updated_at=excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def load_balanced_examples(key: str, limit: int = 300) -> pd.DataFrame:
    per_class = max(1, limit // 3)
    frames = []
    with connect() as connection:
        for label in sorted(VALID_LABELS):
            frame = pd.read_sql_query(
                """
                SELECT phrase AS Phrase, label AS "Model Label",
                       confidence AS "Model Confidence",
                       verification AS "Knowledge Source",
                       source_job AS "Source Job"
                FROM examples
                WHERE topic_key=? AND label=?
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                connection,
                params=(key, label, per_class),
            )
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) > limit:
        result = result.head(limit)
    result["Model Notes"] = "Reused prior reviewed topic knowledge; recheck against current scope"
    return result


def knowledge_status(key: str | None = None) -> dict[str, Any]:
    with connect() as connection:
        if key:
            topic_row = connection.execute(
                "SELECT description, updated_at FROM topics WHERE topic_key=?", (key,)
            ).fetchone()
            counts = connection.execute(
                """
                SELECT label, verification, COUNT(*)
                FROM examples WHERE topic_key=?
                GROUP BY label, verification
                """,
                (key,),
            ).fetchall()
            return {
                "database": str(DATABASE_PATH),
                "topic_key": key,
                "topic": topic_row[0] if topic_row else None,
                "updated_at": topic_row[1] if topic_row else None,
                "counts": [
                    {"label": row[0], "verification": row[1], "count": row[2]}
                    for row in counts
                ],
            }
        topics = connection.execute(
            """
            SELECT t.topic_key, t.description, t.updated_at, COUNT(e.phrase_key)
            FROM topics t LEFT JOIN examples e ON e.topic_key=t.topic_key
            GROUP BY t.topic_key, t.description, t.updated_at
            ORDER BY t.updated_at DESC
            """
        ).fetchall()
    return {
        "database": str(DATABASE_PATH),
        "topics": [
            {
                "topic_key": row[0],
                "description": row[1],
                "updated_at": row[2],
                "examples": row[3],
            }
            for row in topics
        ],
    }


def forget_topic(key: str) -> dict[str, int]:
    with connect() as connection:
        examples = connection.execute(
            "DELETE FROM examples WHERE topic_key=?", (key,)
        ).rowcount
        topics = connection.execute(
            "DELETE FROM topics WHERE topic_key=?", (key,)
        ).rowcount
    return {"topics_removed": topics, "examples_removed": examples}
