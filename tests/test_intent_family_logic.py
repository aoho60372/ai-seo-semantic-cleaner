import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from seo_pipeline import (
    CORE_RESULT_COLUMNS,
    GARBAGE_RESULT_COLUMNS,
    HUMAN_REVIEW_RESULT_COLUMNS,
    build_intent_family_audit,
    compact_result_frame,
    marker_safety_reason,
    predict_two_stage,
)
from seo_prepare import (
    diverse_family_examples,
    intent_family_coverage_rows,
    intent_family_payload,
    phrase_intent_families,
    prepare_job,
)
from seo_workflow import (
    apply_intent_family_labels_inline,
    current_sample_intent_counts,
    family_coverage_status,
    family_overlaps_signals,
    final_policy_candidate_is_safe,
    sanitize_strong_signals,
    select_family_coverage_review,
    next_action,
)


class DummyStructural:
    def transform(self, texts):
        return csr_matrix((len(texts), 1), dtype=float)


class DummyIntentClassifier:
    classes_ = np.asarray(["commercial", "informational"])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return self.probabilities[: features.shape[0]]


class DummyRelevanceClassifier:
    classes_ = np.asarray(["garbage", "relevant"])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return self.probabilities[: features.shape[0]]


class IntentFamilyLogicTests(unittest.TestCase):
    def predict(
        self,
        phrases,
        probabilities,
        family_rules,
        commercial_markers=None,
        informational_markers=None,
        weak_question_markers=None,
    ):
        bundle = {
            "structural": DummyStructural(),
            "relevance_classifier": None,
            "intent_classifier": DummyIntentClassifier(probabilities),
            "settings": {
                "tfidf_weight": 1.0,
                "intent_tfidf_weight": 1.0,
                "garbage_threshold": 0.45,
                "quarantine_margin": 0.10,
            },
        }
        config = {
            "intent": {
                "default": "commercial",
                "informational_decision_margin": 0.12,
                "strong_informational_decision_margin": 0.02,
                "family_override_tolerance": 0.05,
                "commercial_markers": commercial_markers or [],
                "informational_markers": informational_markers or [],
                "weak_question_markers": weak_question_markers or [],
                "family_rules": family_rules,
            },
            "intent_policy": {},
        }
        return predict_two_stage(
            bundle,
            pd.Series(phrases),
            np.zeros((len(phrases), 2), dtype=float),
            config,
        )

    def test_family_only_changes_near_boundary(self):
        result = self.predict(
            ["график вакансии", "свободный график вакансии", "какой график работы"],
            [[0.80, 0.20], [0.525, 0.475], [0.20, 0.80]],
            {"commercial": [], "informational": ["графи*"], "neutral": []},
        )
        self.assertEqual(result["prediction"].tolist(), ["commercial", "informational", "informational"])
        self.assertEqual(result["family_override"].tolist(), [False, True, False])
        self.assertEqual(result["family_conflict"].tolist(), [True, False, False])
        self.assertAlmostEqual(float(result["confidence"][1]), 0.475, places=6)

    def test_opposite_marker_and_family_never_override_each_other(self):
        result = self.predict(
            ["купить график работы"],
            [[0.48, 0.52]],
            {"commercial": [], "informational": ["графи*"], "neutral": []},
            commercial_markers=["купит*"],
        )
        self.assertEqual(result["prediction"].tolist(), ["commercial"])
        self.assertEqual(result["family_override"].tolist(), [False])
        self.assertEqual(result["family_conflict"].tolist(), [True])

    def test_opposite_families_create_conflict_without_override(self):
        result = self.predict(
            ["вакансия свободный график"],
            [[0.65, 0.35]],
            {"commercial": ["вакан*"], "informational": ["графи*"], "neutral": []},
        )
        self.assertEqual(result["prediction"].tolist(), ["commercial"])
        self.assertEqual(result["family_override"].tolist(), [False])
        self.assertEqual(result["family_conflict"].tolist(), [True])

    def test_weak_question_family_is_detected(self):
        self.assertTrue(family_overlaps_signals("можно*", ["можно*", "где", "как"]))
        self.assertFalse(family_overlaps_signals("отзыв*", ["можно*", "где", "как"]))

    def test_existing_weak_question_family_is_neutral_at_runtime(self):
        result = self.predict(
            ["можно устроиться дизайнером"],
            [[0.45, 0.55]],
            {"commercial": [], "informational": ["можно*"], "neutral": []},
            weak_question_markers=["можно*"],
        )
        self.assertEqual(result["prediction"].tolist(), ["informational"])
        self.assertEqual(result["informational_family_hit"].tolist(), [False])

    def test_weak_question_uses_lower_margin_without_transaction(self):
        result = self.predict(
            ["где находится деталь"],
            [[0.48, 0.52]],
            {"commercial": [], "informational": [], "neutral": []},
            weak_question_markers=["где"],
        )
        self.assertEqual(result["prediction"].tolist(), ["informational"])
        self.assertEqual(result["weak_question_without_transaction"].tolist(), [True])

    def test_transaction_marker_protects_weak_question_commercial_intent(self):
        result = self.predict(
            ["где купить деталь"],
            [[0.40, 0.60]],
            {"commercial": [], "informational": [], "neutral": []},
            commercial_markers=["купит*"],
            weak_question_markers=["где"],
        )
        self.assertEqual(result["prediction"].tolist(), ["commercial"])
        self.assertEqual(result["weak_question_without_transaction"].tolist(), [False])

    def test_vacancy_structures_separate_duties_from_openings(self):
        result = self.predict(
            ["какую работу выполняет водитель", "какие вакансии водителя"],
            [[0.52, 0.48], [0.48, 0.52]],
            {
                "commercial": ["какие* вакан*"],
                "informational": ["какую* работ* выпол*"],
                "neutral": ["водит*"],
            },
        )
        self.assertEqual(result["prediction"].tolist(), ["informational", "commercial"])

    def test_autoparts_strong_context_still_beats_weak_question(self):
        result = self.predict(
            ["где купить бампер тойота", "как снять бампер тойота"],
            [[0.40, 0.60], [0.52, 0.48]],
            {"commercial": [], "informational": [], "neutral": ["бампе*", "тойот*"]},
            commercial_markers=["купит*"],
            informational_markers=["снять"],
            weak_question_markers=["где", "как"],
        )
        self.assertEqual(result["prediction"].tolist(), ["commercial", "informational"])

    def test_short_wildcard_cannot_match_an_unrelated_longer_token(self):
        self.assertEqual(marker_safety_reason("опт*"), "wildcard_stem_too_short")
        result = self.predict(
            ["где предохранитель автомобиля оптима"],
            [[0.25, 0.75]],
            {"commercial": [], "informational": [], "neutral": []},
            commercial_markers=["опт*"],
            weak_question_markers=["где"],
        )
        self.assertEqual(result["prediction"].tolist(), ["informational"])
        self.assertEqual(result["commercial_marker_hit"].tolist(), [False])

    def test_label_conflicting_strong_signal_is_rejected(self):
        frame = pd.DataFrame(
            {
                "Phrase": ["общий сигнал один", "общий сигнал два", "общий сигнал три", "общий сигнал четыре"],
                "Model Label": ["commercial", "commercial", "informational", "informational"],
            }
        )
        retained, rejected = sanitize_strong_signals(frame, ["общий сигнал"], "commercial")
        self.assertEqual(retained, [])
        self.assertEqual(rejected[0]["reason"], "reviewed_label_collision")

    def test_family_examples_are_bounded_diverse_and_deterministic(self):
        phrases = [
            "график работы пилотов",
            "вакансия свободный график",
            "сменный график вакансии хабаровск",
            "какой график у врача",
            "график выплат",
            "график отпусков образец",
            "удаленная работа гибкий график",
            "построить график функции",
            "график движения автобусов",
            "ночной график работы охранника",
        ]
        first = diverse_family_examples(phrases, "графи*", 8)
        second = diverse_family_examples(reversed(phrases), "графи*", 8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertIn("построить график функции", first)
        self.assertTrue(any("ваканс" in phrase for phrase in first))

    def test_two_token_structures_are_discovered_without_topic_hardcoding(self):
        families = phrase_intent_families("какую работу выполняет водитель")
        self.assertIn("какую* работ*", families)
        self.assertIn("какую* работ* выпол*", families)
        self.assertIn("работ* выпол*", families)
        self.assertIn("выпол* водит*", families)

        payload = intent_family_payload(
            {"работ* выпол*": 50, "водит*": 80},
            {
                "работ* выпол*": ["какую работу выполняет водитель"],
                "водит*": ["вакансия водителя"],
            },
            10_000,
        )
        by_pattern = {row["pattern"]: row for row in payload["families"]}
        self.assertEqual(by_pattern["работ* выпол*"]["kind"], "structural")
        self.assertTrue(by_pattern["работ* выпол*"]["safe_decisive_lexical"])
        self.assertFalse(by_pattern["водит*"]["safe_decisive_lexical"])

    def test_structural_family_bridges_a_short_preposition(self):
        families = phrase_intent_families("заявление на работу бухгалтером")
        self.assertIn("заявл* работ*", families)

    def test_document_family_is_not_displaced_by_high_frequency_topic_words(self):
        counts = {
            f"слово{index}*": 10_000 - index
            for index in range(100)
        }
        counts.update(
            {
                f"часто{index}* слово{index}*": 8_000 - index
                for index in range(60)
            }
        )
        counts["заявл*"] = 161
        counts["заявл* работ*"] = 56
        payload = intent_family_payload(
            counts,
            {
                "заявл*": ["заявление на работу"],
                "заявл* работ*": ["заявление на работу бухгалтером"],
            },
            151_662,
        )
        patterns = {row["pattern"] for row in payload["families"]}
        self.assertIn("заявл*", patterns)
        self.assertIn("заявл* работ*", patterns)
        document = next(row for row in payload["families"] if row["pattern"] == "заявл*")
        self.assertEqual(document["discovery_priority"], "document_reference")
        self.assertFalse(document["safe_decisive_lexical"])

    def test_family_coverage_rows_reserve_real_document_examples(self):
        phrases = [
            "заявление на работу бухгалтером",
            "заявление о приеме на работу",
            "заявление на удаленную работу образец",
            "подать заявление работодателю",
            "вакансия бухгалтера",
        ]
        grouped = pd.DataFrame(
            {
                "Phrase": phrases,
                "Normalized": phrases,
                "Occurrences": [1] * len(phrases),
                "Search Volume": [0] * len(phrases),
                "Source Row": list(range(2, 2 + len(phrases))),
                "Length": [len(value) for value in phrases],
                "Priority": [0.0] * len(phrases),
            }
        )
        payload = {
            "families": [
                {
                    "pattern": "заявл*",
                    "occurrences": 161,
                    "discovery_priority": "document_reference",
                    "examples": phrases[:4],
                }
            ]
        }
        coverage = intent_family_coverage_rows(grouped, payload, 50)
        self.assertEqual(len(coverage), 4)
        self.assertEqual(set(coverage["Coverage Family"]), {"заявл*"})

    def test_prepare_job_persists_mandatory_family_coverage_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "phrases.xlsx"
            phrases = [
                f"заявление на работу специалистом {index}"
                for index in range(20)
            ] + [f"вакансия специалиста номер {index}" for index in range(581)]
            pd.DataFrame({"query": phrases}).to_excel(input_path, index=False)
            with patch("seo_prepare.JOBS_DIR", root / "jobs"), patch(
                "seo_prepare.representative_sample",
                side_effect=lambda table, sample_size, random_state=42: table.head(sample_size).copy(),
            ):
                job = prepare_job(
                    str(input_path),
                    "поиск работы",
                    sample_size=100,
                    job_name="coverage-smoke",
                    reuse_topic="none",
                )
            labels = pd.read_excel(job / "model_labels.xlsx")
            candidates = json.loads(
                (job / "intent_family_candidates.json").read_text(encoding="utf-8")
            )["families"]
        patterns = {row["pattern"] for row in candidates}
        self.assertIn("заявл*", patterns)
        self.assertIn("заявл* работ*", patterns)
        coverage = labels[
            labels["Knowledge Source"].eq("current family coverage sample")
        ]
        self.assertGreaterEqual(
            int(coverage["Phrase"].str.contains("заявлен", case=False, na=False).sum()),
            4,
        )

    def test_family_coverage_batch_is_mandatory_before_the_run(self):
        frame = pd.DataFrame(
            {
                "Sample ID": [f"row-{index:06d}" for index in range(151)],
                "Phrase": [f"phrase {index}" for index in range(151)],
                "Model Label": ["commercial"] * 75 + ["informational"] * 75 + [""],
                "Model Confidence": [0.9] * 150 + [""],
                "Knowledge Source": ["current representative sample"] * 150
                + ["current family coverage sample"],
                "Coverage Family": [""] * 150 + ["заявл*"],
            }
        )
        status = {
            "inspection": {"unique_normalized_phrases": 10_000},
            "label_quality": {
                "valid_labeled_rows": 150,
                "sample_rows": 151,
                "garbage_share": 0.0,
                "high_priority_unreviewed": 0,
            },
            "labeled_counts": {"commercial": 75, "informational": 75, "garbage": 0},
            "blocking_errors": [],
            "unexpected_python_files": [],
            "ready_for_supervised_run": True,
        }
        with patch("seo_workflow.job_status", return_value=status), patch(
            "seo_workflow.read_state", return_value={"stage": "intent_family_audit"}
        ), patch("seo_workflow.load_label_sheet", return_value=frame), patch(
            "seo_workflow.intent_family_status",
            return_value={"total": 100, "labeled": 100, "remaining": 0, "pending": [], "labels": {}},
        ):
            action = next_action(Path("unused-job"))
        self.assertEqual(action["stage"], "intent_family_coverage")
        self.assertEqual(action["action"], "label_unrepresented_family_examples")

        self.assertEqual(family_coverage_status(frame)["remaining"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            frame.to_excel(job / "model_labels.xlsx", sheet_name="Model labels", index=False)
            (job / "state.json").write_text(json.dumps({"topic": "test"}), encoding="utf-8")
            batch = select_family_coverage_review(job, 50)
        self.assertEqual(batch["rows"][0]["family"], "заявл*")

    def test_unsafe_lexical_topic_noun_is_forced_neutral(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "intent_family_candidates.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "families": [
                            {
                                "id": "IF0001",
                                "pattern": "водит*",
                                "kind": "lexical",
                                "safe_decisive_lexical": False,
                            },
                            {
                                "id": "IF0002",
                                "pattern": "работ* выпол*",
                                "kind": "structural",
                                "safe_decisive_lexical": True,
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (job / "job_config.json").write_text(
                json.dumps({"intent": {"weak_question_markers": []}}),
                encoding="utf-8",
            )
            result = apply_intent_family_labels_inline(
                job, "IF0001|commercial;IF0002|informational"
            )
            config = json.loads((job / "job_config.json").read_text(encoding="utf-8"))
            self.assertIn("водит*", config["intent"]["family_rules"]["neutral"])
            self.assertIn("работ* выпол*", config["intent"]["family_rules"]["informational"])
            self.assertEqual(result["forced_neutral_unsafe_lexical_families"], 1)

    def test_final_policy_regression_guard_rejects_broader_false_positives(self):
        baseline = {
            "eligible": True,
            "quality_score": 0.40,
            "informational_false_positive": 0.05,
            "commercial_false_positive": 0.04,
        }
        candidate = {
            "eligible": True,
            "quality_score": 0.38,
            "informational_false_positive": 0.24,
            "commercial_false_positive": 0.04,
        }
        accepted, reasons = final_policy_candidate_is_safe(baseline, candidate)
        self.assertFalse(accepted)
        self.assertIn("informational_false_positive_regressed", reasons)

    def test_large_core_cannot_run_with_twelve_commercial_examples(self):
        frame = pd.DataFrame(
            {
                "Phrase": [f"phrase {index}" for index in range(800)],
                "Model Label": ["commercial"] * 12
                + ["informational"] * 414
                + ["garbage"] * 374,
                "Model Confidence": [0.9] * 800,
                "Knowledge Source": ["current representative sample"] * 800,
            }
        )
        status = {
            "inspection": {"unique_normalized_phrases": 151_620},
            "label_quality": {
                "valid_labeled_rows": 800,
                "sample_rows": 800,
                "garbage_share": 374 / 800,
                "high_priority_unreviewed": 0,
            },
            "labeled_counts": {
                "commercial": 12,
                "informational": 414,
                "garbage": 374,
            },
            "blocking_errors": [],
            "unexpected_python_files": [],
            "ready_for_supervised_run": True,
        }
        state = {"stage": "active_review"}
        with patch("seo_workflow.job_status", return_value=status), patch(
            "seo_workflow.read_state", return_value=state
        ), patch("seo_workflow.load_label_sheet", return_value=frame):
            action = next_action(Path("unused-job"))
        self.assertEqual(action["status"], "blocked")
        self.assertEqual(action["stage"], "intent_calibration")
        self.assertEqual(action["current_intent_counts"]["commercial"], 12)

    def test_saved_knowledge_does_not_satisfy_current_intent_calibration(self):
        frame = pd.DataFrame(
            {
                "Phrase": ["current vacancy", "prior vacancy", "prior guide"],
                "Model Label": ["commercial", "commercial", "informational"],
                "Knowledge Source": [
                    "current representative sample",
                    "review_corrected",
                    "model_reviewed",
                ],
            }
        )
        self.assertEqual(
            current_sample_intent_counts(frame),
            {"commercial": 1, "informational": 0, "garbage": 0},
        )

    def test_relevance_gate_still_precedes_intent(self):
        bundle = {
            "structural": DummyStructural(),
            "relevance_classifier": DummyRelevanceClassifier([[0.80, 0.20]]),
            "intent_classifier": DummyIntentClassifier([[0.90, 0.10]]),
            "settings": {
                "tfidf_weight": 1.0,
                "intent_tfidf_weight": 1.0,
                "garbage_threshold": 0.45,
                "quarantine_margin": 0.10,
            },
        }
        result = predict_two_stage(
            bundle,
            pd.Series(["устройство работает без интернета"]),
            np.zeros((1, 2), dtype=float),
            {"intent": {"default": "commercial"}, "intent_policy": {}},
        )
        self.assertEqual(result["prediction"].tolist(), ["garbage"])

    def test_family_audit_uses_pre_override_predictions(self):
        frame = pd.DataFrame(
            {
                "Phrase": ["вакансия свободный график", "сменный график вакансии"],
                "Intent": ["informational", "informational"],
                "Pre-family intent": ["commercial", "commercial"],
                "Intent family conflict": [True, True],
                "Intent family override": [False, False],
            }
        )
        audit = build_intent_family_audit(
            [frame],
            {
                "intent": {
                    "family_rules": {
                        "commercial": [],
                        "informational": ["графи*"],
                        "neutral": [],
                    }
                }
            },
        )
        self.assertEqual(audit.iloc[0]["Audit status"], "review")
        self.assertEqual(int(audit.iloc[0]["Pre-family commercial"]), 2)
        self.assertEqual(int(audit.iloc[0]["Family conflicts"]), 2)

    def test_probability_columns_are_exported_on_all_phrase_sheets(self):
        frame = pd.DataFrame(
            {
                "Phrase": ["пример"],
                "Search Volume": [10],
                "Intent": ["commercial"],
                "Cluster": ["C001"],
                "Source Row": [2],
                "P(commercial)": [0.876543],
                "P(informational)": [0.102345],
                "P(garbage)": [0.021112],
            }
        )
        expected_probability_columns = {
            "P(commercial)",
            "P(informational)",
            "P(garbage)",
        }
        for result_type, expected_columns in (
            ("core", CORE_RESULT_COLUMNS),
            ("garbage", GARBAGE_RESULT_COLUMNS),
            ("human_review", HUMAN_REVIEW_RESULT_COLUMNS),
        ):
            exported = compact_result_frame(frame, result_type)
            self.assertEqual(exported.columns.tolist(), expected_columns)
            self.assertTrue(expected_probability_columns.issubset(exported.columns))
            self.assertEqual(float(exported.iloc[0]["P(commercial)"]), 0.8765)


if __name__ == "__main__":
    unittest.main()
