"""Tests for building enterprise ticket benchmark specs from labeled CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from conversational_clustering.evaluation.benchmark_builders import (
    build_enterprise_ticket_benchmark_spec,
    load_labeled_csv_rows,
    write_benchmark_spec,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sample_rows() -> list[dict[str, str]]:
    return [
        {
            "ticket_text": "I need access to the shared finance drive.",
            "department": "Finance",
            "ticket_category": "invoice_issue",
            "intent_group": "finance_procurement",
            "issue_type": "payment_delay",
            "sub_intent": "invoice_mismatch",
            "intent_label_coarse": "finance_procurement",
            "intent_label_normal": "finance_procurement__payment_delay",
            "intent_label_fine": "finance_procurement__payment_delay__invoice_mismatch",
            "organizational_label_coarse": "finance",
            "organizational_label_normal": "finance__accounts_payable",
            "organizational_label_fine": "finance__accounts_payable__shared_services",
            "system_name": "SAP",
            "severity": "moderate",
        },
        {
            "ticket_text": "My Okta MFA prompt never arrives.",
            "department": "IT",
            "ticket_category": "account_issue",
            "intent_group": "access_identity",
            "issue_type": "mfa_problem",
            "sub_intent": "mfa_problem",
            "intent_label_coarse": "access_identity",
            "intent_label_normal": "access_identity__mfa_problem",
            "intent_label_fine": "access_identity__mfa_problem__delivery_failure",
            "organizational_label_coarse": "it",
            "organizational_label_normal": "it__identity_access",
            "organizational_label_fine": "it__identity_access__okta_admin",
            "system_name": "Okta",
            "severity": "major",
        },
        {
            "ticket_text": "Badge access stopped working at the side entrance.",
            "department": "Facilities",
            "ticket_category": "badge_issue",
            "intent_group": "access_identity",
            "issue_type": "badge_not_working",
            "sub_intent": "badge_not_working",
            "intent_label_coarse": "access_identity",
            "intent_label_normal": "access_identity__badge_not_working",
            "intent_label_fine": "access_identity__badge_not_working__side_entrance",
            "organizational_label_coarse": "facilities",
            "organizational_label_normal": "facilities__physical_security",
            "organizational_label_fine": "facilities__physical_security__badge_operations",
            "system_name": "Badge System",
            "severity": "minor",
        },
    ]


def test_build_enterprise_ticket_benchmark_spec_uses_default_themes() -> None:
    payload = build_enterprise_ticket_benchmark_spec(_sample_rows())

    assert payload["dataset_name"] == "enterprise_ticket"
    assert payload["items"] == [row["ticket_text"] for row in _sample_rows()]
    assert [theme["theme_id"] for theme in payload["themes"]] == [
        "department",
        "ticket_category",
        "intent_group",
        "issue_type",
        "sub_intent",
        "system_name",
        "organizational_label_coarse",
        "organizational_label_normal",
        "organizational_label_fine",
        "intent_label_coarse",
        "intent_label_normal",
        "intent_label_fine",
    ]
    assert [reference["reference_id"] for reference in payload["references"]] == [
        "department__normal",
        "ticket_category__normal",
        "intent_group__normal",
        "issue_type__normal",
        "sub_intent__normal",
        "system_name__normal",
        "organizational_label_coarse__coarse",
        "organizational_label_normal__normal",
        "organizational_label_fine__fine",
        "intent_label_coarse__coarse",
        "intent_label_normal__normal",
        "intent_label_fine__fine",
    ]

    department_reference = payload["references"][0]
    assert department_reference["cluster_names"] == ["Facilities", "Finance", "IT"]
    assert department_reference["assignment"] == {
        "I need access to the shared finance drive.": 1,
        "My Okta MFA prompt never arrives.": 2,
        "Badge access stopped working at the side entrance.": 0,
    }

    organizational_normal_theme = payload["themes"][7]
    assert organizational_normal_theme["label"] == "Organizational (Normal)"
    assert organizational_normal_theme["pov_group_id"] == "organizational"
    assert organizational_normal_theme["granularity"] == "normal"
    assert organizational_normal_theme["guidance"].startswith(
        "Group tickets by the primary team area responsible"
    )

    intent_fine_reference = payload["references"][-1]
    assert intent_fine_reference["granularity"] == "fine"
    assert intent_fine_reference["pov_group_id"] == "intent"
    assert intent_fine_reference["granularity_level"] == "fine"
    assert intent_fine_reference["cluster_names"] == sorted(
        [
            "access_identity__badge_not_working__side_entrance",
            "access_identity__mfa_problem__delivery_failure",
            "finance_procurement__payment_delay__invoice_mismatch",
        ]
    )


def test_build_enterprise_ticket_benchmark_spec_supports_custom_theme_columns() -> None:
    payload = build_enterprise_ticket_benchmark_spec(
        _sample_rows(),
        theme_columns=["department", "severity"],
    )

    assert [theme["theme_id"] for theme in payload["themes"]] == ["department", "severity"]
    assert payload["themes"][1]["label"] == "Severity"
    assert payload["themes"][1]["guidance"] == "Group tickets by severity."
    assert payload["references"][1]["reference_id"] == "severity__normal"
    assert payload["references"][1]["cluster_names"] == ["major", "minor", "moderate"]


def test_build_enterprise_ticket_benchmark_spec_rejects_duplicate_ticket_texts() -> None:
    duplicate_rows = _sample_rows()
    duplicate_rows[1]["ticket_text"] = duplicate_rows[0]["ticket_text"]

    with pytest.raises(ValueError, match="Duplicate item text"):
        build_enterprise_ticket_benchmark_spec(duplicate_rows)


def test_load_and_write_benchmark_spec_round_trip(tmp_path: Path) -> None:
    csv_path = tmp_path / "enterprise.csv"
    benchmark_path = tmp_path / "enterprise.benchmark.json"
    _write_csv(csv_path, _sample_rows())

    rows = load_labeled_csv_rows(csv_path)
    payload = build_enterprise_ticket_benchmark_spec(
        rows,
        theme_columns=["department", "system_name"],
    )
    write_benchmark_spec(benchmark_path, payload)

    with open(benchmark_path, encoding="utf-8") as handle:
        written = json.load(handle)

    assert written["dataset_name"] == "enterprise_ticket"
    assert [theme["theme_id"] for theme in written["themes"]] == ["department", "system_name"]
    assert len(written["items"]) == 3
    assert len(written["references"]) == 2


def test_load_labeled_csv_rows_applies_limit_with_deterministic_seed(tmp_path: Path) -> None:
    csv_path = tmp_path / "enterprise.csv"
    _write_csv(csv_path, _sample_rows())

    sampled_once = load_labeled_csv_rows(csv_path, limit=2, random_seed=7)
    sampled_twice = load_labeled_csv_rows(csv_path, limit=2, random_seed=7)
    sampled_other_seed = load_labeled_csv_rows(csv_path, limit=2, random_seed=11)

    assert len(sampled_once) == 2
    assert sampled_once == sampled_twice
    assert sampled_once != sampled_other_seed
