"""Enterprise-ticket metadata sampling, validation, and summary helpers."""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, cast

from dataset_generation.sampling import (
    apply_weight_adjustments,
    build_balanced_coverage_plan,
    copy_weights,
    format_ratio,
    pick_one,
    weighted_choice,
)
from dataset_generation.schema import declared_field_values, validate_record_against_schema

from .alignment import ACCESS_RELATED_ISSUE_TYPES
from .guidance import SEVERITY_RANKS, URGENCY_RANKS
from .spec import (
    ACCESS_ISSUES,
    APPROVAL_REQUIRED_ACCESS_SUB_INTENT_ADJUSTMENTS,
    APPROVAL_REQUIRED_BASE_WEIGHTS,
    APPROVAL_REQUIRED_DEPARTMENT_ADJUSTMENTS,
    APPROVAL_REQUIRED_PRIVILEGED_ACCESS_SUB_INTENT_ADJUSTMENTS,
    APPROVAL_REQUIRED_TICKET_CATEGORY_ADJUSTMENTS,
    APPROVAL_REQUIRED_TICKET_CATEGORY_OVERRIDES,
    BUSINESS_EVENT_CONTEXT_ALLOWED_DEPARTMENTS,
    BUSINESS_EVENT_CONTEXT_BASE_WEIGHTS,
    BUSINESS_EVENT_CONTEXT_CATEGORY_ADJUSTMENTS,
    BUSINESS_EVENT_CONTEXT_DEPARTMENT_ADJUSTMENTS,
    BUSINESS_EVENT_CONTEXT_PREFERRED_DEPARTMENTS,
    BUSINESS_EVENT_CONTEXT_SUB_INTENT_ADJUSTMENTS,
    CATEGORY_WEIGHTS,
    CLARITY_LEVEL_BASE_WEIGHTS,
    CLARITY_LEVEL_REQUESTER_ROLE_ADJUSTMENTS,
    CLARITY_LEVEL_WRITING_STYLE_ADJUSTMENTS,
    CLARITY_LEVEL_WRITING_STYLE_OVERRIDES,
    CRITICAL_SECURITY_ISSUES,
    DEPARTMENT_WEIGHTS,
    DEPENDENCY_TYPE_APPROVAL_REQUIRED_ADJUSTMENTS,
    DEPENDENCY_TYPE_BASE_WEIGHTS,
    DEPENDENCY_TYPE_DOCUMENT_SUB_INTENT_ADJUSTMENTS,
    DEPENDENCY_TYPE_SYSTEM_SYNC_SUB_INTENT_ADJUSTMENTS,
    DEPENDENCY_TYPE_VENDOR_SUB_INTENT_ADJUSTMENTS,
    DEPENDENCY_TYPE_WORKFLOW_STAGE_ADJUSTMENTS,
    DEPENDENCY_TYPE_WORKFLOW_STAGE_OVERRIDES,
    DIFFICULTY_PROFILES,
    GRANULARITY_LABEL_FIELDS,
    IMPACT_SCOPE_BASE_WEIGHTS,
    IMPACT_SCOPE_SECURITY_ADJUSTMENTS,
    IMPACT_SCOPE_TICKET_CATEGORY_ADJUSTMENTS,
    IMPACT_SCOPE_URGENCY_ADJUSTMENTS,
    ISSUE_TYPE_ALIASES,
    MAIN_POV_CROSSTABS,
    OWNING_TEAM_BASE_WEIGHTS,
    OWNING_TEAM_CATEGORY_ADJUSTMENTS,
    PRIVILEGED_ACCESS_ISSUES,
    REQUESTER_ROLE_BASE_WEIGHTS,
    REQUESTER_ROLE_CATEGORY_ADJUSTMENTS,
    REQUESTER_ROLE_DEPARTMENT_ADJUSTMENTS,
    REQUESTER_ROLE_EVENT_CONTEXT_ADJUSTMENTS,
    SCHEMA,
    SECURITY_ISSUES,
    SEVERITY_BASE_WEIGHTS_BY_URGENCY,
    SEVERITY_SECURITY_ADJUSTMENTS,
    SEVERITY_SECURITY_OVERRIDES,
    SEVERITY_TICKET_CATEGORY_ADJUSTMENTS,
    SUB_DEPARTMENT_WEIGHTS,
    SYSTEM_NAME_BASE_ADJUSTMENTS,
    SYSTEM_NAME_CATEGORY_ADJUSTMENTS,
    SYSTEM_NAME_SUB_INTENT_ADJUSTMENTS,
    SYSTEM_SYNC_ISSUES,
    TICKET_CATEGORY_EVENT_CONTEXT_ADJUSTMENTS,
    URGENCY_EVENT_CONTEXT_ADJUSTMENTS,
    VENDOR_ISSUES,
    WRITING_STYLE_BASE_WEIGHTS,
    WRITING_STYLE_DEPARTMENT_ADJUSTMENTS,
    WRITING_STYLE_HIGH_PRESSURE_ADJUSTMENTS,
    WRITING_STYLE_REQUESTER_ROLE_ADJUSTMENTS,
    DifficultyProfile,
    EnterpriseTicketRecord,
    EnterpriseTicketRecordDraft,
    Record,
    build_granularity_label,
)


@dataclass(frozen=True)
class CorePath:
    department: str
    ticket_category: str
    issue_type: str
    intent_group: str
    sub_intent: str
    business_event_context: str


@dataclass(frozen=True)
class OperationalState:
    sub_department: str
    requester_role: str
    owning_team: str
    system_name: str
    workflow_stage: str
    approval_required: str
    dependency_type: str
    urgency: str
    severity: str
    impact_scope: str


@dataclass(frozen=True)
class TextProfileState:
    writing_style: str
    clarity_level: str


@dataclass(frozen=True)
class GranularityLabelSuites:
    intent_label_coarse: str
    intent_label_normal: str
    intent_label_fine: str
    organizational_label_coarse: str
    organizational_label_normal: str
    organizational_label_fine: str


def _nested_rule(
    rules: dict[str, Any],
    *keys: str | None,
) -> dict[str, float] | None:
    current: Any = rules
    for key in keys:
        if key is None or not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    if not isinstance(current, dict):
        return None
    return current


def apply_scenario_rules(
    weights: dict[str, float],
    *rule_sets: dict[str, float] | None,
) -> dict[str, float]:
    adjusted = dict(weights)
    for rule_set in rule_sets:
        if rule_set:
            adjusted = apply_weight_adjustments(adjusted, rule_set)
    return adjusted


def apply_scenario_overrides(
    weights: dict[str, float],
    *override_sets: dict[str, float] | None,
) -> dict[str, float]:
    adjusted = dict(weights)
    for override_set in override_sets:
        if override_set:
            adjusted.update(override_set)
    return adjusted


def schema_values(field: str) -> list[str]:
    return list(SCHEMA[field]["values"])


def canonicalize_issue_type(issue_type: str) -> str:
    return ISSUE_TYPE_ALIASES.get(issue_type, issue_type)


def sub_intents_for(department: str, ticket_category: str) -> list[str]:
    return list(SCHEMA["sub_intent"]["mapping"][(department, ticket_category)])


def issue_type_for_sub_intent(sub_intent: str) -> str:
    return SCHEMA["issue_type"]["mapping"][sub_intent][0]


def intent_group_for_issue_type(issue_type: str) -> str:
    return SCHEMA["intent_group"]["mapping"][canonicalize_issue_type(issue_type)][0]


def difficulty_profile(name: str) -> DifficultyProfile:
    try:
        return DIFFICULTY_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported difficulty: {name}") from exc


def derive_granularity_label_suites(
    record: Record | EnterpriseTicketRecordDraft,
) -> dict[str, str]:
    return {
        "intent_label_coarse": build_granularity_label(record["intent_group"]),
        "intent_label_normal": build_granularity_label(
            record["intent_group"],
            record["issue_type"],
        ),
        "intent_label_fine": build_granularity_label(
            record["intent_group"],
            record["issue_type"],
            record["sub_intent"],
        ),
        "organizational_label_coarse": build_granularity_label(record["department"]),
        "organizational_label_normal": build_granularity_label(
            record["department"],
            record["sub_department"],
        ),
        "organizational_label_fine": build_granularity_label(
            record["department"],
            record["sub_department"],
            record["owning_team"],
        ),
    }


def granularity_label_suites_for_record(record: Record) -> dict[str, dict[str, str]]:
    return {
        family: {
            level: record[field_name]
            for level, field_name in GRANULARITY_LABEL_FIELDS[family].items()
        }
        for family in GRANULARITY_LABEL_FIELDS
    }


def reachable_field_values(field: str) -> list[str]:
    if field == "ticket_category":
        values: list[str] = []
        seen: set[str] = set()
        for department in SCHEMA["department"]["values"]:
            for ticket_category in valid_ticket_categories(department):
                if ticket_category not in seen:
                    seen.add(ticket_category)
                    values.append(ticket_category)
        return values
    return declared_field_values(SCHEMA, field)


def reachable_issue_targets() -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for sub_intent in declared_field_values(SCHEMA, "sub_intent"):
        issue_type = issue_type_for_sub_intent(sub_intent)
        if issue_type not in seen:
            seen.add(issue_type)
            targets.append(issue_type)
    return targets


def audit_schema_support(schema: dict[str, Any]) -> dict[str, list[str]]:
    if schema != SCHEMA:
        raise ValueError("audit_schema_support expects the fixed SCHEMA defined in this module.")

    report: dict[str, list[str]] = defaultdict(list)
    department_values = list(schema["department"]["values"])
    declared_departments = set(department_values)
    declared_requester_roles = set(schema["requester_role"]["values"])
    declared_event_contexts = set(schema["business_event_context"]["values"])
    declared_ticket_categories = {
        ticket_category
        for categories in schema["ticket_category"]["mapping"].values()
        for ticket_category in categories
    }
    declared_sub_intents = set(declared_field_values(schema, "sub_intent"))
    sub_intent_departments = {
        sub_intent: department
        for (department, _ticket_category), sub_intents in schema["sub_intent"]["mapping"].items()
        for sub_intent in sub_intents
    }
    declared_issue_types = {
        issue_type for values in schema["issue_type"]["mapping"].values() for issue_type in values
    }
    declared_intent_groups = {
        intent_group
        for values in schema["intent_group"]["mapping"].values()
        for intent_group in values
    }

    for field in ("sub_department", "owning_team", "ticket_category", "system_name"):
        mapping = schema[field]["mapping"]
        for department in department_values:
            if department not in mapping:
                report["missing_department_mapping"].append(f"{field}: {department}")

    for department in department_values:
        if department not in DEPARTMENT_WEIGHTS:
            report["department_without_weight"].append(department)
        elif DEPARTMENT_WEIGHTS[department] <= 0:
            report["unreachable_department"].append(department)
    for department in DEPARTMENT_WEIGHTS:
        if department not in declared_departments:
            report["unexpected_department_weight"].append(department)

    for department, categories in schema["ticket_category"]["mapping"].items():
        category_weights = CATEGORY_WEIGHTS.get(department)
        if category_weights is None:
            report["department_without_category_weights"].append(department)
            continue

        declared_categories = set(categories)
        weighted_categories = set(category_weights)
        for ticket_category in sorted(declared_categories - weighted_categories):
            report["ticket_category_without_weight"].append(f"{department} -> {ticket_category}")
        for ticket_category in sorted(weighted_categories - declared_categories):
            report["unexpected_ticket_category_weight"].append(f"{department} -> {ticket_category}")

        for ticket_category in categories:
            if (department, ticket_category) not in schema["sub_intent"]["mapping"]:
                report["ticket_category_without_sub_intent"].append(
                    f"{department} -> {ticket_category}"
                )
            if category_weights.get(ticket_category, 0.0) <= 0:
                report["unreachable_ticket_category"].append(f"{department} -> {ticket_category}")

            sub_department_weights = SUB_DEPARTMENT_WEIGHTS.get((department, ticket_category))
            if sub_department_weights is None:
                report["ticket_category_without_sub_department_weights"].append(
                    f"{department} -> {ticket_category}"
                )
            else:
                declared_sub_departments = set(schema["sub_department"]["mapping"][department])
                weighted_sub_departments = set(sub_department_weights)
                for sub_department in sorted(declared_sub_departments - weighted_sub_departments):
                    report["sub_department_without_weight"].append(
                        f"{department} -> {ticket_category} -> {sub_department}"
                    )
                for sub_department in sorted(weighted_sub_departments - declared_sub_departments):
                    report["unexpected_sub_department_weight"].append(
                        f"{department} -> {ticket_category} -> {sub_department}"
                    )
                for sub_department, weight in sub_department_weights.items():
                    if weight <= 0:
                        report["unreachable_sub_department"].append(
                            f"{department} -> {ticket_category} -> {sub_department}"
                        )

    for mapping_key, sub_intents in schema["sub_intent"]["mapping"].items():
        for sub_intent in sub_intents:
            if sub_intent not in schema["issue_type"]["mapping"]:
                report["sub_intent_without_issue_type"].append(
                    f"{mapping_key[0]} -> {mapping_key[1]} -> {sub_intent}"
                )

    referenced_issue_types = set()
    for sub_intent, mapping_values in schema["issue_type"]["mapping"].items():
        for issue_type in mapping_values:
            referenced_issue_types.add(issue_type)
            if issue_type not in schema["intent_group"]["mapping"]:
                report["issue_type_without_intent_group"].append(issue_type)
        if not mapping_values:
            report["sub_intent_without_issue_type"].append(sub_intent)

    for issue_type in schema["intent_group"]["mapping"]:
        if issue_type not in referenced_issue_types:
            report["issue_type_without_sub_intent"].append(issue_type)

    for intent_group in declared_intent_groups:
        if not any(intent_group in values for values in schema["intent_group"]["mapping"].values()):
            report["intent_group_without_issue_type"].append(intent_group)

    for department, teams in schema["owning_team"]["mapping"].items():
        base_weights = OWNING_TEAM_BASE_WEIGHTS.get(department)
        if base_weights is None:
            report["department_without_owning_team_weights"].append(department)
            continue
        declared_teams = set(teams)
        weighted_teams = set(base_weights)
        for team in sorted(declared_teams - weighted_teams):
            report["owning_team_without_weight"].append(f"{department} -> {team}")
        for team in sorted(weighted_teams - declared_teams):
            report["unexpected_owning_team_weight"].append(f"{department} -> {team}")
        for team, weight in base_weights.items():
            if weight <= 0:
                report["unreachable_owning_team"].append(f"{department} -> {team}")

    for department, category_adjustments in OWNING_TEAM_CATEGORY_ADJUSTMENTS.items():
        if department not in declared_departments:
            report["unknown_owning_team_adjustment_department"].append(department)
            continue
        declared_categories = set(schema["ticket_category"]["mapping"][department])
        declared_teams = set(schema["owning_team"]["mapping"][department])
        for ticket_category, team_adjustments in category_adjustments.items():
            if ticket_category not in declared_categories:
                report["unknown_owning_team_adjustment_category"].append(
                    f"{department} -> {ticket_category}"
                )
            for team in team_adjustments:
                if team not in declared_teams:
                    report["unknown_owning_team_adjustment_team"].append(
                        f"{department} -> {ticket_category} -> {team}"
                    )

    requester_role_weight_keys = set(REQUESTER_ROLE_BASE_WEIGHTS)
    for role in sorted(declared_requester_roles - requester_role_weight_keys):
        report["requester_role_without_weight"].append(role)
    for role in sorted(requester_role_weight_keys - declared_requester_roles):
        report["unexpected_requester_role_weight"].append(role)

    for department, role_adjustments in REQUESTER_ROLE_DEPARTMENT_ADJUSTMENTS.items():
        if department not in declared_departments:
            report["unknown_requester_role_department"].append(department)
        for role in role_adjustments:
            if role not in declared_requester_roles:
                report["unknown_requester_role_department_value"].append(f"{department} -> {role}")

    for ticket_category, role_adjustments in REQUESTER_ROLE_CATEGORY_ADJUSTMENTS.items():
        if ticket_category not in declared_ticket_categories:
            report["unknown_requester_role_category"].append(ticket_category)
        for role in role_adjustments:
            if role not in declared_requester_roles:
                report["unknown_requester_role_category_value"].append(
                    f"{ticket_category} -> {role}"
                )

    declared_systems_by_department = {
        department: set(systems) for department, systems in schema["system_name"]["mapping"].items()
    }
    for department, system_adjustments in SYSTEM_NAME_BASE_ADJUSTMENTS.items():
        if department not in declared_departments:
            report["unknown_system_adjustment_department"].append(department)
            continue
        for system_name in system_adjustments:
            if system_name not in declared_systems_by_department[department]:
                report["unknown_system_adjustment_system"].append(f"{department} -> {system_name}")

    for department, category_adjustments in SYSTEM_NAME_CATEGORY_ADJUSTMENTS.items():
        if department not in declared_departments:
            report["unknown_system_adjustment_department"].append(department)
            continue
        declared_categories = set(schema["ticket_category"]["mapping"][department])
        declared_systems = declared_systems_by_department[department]
        for ticket_category, system_adjustments in category_adjustments.items():
            if ticket_category not in declared_categories:
                report["unknown_system_adjustment_category"].append(
                    f"{department} -> {ticket_category}"
                )
            for system_name in system_adjustments:
                if system_name not in declared_systems:
                    report["unknown_system_adjustment_system"].append(
                        f"{department} -> {ticket_category} -> {system_name}"
                    )

    for sub_intent, system_adjustments in SYSTEM_NAME_SUB_INTENT_ADJUSTMENTS.items():
        if sub_intent not in declared_sub_intents:
            report["unknown_system_adjustment_sub_intent"].append(sub_intent)
            continue
        department = sub_intent_departments[sub_intent]
        declared_systems = declared_systems_by_department[department]
        for system_name in system_adjustments:
            if system_name not in declared_systems:
                report["unknown_system_adjustment_system"].append(
                    f"{department} -> {sub_intent} -> {system_name}"
                )

    base_event_contexts = set(BUSINESS_EVENT_CONTEXT_BASE_WEIGHTS)
    for event_context in sorted(declared_event_contexts - base_event_contexts):
        report["event_context_without_base_weight"].append(event_context)
    for event_context in sorted(base_event_contexts - declared_event_contexts):
        report["unexpected_event_context_weight"].append(event_context)

    for event_context, allowed_departments in BUSINESS_EVENT_CONTEXT_ALLOWED_DEPARTMENTS.items():
        if event_context not in declared_event_contexts:
            report["unknown_allowed_event_context"].append(event_context)
        for department in allowed_departments:
            if department not in declared_departments:
                report["unknown_allowed_event_department"].append(
                    f"{event_context} -> {department}"
                )

    for (
        event_context,
        preferred_departments,
    ) in BUSINESS_EVENT_CONTEXT_PREFERRED_DEPARTMENTS.items():
        if event_context not in declared_event_contexts:
            report["unknown_preferred_event_context"].append(event_context)
        for department in preferred_departments:
            if department not in declared_departments:
                report["unknown_preferred_event_department"].append(
                    f"{event_context} -> {department}"
                )

    for department, event_adjustments in BUSINESS_EVENT_CONTEXT_DEPARTMENT_ADJUSTMENTS.items():
        if department not in declared_departments:
            report["unknown_event_adjustment_department"].append(department)
        for event_context in event_adjustments:
            if event_context not in declared_event_contexts:
                report["unknown_event_adjustment_context"].append(
                    f"{department} -> {event_context}"
                )

    for ticket_category, event_adjustments in BUSINESS_EVENT_CONTEXT_CATEGORY_ADJUSTMENTS.items():
        if ticket_category not in declared_ticket_categories:
            report["unknown_event_adjustment_category"].append(ticket_category)
        for event_context in event_adjustments:
            if event_context not in declared_event_contexts:
                report["unknown_event_adjustment_context"].append(
                    f"{ticket_category} -> {event_context}"
                )

    for sub_intent, event_adjustments in BUSINESS_EVENT_CONTEXT_SUB_INTENT_ADJUSTMENTS.items():
        if sub_intent not in declared_sub_intents:
            report["unknown_event_adjustment_sub_intent"].append(sub_intent)
        for event_context in event_adjustments:
            if event_context not in declared_event_contexts:
                report["unknown_event_adjustment_context"].append(
                    f"{sub_intent} -> {event_context}"
                )

    for event_context, department_adjustments in TICKET_CATEGORY_EVENT_CONTEXT_ADJUSTMENTS.items():
        if event_context not in declared_event_contexts:
            report["unknown_ticket_category_event_context"].append(event_context)
        for department, category_adjustments in department_adjustments.items():
            if department not in declared_departments:
                report["unknown_ticket_category_event_department"].append(
                    f"{event_context} -> {department}"
                )
                continue
            declared_categories = set(schema["ticket_category"]["mapping"][department])
            for ticket_category in category_adjustments:
                if ticket_category not in declared_categories:
                    report["unknown_ticket_category_event_category"].append(
                        f"{event_context} -> {department} -> {ticket_category}"
                    )

    for event_context, role_adjustments in REQUESTER_ROLE_EVENT_CONTEXT_ADJUSTMENTS.items():
        if event_context not in declared_event_contexts:
            report["unknown_requester_role_event_context"].append(event_context)
        for role in role_adjustments:
            if role not in declared_requester_roles:
                report["unknown_requester_role_event_value"].append(f"{event_context} -> {role}")

    for event_context, urgency_adjustments in URGENCY_EVENT_CONTEXT_ADJUSTMENTS.items():
        if event_context not in declared_event_contexts:
            report["unknown_urgency_event_context"].append(event_context)
        for urgency in urgency_adjustments:
            if urgency not in set(schema["urgency"]["values"]):
                report["unknown_urgency_event_value"].append(f"{event_context} -> {urgency}")

    for legacy_issue_type, canonical_issue_type in ISSUE_TYPE_ALIASES.items():
        if canonical_issue_type not in declared_issue_types:
            report["issue_type_alias_without_canonical_target"].append(
                f"{legacy_issue_type} -> {canonical_issue_type}"
            )

    return {issue_name: sorted(set(values)) for issue_name, values in report.items()}


def format_schema_support_report(report: dict[str, list[str]]) -> str:
    if not report:
        return "Schema support audit: no unsupported declared labels found."

    lines = ["Schema support audit:"]
    for issue_name in sorted(report):
        lines.append(f"  {issue_name}:")
        for value in report[issue_name]:
            lines.append(f"    - {value}")
    return "\n".join(lines)


def handle_schema_support_report(report: dict[str, list[str]], mode: str) -> None:
    if mode not in {"warn", "error", "ignore"}:
        raise ValueError(f"Unsupported schema audit mode: {mode}")
    if mode == "ignore" or not report:
        return

    message = format_schema_support_report(report)
    if mode == "error":
        raise ValueError(message)
    print(message, file=sys.stderr)


def validate_probability(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive.")
    return value


def clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def normalize_business_event_context(department: str, event_context: str) -> str:
    if not event_context_is_allowed_for_department(department, event_context):
        return "routine_operation"
    return event_context


def event_context_is_allowed_for_department(department: str, event_context: str) -> bool:
    allowed_departments = BUSINESS_EVENT_CONTEXT_ALLOWED_DEPARTMENTS.get(event_context)
    return allowed_departments is None or department in allowed_departments


def preferred_event_context_adjustments(department: str) -> dict[str, float]:
    return {
        event_context: preferred_departments[department]
        for event_context, preferred_departments in (
            BUSINESS_EVENT_CONTEXT_PREFERRED_DEPARTMENTS.items()
        )
        if department in preferred_departments
    }


def valid_ticket_categories(department: str) -> list[str]:
    categories = SCHEMA["ticket_category"]["mapping"][department]
    return [
        category
        for category in categories
        if (department, category) in SCHEMA["sub_intent"]["mapping"]
    ]


def sample_department(rng: random.Random) -> str:
    return weighted_choice(rng, DEPARTMENT_WEIGHTS)


def sample_ticket_category(
    department: str,
    rng: random.Random,
    business_event_context: str | None = None,
) -> str:
    weights = dict(CATEGORY_WEIGHTS[department])
    weights = apply_scenario_rules(
        weights,
        _nested_rule(TICKET_CATEGORY_EVENT_CONTEXT_ADJUSTMENTS, business_event_context, department),
    )
    valid_categories = valid_ticket_categories(department)
    valid_weights = {category: weights[category] for category in valid_categories}
    return weighted_choice(rng, valid_weights)


def sample_sub_intent(
    department: str,
    ticket_category: str,
    rng: random.Random,
    issue_type: str | None = None,
) -> str:
    options = sub_intents_for(department, ticket_category)
    if issue_type is None:
        return pick_one(rng, options)
    canonical_issue_type = canonicalize_issue_type(issue_type)
    matching = [
        sub_intent
        for sub_intent in options
        if issue_type_for_sub_intent(sub_intent) == canonical_issue_type
    ]
    if not matching:
        raise ValueError(
            f"No sub_intent available for {department=} {ticket_category=} {canonical_issue_type=}"
        )
    return pick_one(rng, matching)


def sample_business_event_context(
    department: str,
    ticket_category: str | None,
    sub_intent: str | None,
    rng: random.Random,
) -> str:
    weights = dict(BUSINESS_EVENT_CONTEXT_BASE_WEIGHTS)
    weights = apply_scenario_rules(
        weights,
        _nested_rule(BUSINESS_EVENT_CONTEXT_DEPARTMENT_ADJUSTMENTS, department),
        _nested_rule(BUSINESS_EVENT_CONTEXT_CATEGORY_ADJUSTMENTS, ticket_category),
        _nested_rule(BUSINESS_EVENT_CONTEXT_SUB_INTENT_ADJUSTMENTS, sub_intent),
        preferred_event_context_adjustments(department),
    )
    weights = {
        event_context: weight
        for event_context, weight in weights.items()
        if event_context_is_allowed_for_department(department, event_context)
    }
    return weighted_choice(rng, weights)


def sample_requester_role(
    department: str,
    business_event_context: str,
    ticket_category: str,
    rng: random.Random,
) -> str:
    weights = dict(REQUESTER_ROLE_BASE_WEIGHTS)
    weights = apply_scenario_rules(
        weights,
        _nested_rule(REQUESTER_ROLE_EVENT_CONTEXT_ADJUSTMENTS, business_event_context),
        _nested_rule(REQUESTER_ROLE_DEPARTMENT_ADJUSTMENTS, department),
        _nested_rule(REQUESTER_ROLE_CATEGORY_ADJUSTMENTS, ticket_category),
    )
    return weighted_choice(rng, weights)


def sample_sub_department(
    department: str,
    ticket_category: str,
    rng: random.Random,
) -> str:
    return weighted_choice(rng, SUB_DEPARTMENT_WEIGHTS[(department, ticket_category)])


def sample_owning_team(
    department: str,
    ticket_category: str,
    sub_intent: str,
    rng: random.Random,
) -> str:
    del sub_intent
    weights = dict(OWNING_TEAM_BASE_WEIGHTS[department])
    weights = apply_scenario_rules(
        weights,
        _nested_rule(OWNING_TEAM_CATEGORY_ADJUSTMENTS, department, ticket_category),
    )
    return weighted_choice(rng, weights)


def sample_system_name(
    department: str,
    ticket_category: str,
    sub_intent: str,
    rng: random.Random,
) -> str:
    systems = copy_weights(list(SCHEMA["system_name"]["mapping"][department]), 1.0)
    systems = apply_scenario_rules(
        systems,
        _nested_rule(SYSTEM_NAME_BASE_ADJUSTMENTS, department),
        _nested_rule(SYSTEM_NAME_CATEGORY_ADJUSTMENTS, department, ticket_category),
        _nested_rule(SYSTEM_NAME_SUB_INTENT_ADJUSTMENTS, sub_intent),
    )
    return weighted_choice(rng, systems)


def sample_approval_required(
    department: str,
    ticket_category: str,
    sub_intent: str,
    rng: random.Random,
) -> str:
    weights = apply_scenario_rules(
        APPROVAL_REQUIRED_BASE_WEIGHTS,
        _nested_rule(APPROVAL_REQUIRED_ACCESS_SUB_INTENT_ADJUSTMENTS, sub_intent),
        _nested_rule(APPROVAL_REQUIRED_PRIVILEGED_ACCESS_SUB_INTENT_ADJUSTMENTS, sub_intent),
    )
    weights = apply_scenario_overrides(
        weights,
        _nested_rule(APPROVAL_REQUIRED_TICKET_CATEGORY_OVERRIDES, ticket_category),
    )
    weights = apply_scenario_rules(
        weights,
        _nested_rule(APPROVAL_REQUIRED_TICKET_CATEGORY_ADJUSTMENTS, ticket_category),
        _nested_rule(APPROVAL_REQUIRED_DEPARTMENT_ADJUSTMENTS, department),
    )
    return weighted_choice(rng, weights)


def sample_workflow_stage(record: EnterpriseTicketRecordDraft, rng: random.Random) -> str:
    weights = {
        "new_submission": 0.28,
        "triaged": 0.16,
        "pending_requester_info": 0.10,
        "pending_approval": 0.13,
        "in_progress": 0.19,
        "blocked": 0.08,
        "escalated": 0.06,
    }
    if record["approval_required"] != "none":
        weights["pending_approval"] += 0.18
        weights["new_submission"] -= 0.05
    if record["ticket_category"] in {"vendor_setup", "vendor_request", "contract_purchase"}:
        weights["blocked"] += 0.05
    if record["sub_intent"] in CRITICAL_SECURITY_ISSUES:
        weights["escalated"] += 0.10
        weights["in_progress"] += 0.08
    return weighted_choice(rng, weights)


def sample_dependency_type(record: EnterpriseTicketRecordDraft, rng: random.Random) -> str:
    weights = apply_scenario_overrides(
        DEPENDENCY_TYPE_BASE_WEIGHTS,
        _nested_rule(DEPENDENCY_TYPE_WORKFLOW_STAGE_OVERRIDES, record["workflow_stage"]),
    )
    weights = apply_scenario_rules(
        weights,
        _nested_rule(DEPENDENCY_TYPE_APPROVAL_REQUIRED_ADJUSTMENTS, record["approval_required"]),
        _nested_rule(DEPENDENCY_TYPE_VENDOR_SUB_INTENT_ADJUSTMENTS, record["sub_intent"]),
        _nested_rule(DEPENDENCY_TYPE_DOCUMENT_SUB_INTENT_ADJUSTMENTS, record["sub_intent"]),
        _nested_rule(DEPENDENCY_TYPE_SYSTEM_SYNC_SUB_INTENT_ADJUSTMENTS, record["sub_intent"]),
        _nested_rule(DEPENDENCY_TYPE_WORKFLOW_STAGE_ADJUSTMENTS, record["workflow_stage"]),
    )
    return weighted_choice(rng, weights)


def sample_urgency(record: EnterpriseTicketRecordDraft, rng: random.Random) -> str:
    weights = {"low": 0.22, "medium": 0.48, "high": 0.22, "critical": 0.08}
    if record["sub_intent"] in CRITICAL_SECURITY_ISSUES:
        weights["high"] += 0.22
        weights["critical"] += 0.34
        weights["low"] = 0.01
    elif record["ticket_category"] == "incident_report":
        weights["high"] += 0.22
        weights["critical"] += 0.16
    elif record["ticket_category"] in {"access_request", "account_issue"}:
        weights["high"] += 0.08
    elif record["ticket_category"] in {"policy_clarification", "contract_review"}:
        weights["low"] += 0.08
    elif record["ticket_category"] in {
        "support_platform",
        "developer_tooling",
        "data_platform",
        "pay_cycle_issue",
    }:
        weights["high"] += 0.10
        weights["critical"] += 0.04
    elif record["ticket_category"] in {"travel_booking", "event_marketing"}:
        weights["medium"] += 0.08
        weights["high"] += 0.04

    weights = apply_scenario_rules(
        weights,
        _nested_rule(URGENCY_EVENT_CONTEXT_ADJUSTMENTS, record["business_event_context"]),
    )

    if record["workflow_stage"] == "escalated":
        weights["high"] += 0.14
        weights["critical"] += 0.08
    return weighted_choice(rng, weights)


def sample_severity(record: EnterpriseTicketRecordDraft, rng: random.Random) -> str:
    weights = dict(SEVERITY_BASE_WEIGHTS_BY_URGENCY[record["urgency"]])
    is_security_sensitive = (
        record["department"] == "Security" or record["sub_intent"] in SECURITY_ISSUES
    )
    weights = apply_scenario_overrides(
        weights,
        SEVERITY_SECURITY_OVERRIDES if is_security_sensitive else None,
    )
    weights = apply_scenario_rules(
        weights,
        SEVERITY_SECURITY_ADJUSTMENTS if is_security_sensitive else None,
        _nested_rule(SEVERITY_TICKET_CATEGORY_ADJUSTMENTS, record["ticket_category"]),
    )
    return weighted_choice(rng, weights)


def sample_impact_scope(record: EnterpriseTicketRecordDraft, rng: random.Random) -> str:
    is_security_sensitive = (
        record["department"] == "Security" or record["sub_intent"] in SECURITY_ISSUES
    )
    weights = apply_scenario_rules(
        IMPACT_SCOPE_BASE_WEIGHTS,
        _nested_rule(IMPACT_SCOPE_TICKET_CATEGORY_ADJUSTMENTS, record["ticket_category"]),
        IMPACT_SCOPE_SECURITY_ADJUSTMENTS if is_security_sensitive else None,
        _nested_rule(IMPACT_SCOPE_URGENCY_ADJUSTMENTS, record["urgency"]),
    )
    return weighted_choice(rng, weights)


def sample_writing_style(
    record: EnterpriseTicketRecordDraft,
    rng: random.Random,
    profile: DifficultyProfile | None = None,
) -> str:
    active_profile = profile or difficulty_profile("medium")
    is_high_pressure = record["urgency"] in {"high", "critical"} or record["workflow_stage"] in {
        "blocked",
        "escalated",
    }
    weights = apply_scenario_rules(
        WRITING_STYLE_BASE_WEIGHTS,
        _nested_rule(WRITING_STYLE_DEPARTMENT_ADJUSTMENTS, record["department"]),
        WRITING_STYLE_HIGH_PRESSURE_ADJUSTMENTS if is_high_pressure else None,
        _nested_rule(WRITING_STYLE_REQUESTER_ROLE_ADJUSTMENTS, record["requester_role"]),
        dict(active_profile.writing_style_adjustments),
    )
    return weighted_choice(rng, weights)


def sample_clarity_level(
    record: EnterpriseTicketRecordDraft,
    rng: random.Random,
    profile: DifficultyProfile | None = None,
) -> str:
    active_profile = profile or difficulty_profile("medium")
    weights = apply_scenario_overrides(
        CLARITY_LEVEL_BASE_WEIGHTS,
        _nested_rule(CLARITY_LEVEL_WRITING_STYLE_OVERRIDES, record["writing_style"]),
    )
    weights = apply_scenario_rules(
        weights,
        _nested_rule(CLARITY_LEVEL_WRITING_STYLE_ADJUSTMENTS, record["writing_style"]),
        _nested_rule(CLARITY_LEVEL_REQUESTER_ROLE_ADJUSTMENTS, record["requester_role"]),
        dict(active_profile.clarity_adjustments),
    )
    return weighted_choice(rng, weights)


def sample_issue_path(
    department: str,
    rng: random.Random,
    *,
    business_event_context: str,
    ticket_category: str | None = None,
    issue_type: str | None = None,
) -> tuple[str, str, str]:
    canonical_issue_type = canonicalize_issue_type(issue_type) if issue_type is not None else None
    selected_ticket_category = ticket_category or sample_ticket_category(
        department,
        rng,
        business_event_context=business_event_context,
    )
    sub_intent = sample_sub_intent(
        department,
        selected_ticket_category,
        rng,
        issue_type=canonical_issue_type,
    )
    selected_issue_type = canonical_issue_type or issue_type_for_sub_intent(sub_intent)
    return selected_ticket_category, sub_intent, selected_issue_type


def sample_core_path(
    department: str,
    rng: random.Random,
    *,
    business_event_context: str | None = None,
) -> CorePath:
    selected_event_context = business_event_context
    if selected_event_context is None:
        selected_event_context = sample_business_event_context(
            department,
            None,
            None,
            rng,
        )
    normalized_event_context = normalize_business_event_context(department, selected_event_context)
    ticket_category, sub_intent, issue_type = sample_issue_path(
        department,
        rng,
        business_event_context=normalized_event_context,
    )
    return CorePath(
        department=department,
        ticket_category=ticket_category,
        issue_type=issue_type,
        intent_group=intent_group_for_issue_type(issue_type),
        sub_intent=sub_intent,
        business_event_context=normalized_event_context,
    )


def sample_core_path_for_issue_target(
    department: str,
    ticket_category: str,
    issue_type: str,
    rng: random.Random,
) -> CorePath:
    canonical_issue_type = canonicalize_issue_type(issue_type)
    _, sub_intent, _ = sample_issue_path(
        department,
        rng,
        business_event_context="routine_operation",
        ticket_category=ticket_category,
        issue_type=canonical_issue_type,
    )
    business_event_context = normalize_business_event_context(
        department,
        sample_business_event_context(
            department,
            ticket_category,
            sub_intent,
            rng,
        ),
    )
    return CorePath(
        department=department,
        ticket_category=ticket_category,
        issue_type=canonical_issue_type,
        intent_group=intent_group_for_issue_type(canonical_issue_type),
        sub_intent=sub_intent,
        business_event_context=business_event_context,
    )


def _access_approval_choice(core_path: CorePath, rng: random.Random) -> str:
    if core_path.department == "Security" or core_path.sub_intent in PRIVILEGED_ACCESS_ISSUES:
        return weighted_choice(
            rng,
            {"manager_approval": 0.35, "security_approval": 0.35, "multi_step_approval": 0.30},
        )
    return weighted_choice(
        rng,
        {"manager_approval": 0.58, "security_approval": 0.20, "multi_step_approval": 0.22},
    )


def _purchase_approval_choice(rng: random.Random) -> str:
    return weighted_choice(
        rng,
        {
            "manager_approval": 0.22,
            "finance_approval": 0.34,
            "security_approval": 0.06,
            "multi_step_approval": 0.38,
        },
    )


def _pending_approval_choice(rng: random.Random) -> str:
    return weighted_choice(
        rng,
        {
            "manager_approval": 0.52,
            "finance_approval": 0.20,
            "security_approval": 0.08,
            "multi_step_approval": 0.20,
        },
    )


def ensure_approval_required(
    core_path: CorePath,
    approval_required: str,
    rng: random.Random,
    *,
    workflow_stage: str | None = None,
    dependency_type: str | None = None,
) -> str:
    adjusted_approval = approval_required
    if adjusted_approval == "none" and core_path.sub_intent in ACCESS_ISSUES:
        adjusted_approval = _access_approval_choice(core_path, rng)
    if adjusted_approval == "none" and core_path.ticket_category in {
        "purchase_request",
        "purchase_approval",
        "contract_purchase",
    }:
        adjusted_approval = _purchase_approval_choice(rng)
    if adjusted_approval == "none" and workflow_stage == "pending_approval":
        adjusted_approval = _pending_approval_choice(rng)
    if adjusted_approval == "none" and dependency_type == "waiting_for_manager":
        adjusted_approval = "manager_approval"
    return adjusted_approval


def ensure_dependency_type(
    core_path: CorePath,
    workflow_stage: str,
    dependency_type: str,
    rng: random.Random,
) -> str:
    if workflow_stage != "blocked" or dependency_type != "none":
        return dependency_type

    dependency_weights = {
        "waiting_for_document": 0.18,
        "waiting_for_manager": 0.18,
        "waiting_for_vendor": 0.20 if core_path.sub_intent in VENDOR_ISSUES else 0.08,
        "waiting_for_system_sync": 0.20 if core_path.sub_intent in SYSTEM_SYNC_ISSUES else 0.10,
        "waiting_for_another_team": 0.34,
    }
    return weighted_choice(rng, dependency_weights)


def ensure_urgency(
    core_path: CorePath,
    workflow_stage: str,
    urgency: str,
    rng: random.Random,
) -> str:
    adjusted_urgency = urgency
    if workflow_stage == "escalated" and adjusted_urgency == "low":
        adjusted_urgency = "medium"
    if (
        core_path.department == "Security" or core_path.sub_intent in SECURITY_ISSUES
    ) and urgency_rank(adjusted_urgency) < urgency_rank("high"):
        adjusted_urgency = weighted_choice(rng, {"high": 0.74, "critical": 0.26})
    return adjusted_urgency


def ensure_severity(
    core_path: CorePath,
    urgency: str,
    severity: str,
    rng: random.Random,
) -> tuple[str, str]:
    adjusted_urgency = urgency
    adjusted_severity = severity

    if (
        core_path.department == "Security" or core_path.sub_intent in SECURITY_ISSUES
    ) and severity_rank(adjusted_severity) < severity_rank("major"):
        adjusted_severity = weighted_choice(rng, {"major": 0.68, "business_critical": 0.32})

    if adjusted_urgency == "high" and severity_rank(adjusted_severity) < severity_rank("moderate"):
        adjusted_severity = "moderate"
    elif adjusted_urgency == "critical" and severity_rank(adjusted_severity) < severity_rank(
        "major"
    ):
        adjusted_severity = weighted_choice(rng, {"major": 0.45, "business_critical": 0.55})

    if adjusted_severity == "business_critical" and adjusted_urgency in {"low", "medium"}:
        adjusted_urgency = "high"

    return adjusted_urgency, adjusted_severity


def ensure_impact_scope(
    urgency: str,
    impact_scope: str,
    rng: random.Random,
) -> tuple[str, str]:
    adjusted_urgency = urgency
    adjusted_impact_scope = impact_scope

    if adjusted_impact_scope == "company_wide" and adjusted_urgency == "low":
        adjusted_urgency = "medium"
    if adjusted_urgency == "critical" and adjusted_impact_scope == "single_user":
        adjusted_impact_scope = weighted_choice(
            rng,
            {
                "single_team": 0.42,
                "single_department": 0.32,
                "multi_department": 0.18,
                "company_wide": 0.08,
            },
        )

    return adjusted_urgency, adjusted_impact_scope


def sample_operational_state(
    core_path: CorePath,
    rng: random.Random,
) -> OperationalState:
    sub_department = sample_sub_department(core_path.department, core_path.ticket_category, rng)
    requester_role = sample_requester_role(
        core_path.department,
        core_path.business_event_context,
        core_path.ticket_category,
        rng,
    )
    owning_team = sample_owning_team(
        core_path.department,
        core_path.ticket_category,
        core_path.sub_intent,
        rng,
    )
    system_name = sample_system_name(
        core_path.department,
        core_path.ticket_category,
        core_path.sub_intent,
        rng,
    )

    approval_required = ensure_approval_required(
        core_path,
        sample_approval_required(
            core_path.department,
            core_path.ticket_category,
            core_path.sub_intent,
            rng,
        ),
        rng,
    )

    base_record: EnterpriseTicketRecordDraft = {
        "department": core_path.department,
        "ticket_category": core_path.ticket_category,
        "issue_type": core_path.issue_type,
        "intent_group": core_path.intent_group,
        "sub_intent": core_path.sub_intent,
        "business_event_context": core_path.business_event_context,
        "sub_department": sub_department,
        "requester_role": requester_role,
        "owning_team": owning_team,
        "system_name": system_name,
        "approval_required": approval_required,
    }
    workflow_stage = sample_workflow_stage(base_record, rng)
    approval_required = ensure_approval_required(
        core_path,
        approval_required,
        rng,
        workflow_stage=workflow_stage,
    )

    dependency_record = cast(EnterpriseTicketRecordDraft, dict(base_record))
    dependency_record["workflow_stage"] = workflow_stage
    dependency_record["approval_required"] = approval_required
    dependency_type = ensure_dependency_type(
        core_path,
        workflow_stage,
        sample_dependency_type(dependency_record, rng),
        rng,
    )
    approval_required = ensure_approval_required(
        core_path,
        approval_required,
        rng,
        workflow_stage=workflow_stage,
        dependency_type=dependency_type,
    )

    urgency_record = cast(EnterpriseTicketRecordDraft, dict(dependency_record))
    urgency_record["dependency_type"] = dependency_type
    urgency = ensure_urgency(
        core_path,
        workflow_stage,
        sample_urgency(urgency_record, rng),
        rng,
    )

    severity_record = cast(EnterpriseTicketRecordDraft, dict(urgency_record))
    severity_record["urgency"] = urgency
    urgency, severity = ensure_severity(
        core_path,
        urgency,
        sample_severity(severity_record, rng),
        rng,
    )

    impact_record = cast(EnterpriseTicketRecordDraft, dict(severity_record))
    impact_record["severity"] = severity
    impact_scope = sample_impact_scope(impact_record, rng)
    urgency, impact_scope = ensure_impact_scope(urgency, impact_scope, rng)

    return OperationalState(
        sub_department=sub_department,
        requester_role=requester_role,
        owning_team=owning_team,
        system_name=system_name,
        workflow_stage=workflow_stage,
        approval_required=approval_required,
        dependency_type=dependency_type,
        urgency=urgency,
        severity=severity,
        impact_scope=impact_scope,
    )


def sample_text_profile_state(
    core_path: CorePath,
    operational_state: OperationalState,
    rng: random.Random,
    profile: DifficultyProfile,
) -> TextProfileState:
    record_without_text: EnterpriseTicketRecordDraft = {
        "department": core_path.department,
        "sub_department": operational_state.sub_department,
        "requester_role": operational_state.requester_role,
        "owning_team": operational_state.owning_team,
        "ticket_category": core_path.ticket_category,
        "issue_type": core_path.issue_type,
        "intent_group": core_path.intent_group,
        "sub_intent": core_path.sub_intent,
        "system_name": operational_state.system_name,
        "business_event_context": core_path.business_event_context,
        "workflow_stage": operational_state.workflow_stage,
        "approval_required": operational_state.approval_required,
        "dependency_type": operational_state.dependency_type,
        "urgency": operational_state.urgency,
        "severity": operational_state.severity,
        "impact_scope": operational_state.impact_scope,
    }
    writing_style = sample_writing_style(record_without_text, rng, profile)
    clarity_record = cast(EnterpriseTicketRecordDraft, dict(record_without_text))
    clarity_record["writing_style"] = writing_style
    clarity_level = sample_clarity_level(clarity_record, rng, profile)
    return TextProfileState(
        writing_style=writing_style,
        clarity_level=clarity_level,
    )


def build_record_from_stages(
    core_path: CorePath,
    operational_state: OperationalState,
    text_profile_state: TextProfileState,
    granularity_label_suites: GranularityLabelSuites,
) -> Record:
    return cast(
        Record,
        EnterpriseTicketRecord(
            department=core_path.department,
            sub_department=operational_state.sub_department,
            requester_role=operational_state.requester_role,
            owning_team=operational_state.owning_team,
            ticket_category=core_path.ticket_category,
            issue_type=core_path.issue_type,
            intent_group=core_path.intent_group,
            sub_intent=core_path.sub_intent,
            intent_label_coarse=granularity_label_suites.intent_label_coarse,
            intent_label_normal=granularity_label_suites.intent_label_normal,
            intent_label_fine=granularity_label_suites.intent_label_fine,
            organizational_label_coarse=granularity_label_suites.organizational_label_coarse,
            organizational_label_normal=granularity_label_suites.organizational_label_normal,
            organizational_label_fine=granularity_label_suites.organizational_label_fine,
            system_name=operational_state.system_name,
            business_event_context=core_path.business_event_context,
            workflow_stage=operational_state.workflow_stage,
            approval_required=operational_state.approval_required,
            dependency_type=operational_state.dependency_type,
            urgency=operational_state.urgency,
            severity=operational_state.severity,
            impact_scope=operational_state.impact_scope,
            writing_style=text_profile_state.writing_style,
            clarity_level=text_profile_state.clarity_level,
        ),
    )


def sample_complete_record(
    core_path: CorePath,
    rng: random.Random,
    profile: DifficultyProfile,
) -> Record:
    operational_state = sample_operational_state(core_path, rng)
    text_profile_state = sample_text_profile_state(core_path, operational_state, rng, profile)
    granularity_label_suites = GranularityLabelSuites(
        intent_label_coarse=build_granularity_label(core_path.intent_group),
        intent_label_normal=build_granularity_label(core_path.intent_group, core_path.issue_type),
        intent_label_fine=build_granularity_label(
            core_path.intent_group,
            core_path.issue_type,
            core_path.sub_intent,
        ),
        organizational_label_coarse=build_granularity_label(core_path.department),
        organizational_label_normal=build_granularity_label(
            core_path.department,
            operational_state.sub_department,
        ),
        organizational_label_fine=build_granularity_label(
            core_path.department,
            operational_state.sub_department,
            operational_state.owning_team,
        ),
    )
    return build_record_from_stages(
        core_path,
        operational_state,
        text_profile_state,
        granularity_label_suites,
    )


def sample_record_for_issue_target(
    department: str,
    ticket_category: str,
    issue_type: str,
    rng: random.Random,
    profile: DifficultyProfile,
) -> Record:
    core_path = sample_core_path_for_issue_target(
        department,
        ticket_category,
        issue_type,
        rng,
    )
    return sample_complete_record(
        core_path,
        rng,
        profile,
    )


def generate_records_for_issue_targets(
    issue_types: list[str],
    rng: random.Random | None = None,
    *,
    difficulty: str = "medium",
) -> list[Record]:
    prng = rng or random.Random()
    profile = difficulty_profile(difficulty)
    records: list[Record] = []
    for issue_type in issue_types:
        department, ticket_category = sample_issue_type_path(issue_type, prng)
        records.append(
            sample_record_for_issue_target(
                department,
                ticket_category,
                issue_type,
                prng,
                profile,
            )
        )
    return records


def refresh_department_branch(
    record: dict[str, str],
    rng: random.Random,
    department: str | None = None,
    profile: DifficultyProfile | None = None,
) -> None:
    active_profile = profile or difficulty_profile("medium")
    target_department = department or record["department"]
    refreshed_record = sample_complete_record(
        sample_core_path(
            target_department,
            rng,
            business_event_context=record["business_event_context"],
        ),
        rng,
        active_profile,
    )
    record.clear()
    record.update(refreshed_record)


def severity_rank(value: str) -> int:
    return SEVERITY_RANKS[value]


def urgency_rank(value: str) -> int:
    return URGENCY_RANKS[value]


def sample_record(
    schema: dict[str, Any],
    rng: random.Random | None = None,
    difficulty: str = "medium",
) -> Record:
    if schema != SCHEMA:
        raise ValueError("sample_record expects the fixed SCHEMA defined in this module.")
    profile = difficulty_profile(difficulty)
    prng = rng or random.Random()
    department = sample_department(prng)
    core_path = sample_core_path(department, prng)
    return sample_complete_record(
        core_path,
        prng,
        profile,
    )


def apply_business_rules(
    record: Record,
    rng: random.Random | None = None,
    difficulty: str = "medium",
) -> Record:
    profile = difficulty_profile(difficulty)
    prng = rng or random.Random()
    corrected = dict(record)
    corrected["issue_type"] = issue_type_for_sub_intent(corrected["sub_intent"])
    corrected["intent_group"] = intent_group_for_issue_type(corrected["issue_type"])

    event_context = corrected["business_event_context"]
    allowed_departments = BUSINESS_EVENT_CONTEXT_ALLOWED_DEPARTMENTS.get(event_context)
    if allowed_departments is not None and corrected["department"] not in allowed_departments:
        preferred_departments = BUSINESS_EVENT_CONTEXT_PREFERRED_DEPARTMENTS.get(event_context)
        if preferred_departments:
            refresh_department_branch(
                corrected,
                prng,
                department=weighted_choice(prng, preferred_departments),
                profile=profile,
            )
        else:
            corrected["business_event_context"] = "routine_operation"

    core_path = CorePath(
        department=corrected["department"],
        ticket_category=corrected["ticket_category"],
        issue_type=corrected["issue_type"],
        intent_group=corrected["intent_group"],
        sub_intent=corrected["sub_intent"],
        business_event_context=corrected["business_event_context"],
    )
    corrected["approval_required"] = ensure_approval_required(
        core_path,
        corrected["approval_required"],
        prng,
        workflow_stage=corrected["workflow_stage"],
        dependency_type=corrected["dependency_type"],
    )
    corrected["dependency_type"] = ensure_dependency_type(
        core_path,
        corrected["workflow_stage"],
        corrected["dependency_type"],
        prng,
    )
    corrected["approval_required"] = ensure_approval_required(
        core_path,
        corrected["approval_required"],
        prng,
        workflow_stage=corrected["workflow_stage"],
        dependency_type=corrected["dependency_type"],
    )
    corrected["urgency"] = ensure_urgency(
        core_path,
        corrected["workflow_stage"],
        corrected["urgency"],
        prng,
    )
    corrected["urgency"], corrected["severity"] = ensure_severity(
        core_path,
        corrected["urgency"],
        corrected["severity"],
        prng,
    )
    corrected["urgency"], corrected["impact_scope"] = ensure_impact_scope(
        corrected["urgency"],
        corrected["impact_scope"],
        prng,
    )
    corrected.update(derive_granularity_label_suites(corrected))

    return corrected


def validate_record(record: Record) -> None:
    validate_record_against_schema(SCHEMA, record)


def build_balanced_issue_plan(
    n: int,
    rng: random.Random,
    min_coverage_target_count: int | None = None,
) -> list[str]:
    return build_balanced_coverage_plan(
        reachable_issue_targets(),
        n,
        rng,
        min_count=min_coverage_target_count,
        coverage_label="issue_type",
    )


def reachable_paths_for_issue_type(issue_type: str) -> list[tuple[str, str]]:
    canonical_issue_type = canonicalize_issue_type(issue_type)
    paths: list[tuple[str, str]] = []
    for (department, ticket_category), sub_intents in SCHEMA["sub_intent"]["mapping"].items():
        if any(
            issue_type_for_sub_intent(sub_intent) == canonical_issue_type
            for sub_intent in sub_intents
        ):
            paths.append((department, ticket_category))
    return paths


def sample_issue_type_path(issue_type: str, rng: random.Random) -> tuple[str, str]:
    canonical_issue_type = canonicalize_issue_type(issue_type)
    paths = reachable_paths_for_issue_type(canonical_issue_type)
    if not paths:
        raise ValueError(f"No reachable path found for issue_type={canonical_issue_type}")
    weights = [
        DEPARTMENT_WEIGHTS[department] * CATEGORY_WEIGHTS[department][ticket_category]
        for department, ticket_category in paths
    ]
    return rng.choices(paths, weights=weights, k=1)[0]


def field_coverage_metrics(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    reachable = reachable_field_values(field)
    declared = declared_field_values(SCHEMA, field)
    counts = Counter(row[field] for row in rows)
    observed = [value for value in declared if counts.get(value, 0) > 0]
    unreachable = [value for value in declared if value not in reachable]
    missing_reachable = [value for value in reachable if counts.get(value, 0) == 0]
    reachable_counts = [counts.get(value, 0) for value in reachable]
    if reachable_counts:
        min_count = min(reachable_counts)
        max_count = max(reachable_counts)
        mean_count = sum(reachable_counts) / len(reachable_counts)
    else:
        min_count = 0
        max_count = 0
        mean_count = 0.0
    return {
        "declared": declared,
        "reachable": reachable,
        "observed": observed,
        "unreachable": unreachable,
        "missing_reachable": missing_reachable,
        "counts": counts,
        "declared_count": len(declared),
        "reachable_count": len(reachable),
        "observed_count": len(observed),
        "declared_coverage_ratio": format_ratio(len(observed), len(declared)),
        "reachable_coverage_ratio": format_ratio(
            len(reachable) - len(missing_reachable),
            len(reachable),
        ),
        "min_count": min_count,
        "max_count": max_count,
        "mean_count": mean_count,
    }


def build_dataset_summary(
    rows: list[dict[str, str]],
    *,
    mode: str,
    difficulty: str,
    schema_support_report: dict[str, list[str]],
    system_mention_probability: float | None = None,
    openrouter_model: str | None = None,
) -> str:
    access_related_rows = sum(1 for row in rows if row["issue_type"] in ACCESS_RELATED_ISSUE_TYPES)
    lines = [
        "Dataset summary",
        f"Rows: {len(rows)}",
        f"Mode: {mode}",
        f"Difficulty: {difficulty}",
        "Text generation backend: openrouter",
        f"OpenRouter model: {openrouter_model or 'n/a'}",
        (
            f"System mention probability: {system_mention_probability:.3f}"
            if system_mention_probability is not None
            else "System mention probability: profile_default"
        ),
        f"Access-related issue share: {format_ratio(access_related_rows, len(rows))}",
        "",
        format_schema_support_report(schema_support_report),
        "",
        "Field support:",
    ]

    for field in SCHEMA.keys():
        metrics = field_coverage_metrics(rows, field)
        lines.append(f"{field}:")
        lines.append(f"  declared_labels: {metrics['declared_count']}")
        lines.append(f"  reachable_labels: {metrics['reachable_count']}")
        lines.append(f"  observed_labels: {metrics['observed_count']}")
        lines.append(f"  declared_coverage_ratio: {metrics['declared_coverage_ratio']}")
        lines.append(f"  reachable_coverage_ratio: {metrics['reachable_coverage_ratio']}")
        lines.append(f"  reachable_count_min: {metrics['min_count']}")
        lines.append(f"  reachable_count_max: {metrics['max_count']}")
        lines.append(f"  reachable_count_mean: {metrics['mean_count']:.2f}")
        unreachable_values = ", ".join(metrics["unreachable"]) if metrics["unreachable"] else "none"
        missing_values = (
            ", ".join(metrics["missing_reachable"]) if metrics["missing_reachable"] else "none"
        )
        lines.append(f"  unreachable_declared_labels: {unreachable_values}")
        lines.append(f"  missing_reachable_labels: {missing_values}")
        lines.append("")

    lines.extend(
        [
            "Coverage control:",
            (
                "  balanced mode targets reachable issue_type coverage"
                if mode == "balanced"
                else "  random mode uses weighted sampling without explicit coverage guarantees"
            ),
            "",
            "Label counts:",
        ]
    )

    for field in SCHEMA.keys():
        counts = Counter(row[field] for row in rows)
        lines.append(f"{field}:")
        for value in declared_field_values(SCHEMA, field):
            lines.append(f"  {value}: {counts.get(value, 0)}")
        lines.append("")

    lines.append("Cross-tabs:")
    for left_field, right_field in MAIN_POV_CROSSTABS:
        lines.append(f"{left_field} x {right_field}:")
        cross_tab: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            cross_tab[row[left_field]][row[right_field]] += 1
        for left_value in declared_field_values(SCHEMA, left_field):
            present_pairs = [
                f"{right_value}={cross_tab[left_value].get(right_value, 0)}"
                for right_value in declared_field_values(SCHEMA, right_field)
                if cross_tab[left_value].get(right_value, 0) > 0
            ]
            total = sum(cross_tab[left_value].values())
            if present_pairs:
                joined = ", ".join(present_pairs)
                lines.append(f"  {left_value} (total={total}): {joined}")
            else:
                lines.append(f"  {left_value} (total=0):")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate_records(
    n: int,
    rng: random.Random | None = None,
    *,
    mode: str = "balanced",
    difficulty: str = "medium",
    min_coverage_target_count: int | None = None,
) -> list[Record]:
    if n <= 0:
        raise ValueError("n must be positive")
    if mode not in {"random", "balanced"}:
        raise ValueError(f"Unsupported generation mode: {mode}")

    prng = rng or random.Random()
    profile = difficulty_profile(difficulty)
    records: list[Record] = []
    if mode == "random":
        target_plan = [None] * n
    else:
        target_plan = build_balanced_issue_plan(n, prng, min_coverage_target_count)

    for target in target_plan:
        if target is None:
            record = apply_business_rules(
                sample_record(
                    SCHEMA,
                    prng,
                    difficulty=difficulty,
                ),
                prng,
                difficulty=difficulty,
            )
        else:
            issue_type = target
            department, ticket_category = sample_issue_type_path(issue_type, prng)
            record = apply_business_rules(
                sample_record_for_issue_target(
                    department,
                    ticket_category,
                    issue_type,
                    prng,
                    profile,
                ),
                prng,
                difficulty=difficulty,
            )
            if record["issue_type"] != issue_type:
                raise ValueError(
                    "Balanced generation drifted away from the planned issue_type: "
                    f"expected {issue_type}, got {record['issue_type']}"
                )
        validate_record(record)
        records.append(record)
    return records
