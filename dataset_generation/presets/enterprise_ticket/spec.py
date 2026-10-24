"""Static resources and typed profiles for the enterprise ticket preset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Required, TypeAlias, TypedDict

from . import spec_data


@dataclass(frozen=True)
class DifficultyProfile:
    name: str
    writing_style_adjustments: Mapping[str, float]
    clarity_adjustments: Mapping[str, float]
    opener_probability: float
    explicit_system_probability: float
    explicit_resolution_probability: float
    context_probability: float
    progress_probability: float
    pressure_probability: float
    indirect_issue_probability: float
    symptom_first_probability: float
    sentence_pattern_weights: Mapping[str, float]


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    temperature: float
    batch_size: int
    max_retries: int


@dataclass(frozen=True)
class TextVariationProfile:
    structural_mode: str
    knowledge_state: str
    intent_mode: str
    emotional_tone: str
    temporal_pov: str
    interaction_pov: str
    message_channel: str
    length_style: str
    completeness_mode: str
    messiness: str
    mention_system_name: bool
    include_greeting: bool
    include_context: bool
    mention_previous_attempts_or_follow_up: bool
    mention_time_reference: bool
    mention_work_consequence: bool
    mention_other_people_affected: bool
    include_explicit_resolution_request: bool
    prefer_indirect_issue_description: bool
    prefer_symptom_before_request: bool
    mention_uncertainty: bool
    mention_self_help_attempt: bool
    allow_incomplete_details: bool
    allow_fragment_style: bool
    sentence_pattern: str


class _EnterpriseTicketRecordFields(TypedDict, total=False):
    department: str
    sub_department: str
    requester_role: str
    owning_team: str
    ticket_category: str
    issue_type: str
    intent_group: str
    sub_intent: str
    intent_label_coarse: str
    intent_label_normal: str
    intent_label_fine: str
    organizational_label_coarse: str
    organizational_label_normal: str
    organizational_label_fine: str
    system_name: str
    business_event_context: str
    workflow_stage: str
    approval_required: str
    dependency_type: str
    urgency: str
    severity: str
    impact_scope: str
    writing_style: str
    clarity_level: str


class EnterpriseTicketRecordDraft(_EnterpriseTicketRecordFields):
    pass


class EnterpriseTicketRecord(_EnterpriseTicketRecordFields):
    department: Required[str]
    sub_department: Required[str]
    requester_role: Required[str]
    owning_team: Required[str]
    ticket_category: Required[str]
    issue_type: Required[str]
    intent_group: Required[str]
    sub_intent: Required[str]
    intent_label_coarse: Required[str]
    intent_label_normal: Required[str]
    intent_label_fine: Required[str]
    organizational_label_coarse: Required[str]
    organizational_label_normal: Required[str]
    organizational_label_fine: Required[str]
    system_name: Required[str]
    business_event_context: Required[str]
    workflow_stage: Required[str]
    approval_required: Required[str]
    dependency_type: Required[str]
    urgency: Required[str]
    severity: Required[str]
    impact_scope: Required[str]
    writing_style: Required[str]
    clarity_level: Required[str]


RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


def _load_text_resource(filename: str) -> str:
    return (RESOURCE_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


SCHEMA = spec_data.SCHEMA
RecordDraft: TypeAlias = dict[str, str]
Record: TypeAlias = dict[str, str]

DEFAULT_OUTPUT_PATH = "data/enterprise_ticket_dataset.csv"
CSV_COLUMNS = ["ticket_text", *SCHEMA.keys()]

SURFACE_METADATA_FIELDS = spec_data.SURFACE_METADATA_FIELDS
ISSUE_TYPE_ALIASES = spec_data.ISSUE_TYPE_ALIASES
GRANULARITY_LABEL_COMPONENTS = spec_data.GRANULARITY_LABEL_COMPONENTS
GRANULARITY_LABEL_FIELDS = spec_data.GRANULARITY_LABEL_FIELDS
build_granularity_label = spec_data.build_granularity_label

DEPARTMENT_WEIGHTS = spec_data.DEPARTMENT_WEIGHTS
CATEGORY_WEIGHTS = spec_data.CATEGORY_WEIGHTS
SUB_DEPARTMENT_WEIGHTS = spec_data.SUB_DEPARTMENT_WEIGHTS
REQUESTER_ROLE_BASE_WEIGHTS = spec_data.REQUESTER_ROLE_BASE_WEIGHTS
REQUESTER_ROLE_DEPARTMENT_ADJUSTMENTS = spec_data.REQUESTER_ROLE_DEPARTMENT_ADJUSTMENTS
REQUESTER_ROLE_CATEGORY_ADJUSTMENTS = spec_data.REQUESTER_ROLE_CATEGORY_ADJUSTMENTS
BUSINESS_EVENT_CONTEXT_BASE_WEIGHTS = spec_data.BUSINESS_EVENT_CONTEXT_BASE_WEIGHTS
BUSINESS_EVENT_CONTEXT_ALLOWED_DEPARTMENTS = spec_data.BUSINESS_EVENT_CONTEXT_ALLOWED_DEPARTMENTS
BUSINESS_EVENT_CONTEXT_PREFERRED_DEPARTMENTS = (
    spec_data.BUSINESS_EVENT_CONTEXT_PREFERRED_DEPARTMENTS
)
BUSINESS_EVENT_CONTEXT_DEPARTMENT_ADJUSTMENTS = (
    spec_data.BUSINESS_EVENT_CONTEXT_DEPARTMENT_ADJUSTMENTS
)
BUSINESS_EVENT_CONTEXT_CATEGORY_ADJUSTMENTS = spec_data.BUSINESS_EVENT_CONTEXT_CATEGORY_ADJUSTMENTS
BUSINESS_EVENT_CONTEXT_SUB_INTENT_ADJUSTMENTS = (
    spec_data.BUSINESS_EVENT_CONTEXT_SUB_INTENT_ADJUSTMENTS
)
TICKET_CATEGORY_EVENT_CONTEXT_ADJUSTMENTS = spec_data.TICKET_CATEGORY_EVENT_CONTEXT_ADJUSTMENTS
REQUESTER_ROLE_EVENT_CONTEXT_ADJUSTMENTS = spec_data.REQUESTER_ROLE_EVENT_CONTEXT_ADJUSTMENTS
URGENCY_EVENT_CONTEXT_ADJUSTMENTS = spec_data.URGENCY_EVENT_CONTEXT_ADJUSTMENTS
OWNING_TEAM_BASE_WEIGHTS = spec_data.OWNING_TEAM_BASE_WEIGHTS
OWNING_TEAM_CATEGORY_ADJUSTMENTS = spec_data.OWNING_TEAM_CATEGORY_ADJUSTMENTS
SYSTEM_NAME_BASE_ADJUSTMENTS = spec_data.SYSTEM_NAME_BASE_ADJUSTMENTS
SYSTEM_NAME_CATEGORY_ADJUSTMENTS = spec_data.SYSTEM_NAME_CATEGORY_ADJUSTMENTS
SYSTEM_NAME_SUB_INTENT_ADJUSTMENTS = spec_data.SYSTEM_NAME_SUB_INTENT_ADJUSTMENTS
APPROVAL_REQUIRED_BASE_WEIGHTS = spec_data.APPROVAL_REQUIRED_BASE_WEIGHTS
APPROVAL_REQUIRED_ACCESS_SUB_INTENT_ADJUSTMENTS = (
    spec_data.APPROVAL_REQUIRED_ACCESS_SUB_INTENT_ADJUSTMENTS
)
APPROVAL_REQUIRED_PRIVILEGED_ACCESS_SUB_INTENT_ADJUSTMENTS = (
    spec_data.APPROVAL_REQUIRED_PRIVILEGED_ACCESS_SUB_INTENT_ADJUSTMENTS
)
APPROVAL_REQUIRED_TICKET_CATEGORY_OVERRIDES = spec_data.APPROVAL_REQUIRED_TICKET_CATEGORY_OVERRIDES
APPROVAL_REQUIRED_TICKET_CATEGORY_ADJUSTMENTS = (
    spec_data.APPROVAL_REQUIRED_TICKET_CATEGORY_ADJUSTMENTS
)
APPROVAL_REQUIRED_DEPARTMENT_ADJUSTMENTS = spec_data.APPROVAL_REQUIRED_DEPARTMENT_ADJUSTMENTS
DEPENDENCY_TYPE_BASE_WEIGHTS = spec_data.DEPENDENCY_TYPE_BASE_WEIGHTS
DEPENDENCY_TYPE_APPROVAL_REQUIRED_ADJUSTMENTS = (
    spec_data.DEPENDENCY_TYPE_APPROVAL_REQUIRED_ADJUSTMENTS
)
DEPENDENCY_TYPE_VENDOR_SUB_INTENT_ADJUSTMENTS = (
    spec_data.DEPENDENCY_TYPE_VENDOR_SUB_INTENT_ADJUSTMENTS
)
DEPENDENCY_TYPE_DOCUMENT_SUB_INTENT_ADJUSTMENTS = (
    spec_data.DEPENDENCY_TYPE_DOCUMENT_SUB_INTENT_ADJUSTMENTS
)
DEPENDENCY_TYPE_SYSTEM_SYNC_SUB_INTENT_ADJUSTMENTS = (
    spec_data.DEPENDENCY_TYPE_SYSTEM_SYNC_SUB_INTENT_ADJUSTMENTS
)
DEPENDENCY_TYPE_WORKFLOW_STAGE_OVERRIDES = spec_data.DEPENDENCY_TYPE_WORKFLOW_STAGE_OVERRIDES
DEPENDENCY_TYPE_WORKFLOW_STAGE_ADJUSTMENTS = spec_data.DEPENDENCY_TYPE_WORKFLOW_STAGE_ADJUSTMENTS
SEVERITY_BASE_WEIGHTS_BY_URGENCY = spec_data.SEVERITY_BASE_WEIGHTS_BY_URGENCY
SEVERITY_SECURITY_OVERRIDES = spec_data.SEVERITY_SECURITY_OVERRIDES
SEVERITY_SECURITY_ADJUSTMENTS = spec_data.SEVERITY_SECURITY_ADJUSTMENTS
SEVERITY_TICKET_CATEGORY_ADJUSTMENTS = spec_data.SEVERITY_TICKET_CATEGORY_ADJUSTMENTS
IMPACT_SCOPE_BASE_WEIGHTS = spec_data.IMPACT_SCOPE_BASE_WEIGHTS
IMPACT_SCOPE_TICKET_CATEGORY_ADJUSTMENTS = spec_data.IMPACT_SCOPE_TICKET_CATEGORY_ADJUSTMENTS
IMPACT_SCOPE_SECURITY_ADJUSTMENTS = spec_data.IMPACT_SCOPE_SECURITY_ADJUSTMENTS
IMPACT_SCOPE_URGENCY_ADJUSTMENTS = spec_data.IMPACT_SCOPE_URGENCY_ADJUSTMENTS
WRITING_STYLE_BASE_WEIGHTS = spec_data.WRITING_STYLE_BASE_WEIGHTS
WRITING_STYLE_DEPARTMENT_ADJUSTMENTS = spec_data.WRITING_STYLE_DEPARTMENT_ADJUSTMENTS
WRITING_STYLE_HIGH_PRESSURE_ADJUSTMENTS = spec_data.WRITING_STYLE_HIGH_PRESSURE_ADJUSTMENTS
WRITING_STYLE_REQUESTER_ROLE_ADJUSTMENTS = spec_data.WRITING_STYLE_REQUESTER_ROLE_ADJUSTMENTS
CLARITY_LEVEL_BASE_WEIGHTS = spec_data.CLARITY_LEVEL_BASE_WEIGHTS
CLARITY_LEVEL_WRITING_STYLE_OVERRIDES = spec_data.CLARITY_LEVEL_WRITING_STYLE_OVERRIDES
CLARITY_LEVEL_WRITING_STYLE_ADJUSTMENTS = spec_data.CLARITY_LEVEL_WRITING_STYLE_ADJUSTMENTS
CLARITY_LEVEL_REQUESTER_ROLE_ADJUSTMENTS = spec_data.CLARITY_LEVEL_REQUESTER_ROLE_ADJUSTMENTS

ACCESS_ISSUES = frozenset(spec_data.ACCESS_ISSUES)
PRIVILEGED_ACCESS_ISSUES = frozenset(spec_data.PRIVILEGED_ACCESS_ISSUES)
SECURITY_ISSUES = frozenset(spec_data.SECURITY_ISSUES)
CRITICAL_SECURITY_ISSUES = frozenset(spec_data.CRITICAL_SECURITY_ISSUES)
PURCHASE_ISSUES = frozenset(spec_data.PURCHASE_ISSUES)
VENDOR_ISSUES = frozenset(spec_data.VENDOR_ISSUES)
DOCUMENT_ISSUES = frozenset(spec_data.DOCUMENT_ISSUES)
SYSTEM_SYNC_ISSUES = frozenset(spec_data.SYSTEM_SYNC_ISSUES)

MAIN_POV_CROSSTABS = spec_data.MAIN_POV_CROSSTABS


def _freeze_mapping(mapping: dict[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(mapping))


DIFFICULTY_PROFILES = {
    name: DifficultyProfile(
        name=config["name"],
        writing_style_adjustments=_freeze_mapping(config["writing_style_adjustments"]),
        clarity_adjustments=_freeze_mapping(config["clarity_adjustments"]),
        opener_probability=config["opener_probability"],
        explicit_system_probability=config["explicit_system_probability"],
        explicit_resolution_probability=config["explicit_resolution_probability"],
        context_probability=config["context_probability"],
        progress_probability=config["progress_probability"],
        pressure_probability=config["pressure_probability"],
        indirect_issue_probability=config["indirect_issue_probability"],
        symptom_first_probability=config["symptom_first_probability"],
        sentence_pattern_weights=_freeze_mapping(config["sentence_pattern_weights"]),
    )
    for name, config in spec_data.DIFFICULTY_PROFILES.items()
}

OPENROUTER_SYSTEM_PROMPT = _load_text_resource("enterprise_ticket_system_prompt.txt")
OPENROUTER_EVALUATION_PROMPT = _load_text_resource("enterprise_ticket_evaluation_prompt.txt")
BANNED_INTERNAL_PROCESS_PHRASES = spec_data.BANNED_INTERNAL_PROCESS_PHRASES
