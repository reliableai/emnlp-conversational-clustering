"""Text-label alignment helpers for the enterprise ticket preset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import spec_data

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_RAW_ALIGNMENT = json.loads((RESOURCE_DIR / "alignment.json").read_text(encoding="utf-8"))

ACCESS_RELATED_ISSUE_TYPES = frozenset(_RAW_ALIGNMENT["ACCESS_RELATED_ISSUE_TYPES"])
_GENERIC_LABEL_TOKENS = frozenset(_RAW_ALIGNMENT["GENERIC_LABEL_TOKENS"])
_MATCH_STOPWORDS = frozenset(_RAW_ALIGNMENT["MATCH_STOPWORDS"])
_DISPLAY_TOKENS = _RAW_ALIGNMENT["DISPLAY_TOKENS"]
_TOKEN_ALIASES = _RAW_ALIGNMENT["TOKEN_ALIASES"]

_ISSUE_TYPE_HINTS = _RAW_ALIGNMENT["ISSUE_TYPE_HINTS"]
_SUB_INTENT_HINTS = _RAW_ALIGNMENT["SUB_INTENT_HINTS"]
_ISSUE_TYPE_COMPATIBILITY_LABELS: dict[str, tuple[str, ...]] = {}
for legacy_issue_type, canonical_issue_type in spec_data.ISSUE_TYPE_ALIASES.items():
    _ISSUE_TYPE_COMPATIBILITY_LABELS.setdefault(canonical_issue_type, ())
    _ISSUE_TYPE_COMPATIBILITY_LABELS[canonical_issue_type] = (
        *_ISSUE_TYPE_COMPATIBILITY_LABELS[canonical_issue_type],
        legacy_issue_type,
    )


@dataclass(frozen=True)
class AlignmentAssessment:
    issue_type_match: bool
    sub_intent_match: bool
    issue_type_terms: tuple[str, ...]
    sub_intent_terms: tuple[str, ...]
    context_terms: tuple[str, ...]


def normalize_ticket_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _normalize_term(term: str) -> str:
    return normalize_ticket_text(term)


def humanize_label(label: str) -> str:
    parts = []
    for token in label.split("_"):
        parts.append(_DISPLAY_TOKENS.get(token, token))
    return " ".join(parts)


def canonicalize_issue_type(issue_type: str) -> str:
    return spec_data.ISSUE_TYPE_ALIASES.get(issue_type, issue_type)


def issue_type_compatibility_labels(issue_type: str) -> tuple[str, ...]:
    canonical_issue_type = canonicalize_issue_type(issue_type)
    return (canonical_issue_type, *_ISSUE_TYPE_COMPATIBILITY_LABELS.get(canonical_issue_type, ()))


def preferred_sub_intent_phrase(sub_intent: str) -> str:
    hints = _SUB_INTENT_HINTS.get(sub_intent, ())
    for hint in hints:
        normalized_hint = normalize_ticket_text(hint)
        if " " in normalized_hint and len(normalized_hint.split()) <= 4:
            return hint
    if hints:
        return hints[0]
    return humanize_label(sub_intent)


def _label_terms(label: str) -> tuple[str, ...]:
    normalized = label.replace("_", " ")
    pieces = [piece for piece in normalized.split() if piece not in _GENERIC_LABEL_TOKENS]
    terms: list[str] = [normalized]
    for piece in pieces:
        terms.extend(_TOKEN_ALIASES.get(piece, (piece,)))
    return tuple(dict.fromkeys(_normalize_term(term) for term in terms if term))


def build_alignment_terms(issue_type: str, sub_intent: str) -> tuple[str, ...]:
    terms = list(_SUB_INTENT_HINTS.get(sub_intent, ()))
    for issue_type_label in issue_type_compatibility_labels(issue_type):
        terms.extend(_ISSUE_TYPE_HINTS.get(issue_type_label, ()))
        terms.extend(_label_terms(issue_type_label))
    terms.extend(_label_terms(sub_intent))
    return tuple(dict.fromkeys(_normalize_term(term) for term in terms if term))


def build_issue_type_terms(issue_type: str) -> tuple[str, ...]:
    terms: list[str] = []
    for issue_type_label in issue_type_compatibility_labels(issue_type):
        terms.extend(_ISSUE_TYPE_HINTS.get(issue_type_label, ()))
        terms.extend(_label_terms(issue_type_label))
    return tuple(dict.fromkeys(_normalize_term(term) for term in terms if term))


def build_sub_intent_terms(sub_intent: str) -> tuple[str, ...]:
    terms = list(_SUB_INTENT_HINTS.get(sub_intent, ()))
    terms.extend(_label_terms(sub_intent))
    return tuple(dict.fromkeys(_normalize_term(term) for term in terms if term))


def build_context_terms(record: dict[str, str]) -> tuple[str, ...]:
    terms: list[str] = []
    for field in ("ticket_category", "system_name", "department", "sub_department"):
        value = record.get(field)
        if value:
            terms.extend(_label_terms(value))
    return tuple(dict.fromkeys(_normalize_term(term) for term in terms if term))


def _token_match_variants(token: str) -> tuple[str, ...]:
    variants = {token}

    if len(token) > 4 and token.endswith("ies"):
        variants.add(token[:-3] + "y")
    if len(token) > 4 and token.endswith("ied"):
        variants.add(token[:-3] + "y")
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        variants.add(base)
        if len(base) > 2 and base[-1] == base[-2]:
            variants.add(base[:-1])
        if not base.endswith("e"):
            variants.add(base + "e")
    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]
        variants.add(base)
        if len(base) > 2 and base[-1] == base[-2]:
            variants.add(base[:-1])
        if not base.endswith("e"):
            variants.add(base + "e")
    if len(token) > 4 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        variants.add(token[:-1])

    return tuple(variants)


def _tokens_match(term_token: str, text_token: str) -> bool:
    return bool(set(_token_match_variants(term_token)) & set(_token_match_variants(text_token)))


def _term_matches(normalized_text: str, term: str) -> bool:
    text_tokens = normalized_text.split()
    term_tokens = term.split()
    if not term_tokens:
        return False
    if len(term_tokens) == 1:
        return any(_tokens_match(term_tokens[0], text_token) for text_token in text_tokens)
    for start_index in range(len(text_tokens) - len(term_tokens) + 1):
        if all(
            _tokens_match(term_token, text_token)
            for term_token, text_token in zip(
                term_tokens,
                text_tokens[start_index : start_index + len(term_tokens)],
                strict=True,
            )
        ):
            return True
    significant_tokens = [token for token in term_tokens if token not in _MATCH_STOPWORDS]
    if len(significant_tokens) > 1 and all(
        any(_tokens_match(token, text_token) for text_token in text_tokens)
        for token in significant_tokens
    ):
        return True

    return False


def _matched_terms(normalized_text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in terms if _term_matches(normalized_text, term))


def assess_ticket_text_alignment(
    record: dict[str, str],
    ticket_text: str,
) -> AlignmentAssessment:
    normalized_text = normalize_ticket_text(ticket_text)
    if not normalized_text:
        return AlignmentAssessment(False, False, (), (), ())

    issue_type_terms = build_issue_type_terms(record["issue_type"])
    sub_intent_terms = build_sub_intent_terms(record["sub_intent"])
    context_terms = build_context_terms(record)

    matched_issue_type_terms = _matched_terms(normalized_text, issue_type_terms)
    matched_sub_intent_terms = _matched_terms(normalized_text, sub_intent_terms)
    matched_context_terms = _matched_terms(normalized_text, context_terms)

    issue_type_match = bool(matched_issue_type_terms or matched_sub_intent_terms)
    sub_intent_match = bool(matched_sub_intent_terms)
    return AlignmentAssessment(
        issue_type_match=issue_type_match,
        sub_intent_match=sub_intent_match,
        issue_type_terms=matched_issue_type_terms,
        sub_intent_terms=matched_sub_intent_terms,
        context_terms=matched_context_terms,
    )


def ticket_text_matches_labels(issue_type: str, sub_intent: str, ticket_text: str) -> bool:
    normalized_text = normalize_ticket_text(ticket_text)
    if not normalized_text:
        return False
    return any(
        _term_matches(normalized_text, term)
        for term in build_alignment_terms(issue_type, sub_intent)
    )
