"""Unit tests for shared clustering POV-bank helpers used across backends."""

from dataclasses import dataclass
from typing import Any, cast

from conversational_clustering.clustering_agent.backends.pov_bank import (
    append_clustering_povs_to_bank,
    build_clustering_pov_bank_message,
    coerce_clustering_pov_bank,
    coerce_clustering_pov_history,
    make_append_clustering_povs_tool,
    upsert_clustering_pov_history,
)


@dataclass
class FakePovBankState:
    clustering_pov_bank: list[str] | None = None


def test_coerce_clustering_pov_bank_normalizes_and_deduplicates() -> None:
    assert coerce_clustering_pov_bank([" Genre ", "age   rating", "genre", "", "director"]) == [
        "Genre",
        "age rating",
        "director",
    ]


def test_append_clustering_povs_to_bank_returns_merged_bank() -> None:
    result = append_clustering_povs_to_bank(
        existing_pov_bank=["Genre", "age rating"],
        povs=["genre", "release decade", "Age   Rating", "director"],
    )

    assert result.added_povs == ["release decade", "director"]
    assert result.num_added == 2
    assert result.total_povs == 4
    assert result.merged_pov_bank == [
        "Genre",
        "age rating",
        "release decade",
        "director",
    ]


def test_make_append_clustering_povs_tool_updates_generic_runtime_state() -> None:
    runtime_state = FakePovBankState(clustering_pov_bank=["Genre"])
    tool = make_append_clustering_povs_tool(runtime_state=runtime_state)

    response = cast(Any, tool).invoke({"povs": ["genre", "release decade"]})

    assert response == ('{"added_povs": ["release decade"], "num_added": 1, "total_povs": 2}')
    assert runtime_state.clustering_pov_bank == ["Genre", "release decade"]


def test_coerce_clustering_pov_history_normalizes_entries_and_overwrites_by_turn() -> None:
    history = coerce_clustering_pov_history(
        [
            {"turn_index": 1, "povs": [" Genre ", "age   rating"]},
            {"turn_index": True, "povs": ["invalid"]},
            {"turn_index": 2, "povs": ["   ", "\n"]},
            {"turn_index": 1, "povs": ["genre", "director"]},
            {"turn_index": 3, "povs": ["audience", "Audience", "tone"]},
        ]
    )

    assert history == [
        {"turn_index": 1, "povs": ["genre", "director"]},
        {"turn_index": 3, "povs": ["audience", "tone"]},
    ]


def test_upsert_clustering_pov_history_appends_first_snapshot() -> None:
    history = upsert_clustering_pov_history(history=None, turn_index=1, povs=["genre", "tone"])

    assert history == [{"turn_index": 1, "povs": ["genre", "tone"]}]


def test_upsert_clustering_pov_history_overwrites_existing_turn_and_preserves_order() -> None:
    history = upsert_clustering_pov_history(
        history=[
            {"turn_index": 1, "povs": ["genre"]},
            {"turn_index": 2, "povs": ["genre", "tone"]},
        ],
        turn_index=2,
        povs=["genre", "tone", "audience"],
    )

    assert history == [
        {"turn_index": 1, "povs": ["genre"]},
        {"turn_index": 2, "povs": ["genre", "tone", "audience"]},
    ]


def test_upsert_clustering_pov_history_appends_later_turn() -> None:
    history = upsert_clustering_pov_history(
        history=[{"turn_index": 1, "povs": ["genre"]}],
        turn_index=3,
        povs=["genre", "tone"],
    )

    assert history == [
        {"turn_index": 1, "povs": ["genre"]},
        {"turn_index": 3, "povs": ["genre", "tone"]},
    ]


def test_build_clustering_pov_bank_message_formats_numbered_bank() -> None:
    message = build_clustering_pov_bank_message(["genre", "age rating", "director"])

    assert cast(str, message.content) == (
        "Internal clustering POV bank so far "
        "(dataset-grounded possibilities, not necessarily user-desired):\n"
        "1. genre\n"
        "2. age rating\n"
        "3. director"
    )
