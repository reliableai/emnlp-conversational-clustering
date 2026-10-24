"""Unit tests for human-agent sanitization and verifier-driven retry-loop behavior."""

from typing import cast
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

import conversational_clustering.human_agent as human_agent
from conversational_clustering.human_agent import (
    _filter_user_visible_messages,
    _sanitize_ai_message,
    build_human_app,
)
from conversational_clustering.human_turn_injection import HumanTurnInjectionConfig
from conversational_clustering.topic_discovery import TopicDiscoveryConfig


def test_sanitize_drops_tool_call_only_message_without_content() -> None:
    message = AIMessage(content="", tool_calls=[{"name": "read_dataset", "args": {}, "id": "1"}])

    sanitized = _sanitize_ai_message(message)

    assert sanitized is None


def test_sanitize_keeps_content_when_tool_calls_present() -> None:
    message = AIMessage(
        content="Here is my plan.",
        tool_calls=[{"name": "read_dataset", "args": {}, "id": "1"}],
    )

    sanitized = _sanitize_ai_message(message)

    assert sanitized is not None
    assert sanitized.content == "Here is my plan."
    assert getattr(sanitized, "tool_calls", []) == []


def test_sanitize_removes_provider_specific_kwargs() -> None:
    message = AIMessage(
        content="Hello",
        additional_kwargs={
            "function_call": {"name": "f"},
            "tool_calls": [{"id": "x"}],
            "reasoning_content": "internal reasoning",
            "reasoning_details": [{"type": "reasoning.summary", "summary": "step"}],
            "preserved": 123,
        },
    )

    sanitized = _sanitize_ai_message(message)

    assert sanitized is not None
    assert sanitized.content == "Hello"
    assert sanitized.additional_kwargs == {"preserved": 123}


def test_filter_strips_reasoning_metadata_from_ai_messages() -> None:
    messages = [
        HumanMessage(content="user"),
        AIMessage(
            content="assistant reply",
            additional_kwargs={
                "reasoning_content": "hidden",
                "reasoning_details": [{"type": "reasoning.summary", "summary": "step"}],
                "keep": "x",
            },
        ),
    ]

    visible = _filter_user_visible_messages(messages)

    assert len(visible) == 2
    assert isinstance(visible[1], AIMessage)
    assert visible[1].additional_kwargs == {"keep": "x"}


def test_sanitize_returns_original_message_when_no_cleanup_needed() -> None:
    message = AIMessage(content="hello")

    sanitized = _sanitize_ai_message(message)

    assert sanitized is message


def test_filter_keeps_human_and_sanitized_ai_messages() -> None:
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="user says hi"),
        AIMessage(content="", tool_calls=[{"name": "tool", "args": {}, "id": "1"}]),
        AIMessage(
            content="assistant",
            additional_kwargs={"function_call": {"name": "f"}, "keep": True},
        ),
    ]

    visible = _filter_user_visible_messages(messages)

    assert len(visible) == 2
    assert isinstance(visible[0], HumanMessage)
    assert isinstance(visible[1], AIMessage)
    assert visible[1].content == "assistant"
    assert visible[1].additional_kwargs == {"keep": True}


def test_human_turn_draft_strict_schema_excludes_topic_id() -> None:
    schema = convert_to_openai_tool(human_agent.HumanTurnDraft, strict=True)
    topic_item_parameters = schema["function"]["parameters"]["properties"]["topics"]["items"]

    assert topic_item_parameters["required"] == ["label", "guidance", "rationale"]
    assert "topic_id" not in topic_item_parameters["properties"]


class FakeLLM:
    def __init__(self, responses: list[AIMessage | dict[str, object]]) -> None:
        self._responses = list(responses)
        self.invocations: list[list] = []
        self._structured_schema: type | None = None

    def with_structured_output(
        self,
        schema: object,
        *,
        strict: bool = False,
        include_raw: bool = False,
    ) -> "FakeLLM":
        _ = strict, include_raw
        self._structured_schema = schema if isinstance(schema, type) else None
        return self

    def invoke(self, messages: list) -> object:
        self.invocations.append(messages)
        if not self._responses:
            raise AssertionError("No fake responses remaining.")
        response = self._responses.pop(0)
        if self._structured_schema is not None and isinstance(response, dict):
            parsed = response.get("parsed")
            if isinstance(parsed, dict) and all(isinstance(key, str) for key in parsed):
                parsed_kwargs = cast(dict[str, object], parsed)
                return {**response, "parsed": self._structured_schema(**parsed_kwargs)}
        return response


def _human_draft_response(
    *,
    message: str,
    thought: str,
    input_tokens: int,
    output_tokens: int,
    action: str = "message",
    topics: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "parsed": {
            "action": action,
            "message": message,
            "thought": thought,
            "topics": topics or [],
        },
        "raw": AIMessage(
            content='{"message":"draft","thought":"note"}',
            additional_kwargs={
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            },
        ),
    }


def _theme_switch_turn_injection(
    turn_index: int,
    *,
    source_theme_id: str = "department",
    target_theme_id: str = "ticket_category",
) -> HumanTurnInjectionConfig:
    return HumanTurnInjectionConfig(
        turn_index=turn_index,
        edit_primitive="theme_switch",
        source_theme_id=source_theme_id,
        target_theme_id=target_theme_id,
    )


def _granularity_turn_injection(
    turn_index: int,
    *,
    edit_primitive: str,
) -> HumanTurnInjectionConfig:
    return HumanTurnInjectionConfig(
        turn_index=turn_index,
        edit_primitive=edit_primitive,
    )


def test_build_human_app_retries_until_verifier_allows() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Here is a dense recap with too much detail.",
                thought="I am trying to summarize everything before asking for details.",
                input_tokens=1,
                output_tokens=2,
            ),
            _human_draft_response(
                message="Could you share a quick summary by topic?",
                thought="I want one concise next step that keeps the conversation natural.",
                input_tokens=3,
                output_tokens=4,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "retry", "feedback": "Keep it to one natural sentence."},
                "raw": AIMessage(
                    content='{"decision":"retry","feedback":"Keep it to one natural sentence."}',
                    additional_kwargs={"token_usage": {"input_tokens": 5, "output_tokens": 6}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 7, "output_tokens": 8}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance="Focus on urgency themes",
            system_prompt="Guidance: {guidance}",
            max_verification_retries=2,
        )

    state = {
        "messages": [AIMessage(content="What would you like to cluster?")],
        "dataset_metadata": {"size": 1},
        "iteration_count": 0,
    }
    result = app.invoke(state)
    final_message = result["messages"][-1]

    assert isinstance(final_message, HumanMessage)
    assert final_message.content == "Could you share a quick summary by topic?"
    assert final_message.additional_kwargs["token_usage"] == {
        "input_tokens": 16,
        "output_tokens": 20,
        "total_tokens": 36,
    }
    assert (
        final_message.additional_kwargs["human_thought"]
        == "I want one concise next step that keeps the conversation natural."
    )
    assert len(human_llm.invocations) == 2
    assert len(verifier_llm.invocations) == 2
    assert any(
        isinstance(message, SystemMessage)
        and "Keep it to one natural sentence." in str(message.content)
        for message in human_llm.invocations[1]
    )
    first_invocation_last_message = human_llm.invocations[0][-1]
    assert isinstance(first_invocation_last_message, HumanMessage)
    assert "Write your next single user message" in str(first_invocation_last_message.content)


def test_build_human_app_forces_allow_when_retry_limit_reached() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Please cluster by account risk.",
                thought="I want to direct the clustering outcome toward risk themes.",
                input_tokens=2,
                output_tokens=3,
            )
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "retry", "feedback": "Ask for dataset context first."},
                "raw": AIMessage(
                    content='{"decision":"retry","feedback":"Ask for dataset context first."}',
                    additional_kwargs={"token_usage": {"input_tokens": 4, "output_tokens": 5}},
                ),
            }
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance="Focus on risk-related clusters",
            system_prompt="Guidance: {guidance}",
            max_verification_retries=0,
        )

    state = {
        "messages": [AIMessage(content="What would you like to cluster?")],
        "dataset_metadata": {"size": 1},
        "iteration_count": 0,
    }
    result = app.invoke(state)
    final_message = result["messages"][-1]

    assert isinstance(final_message, HumanMessage)
    assert final_message.content == "Please cluster by account risk."
    assert final_message.additional_kwargs["token_usage"] == {
        "input_tokens": 6,
        "output_tokens": 8,
        "total_tokens": 14,
    }
    assert (
        final_message.additional_kwargs["human_thought"]
        == "I want to direct the clustering outcome toward risk themes."
    )
    assert len(human_llm.invocations) == 1
    assert len(verifier_llm.invocations) == 1
    assert isinstance(human_llm.invocations[0][-1], HumanMessage)


def test_build_human_app_logs_draft_verifier_and_allow_events() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Could you share a quick summary by topic?",
                thought="I need a concise prompt that elicits summary-level information.",
                input_tokens=1,
                output_tokens=2,
            )
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 3, "output_tokens": 4}},
                ),
            }
        ]
    )
    logger = mock.Mock()

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
        mock.patch.object(human_agent, "log_json") as log_json_mock,
    ):
        app = build_human_app(
            guidance="Focus on urgency themes",
            system_prompt="Guidance: {guidance}",
            logger=logger,
        )
        app.invoke(
            {
                "messages": [AIMessage(content="What would you like to cluster?")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 0,
            }
        )

    events = [call.args[1] for call in log_json_mock.call_args_list]
    assert events == [
        "human_agent_draft",
        "human_agent_verifier_decision",
        "human_agent_message_allowed",
    ]


def test_build_human_app_logs_turn_injection_when_applied() -> None:
    turn_injection = _theme_switch_turn_injection(2)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Please regroup these more broadly.",
                thought="The injected instruction should now steer the request.",
                input_tokens=1,
                output_tokens=2,
            )
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 3, "output_tokens": 4}},
                ),
            }
        ]
    )
    logger = mock.Mock()

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
        mock.patch.object(human_agent, "log_json") as log_json_mock,
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            logger=logger,
            turn_injection=turn_injection,
        )
        app.invoke(
            {
                "messages": [AIMessage(content="What would you like to cluster?")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 1,
            }
        )

    injection_call = log_json_mock.call_args_list[0]
    assert injection_call.args[1] == "human_turn_injection_applied"
    assert injection_call.args[2] == {
        "turn_index": 2,
        "configured_turn_index": 2,
        "edit_primitive": "theme_switch",
        "resolved_prompt_text": resolved_prompt_text,
        "retry_count": 0,
    }


def test_build_human_app_raises_when_verifier_structured_output_is_invalid() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Could you share a quick summary by topic?",
                thought="I want to keep this lightweight.",
                input_tokens=1,
                output_tokens=2,
            )
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "retry", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"retry","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 3, "output_tokens": 4}},
                ),
            }
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance="Focus on urgency themes",
            system_prompt="Guidance: {guidance}",
        )

    with mock.patch.object(human_agent, "log_json"):
        with pytest.raises(ValueError, match="requires non-empty feedback"):
            app.invoke(
                {
                    "messages": [AIMessage(content="What would you like to cluster?")],
                    "dataset_metadata": {"size": 1},
                    "iteration_count": 0,
                }
            )


def test_build_human_app_retries_after_missing_structured_human_payload() -> None:
    human_llm = FakeLLM(
        responses=[
            {
                "parsed": None,
                "raw": AIMessage(
                    content="",
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 0}},
                ),
            },
            _human_draft_response(
                message="Could you summarize the dominant themes first?",
                thought="Recover from the malformed empty response with a valid next turn.",
                input_tokens=2,
                output_tokens=3,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 4, "output_tokens": 5}},
                ),
            }
        ]
    )
    logger = mock.Mock()

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
        mock.patch.object(human_agent, "log_json") as log_json_mock,
    ):
        app = build_human_app(
            guidance="Focus on urgency themes",
            system_prompt="Guidance: {guidance}",
            max_verification_retries=2,
            logger=logger,
        )
        result = app.invoke(
            {
                "messages": [AIMessage(content="What would you like to cluster?")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 0,
            }
        )

    final_message = result["messages"][-1]
    assert isinstance(final_message, HumanMessage)
    assert final_message.content == "Could you summarize the dominant themes first?"
    assert final_message.additional_kwargs["token_usage"] == {
        "input_tokens": 7,
        "output_tokens": 8,
        "total_tokens": 15,
    }
    assert len(human_llm.invocations) == 2
    assert any(
        call.args[1] == "human_agent_malformed_structured_output"
        for call in log_json_mock.call_args_list
    )
    assert any(
        isinstance(message, SystemMessage)
        and "required structured output format" in str(message.content)
        for message in human_llm.invocations[1]
    )


def test_build_human_app_retries_node_after_structured_human_payload_budget_exhausted() -> None:
    human_llm = FakeLLM(
        responses=[
            {
                "parsed": None,
                "raw": AIMessage(
                    content="",
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 0}},
                ),
            },
            {
                "parsed": None,
                "raw": AIMessage(
                    content="",
                    additional_kwargs={"token_usage": {"input_tokens": 2, "output_tokens": 0}},
                ),
            },
            _human_draft_response(
                message="Could you summarize the dominant themes first?",
                thought="Recover from the malformed response after node-level retry.",
                input_tokens=3,
                output_tokens=4,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 5, "output_tokens": 6}},
                ),
            }
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
        mock.patch.object(human_agent, "log_json"),
    ):
        app = build_human_app(
            guidance="Focus on urgency themes",
            system_prompt="Guidance: {guidance}",
            max_verification_retries=1,
        )
        result = app.invoke(
            {
                "messages": [AIMessage(content="What would you like to cluster?")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 0,
            }
        )

    final_message = result["messages"][-1]
    assert isinstance(final_message, HumanMessage)
    assert final_message.content == "Could you summarize the dominant themes first?"
    assert len(human_llm.invocations) == 3
    assert len(verifier_llm.invocations) == 1
    assert any(
        isinstance(message, SystemMessage)
        and "required structured output format" in str(message.content)
        for message in human_llm.invocations[2]
    )


def test_build_human_app_allows_empty_candidate_when_verifier_allows() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="   ",
                thought="Need a minimal placeholder response.",
                input_tokens=1,
                output_tokens=1,
            )
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 2, "output_tokens": 2}},
                ),
            }
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance="Focus on urgency themes",
            system_prompt="Guidance: {guidance}",
        )

    with mock.patch.object(human_agent, "log_json"):
        result = app.invoke(
            {
                "messages": [AIMessage(content="What would you like to cluster?")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 0,
            }
        )

    final_message = result["messages"][-1]
    assert isinstance(final_message, HumanMessage)
    assert final_message.content == ""
    assert (
        final_message.additional_kwargs["human_thought"] == "Need a minimal placeholder response."
    )
    assert len(verifier_llm.invocations) == 1


def test_build_human_app_uses_guidance_free_system_prompt_without_substitution() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Could you show me a few representative samples first?",
                thought="I want to understand the dataset before choosing a clustering basis.",
                input_tokens=1,
                output_tokens=2,
            )
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 3, "output_tokens": 4}},
                ),
            }
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 0,
        }
    )

    final_message = result["messages"][-1]
    assert isinstance(final_message, HumanMessage)
    assert final_message.content == "Could you show me a few representative samples first?"


def test_build_human_app_activates_turn_injection_at_configured_turn() -> None:
    turn_injection = _theme_switch_turn_injection(2)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="First turn question.",
                thought="Stay on the default path.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Second turn question.",
                thought="Shift direction after the injection activates.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Third turn question.",
                thought="Continue naturally without replaying the injection.",
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            turn_injection=turn_injection,
        )

    app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 0,
        }
    )
    app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster next?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 1,
        }
    )
    app.invoke(
        {
            "messages": [AIMessage(content="What next?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
        }
    )

    first_system_messages = [
        str(message.content)
        for message in human_llm.invocations[0]
        if isinstance(message, SystemMessage)
    ]
    second_invocation = human_llm.invocations[1]
    second_system_messages = [
        str(message.content) for message in second_invocation if isinstance(message, SystemMessage)
    ]
    third_invocation = human_llm.invocations[2]
    third_system_messages = [
        str(message.content) for message in third_invocation if isinstance(message, SystemMessage)
    ]

    assert first_system_messages == ["Base system prompt."]
    assert second_system_messages[0] == "Base system prompt."
    assert "A new instruction takes effect starting this turn" in second_system_messages[1]
    assert "overrides any earlier clustering guidance or direction" in second_system_messages[1]
    assert f"Injected instruction:\n{resolved_prompt_text}" in second_system_messages[1]
    assert third_system_messages[0] == "Base system prompt."
    assert (
        "An injected instruction changed your clustering direction on an earlier turn."
        in (third_system_messages[1])
    )
    assert (
        "Keep following that injected direction on this turn and later turns"
        in (third_system_messages[1])
    )
    assert f"Original injected instruction:\n{resolved_prompt_text}" in third_system_messages[1]
    assert isinstance(second_invocation[0], SystemMessage)
    assert str(second_invocation[0].content) == "Base system prompt."
    assert isinstance(second_invocation[1], AIMessage)
    assert str(second_invocation[1].content) == "What would you like to cluster next?"
    assert isinstance(second_invocation[2], SystemMessage)
    assert "A new instruction takes effect starting this turn" in str(second_invocation[2].content)
    assert "overrides any earlier clustering guidance or direction" in str(
        second_invocation[2].content
    )
    assert f"Injected instruction:\n{resolved_prompt_text}" in str(second_invocation[2].content)
    assert isinstance(third_invocation[0], SystemMessage)
    assert str(third_invocation[0].content) == "Base system prompt."
    assert isinstance(third_invocation[1], AIMessage)
    assert str(third_invocation[1].content) == "What next?"
    assert isinstance(third_invocation[2], SystemMessage)
    assert "An injected instruction changed your clustering direction on an earlier turn." in str(
        third_invocation[2].content
    )
    assert "Keep following that injected direction on this turn and later turns" in str(
        third_invocation[2].content
    )
    assert f"Original injected instruction:\n{resolved_prompt_text}" in str(
        third_invocation[2].content
    )


def test_build_human_app_keeps_turn_injection_active_during_retry() -> None:
    turn_injection = _theme_switch_turn_injection(2)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Too vague.",
                thought="First draft after the injected instruction.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Please merge these into broader buckets.",
                thought="Retry while keeping the injected direction.",
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "retry", "feedback": "Make the message more explicit."},
                "raw": AIMessage(
                    content='{"decision":"retry","feedback":"Make the message more explicit."}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            turn_injection=turn_injection,
            max_verification_retries=2,
        )

    app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster next?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 1,
        }
    )

    first_invocation = human_llm.invocations[0]
    retry_invocation = human_llm.invocations[1]
    first_system_messages = [
        str(message.content) for message in first_invocation if isinstance(message, SystemMessage)
    ]
    retry_system_messages = [
        str(message.content) for message in retry_invocation if isinstance(message, SystemMessage)
    ]

    assert first_system_messages[0] == "Base system prompt."
    assert "A new instruction takes effect starting this turn" in first_system_messages[1]
    assert "overrides any earlier clustering guidance or direction" in first_system_messages[1]
    assert f"Injected instruction:\n{resolved_prompt_text}" in first_system_messages[1]
    assert retry_system_messages[0] == "Base system prompt."
    assert "A new instruction takes effect starting this turn" in retry_system_messages[1]
    assert "overrides any earlier clustering guidance or direction" in retry_system_messages[1]
    assert f"Injected instruction:\n{resolved_prompt_text}" in retry_system_messages[1]
    assert isinstance(first_invocation[1], AIMessage)
    assert str(first_invocation[1].content) == "What would you like to cluster next?"
    assert isinstance(first_invocation[2], SystemMessage)
    assert "A new instruction takes effect starting this turn" in str(first_invocation[2].content)
    assert "overrides any earlier clustering guidance or direction" in str(
        first_invocation[2].content
    )
    assert f"Injected instruction:\n{resolved_prompt_text}" in str(first_invocation[2].content)
    assert isinstance(retry_invocation[1], AIMessage)
    assert str(retry_invocation[1].content) == "What would you like to cluster next?"
    assert isinstance(retry_invocation[2], SystemMessage)
    assert "A new instruction takes effect starting this turn" in str(retry_invocation[2].content)
    assert "overrides any earlier clustering guidance or direction" in str(
        retry_invocation[2].content
    )
    assert any("Make the message more explicit." in message for message in retry_system_messages)


def test_build_human_app_stabilizes_theme_switch_after_trigger_turn() -> None:
    turn_injection = _theme_switch_turn_injection(2)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Let's regroup these around ticket category instead.",
                thought="Switch to a new theme on the trigger turn.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Can you describe the current ticket-category clusters before saving?",
                thought="Keep the new theme stable instead of switching again.",
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            turn_injection=turn_injection,
        )

    app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster next?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 1,
        }
    )
    app.invoke(
        {
            "messages": [AIMessage(content="Anything else before I save?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
        }
    )

    trigger_turn_system_messages = [
        str(message.content)
        for message in human_llm.invocations[0]
        if isinstance(message, SystemMessage)
    ]
    later_turn_system_messages = [
        str(message.content)
        for message in human_llm.invocations[1]
        if isinstance(message, SystemMessage)
    ]

    assert any(
        "A new instruction takes effect starting this turn" in message
        for message in trigger_turn_system_messages
    )
    assert any(
        f"Injected instruction:\n{resolved_prompt_text}" in message
        for message in trigger_turn_system_messages
    )
    assert not any(
        "A new instruction takes effect starting this turn" in message
        for message in later_turn_system_messages
    )
    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in later_turn_system_messages
    )


def test_build_human_app_suppresses_selected_topic_guidance_when_theme_switch_is_active() -> None:
    turn_injection = _theme_switch_turn_injection(2)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Let's switch to ticket category instead.",
                thought="Override the selected-topic direction on the trigger turn.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Keep describing the ticket-category clusters before saving.",
                thought="Continue the switched theme on later turns.",
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            turn_injection=turn_injection,
            selected_topic_label="Tone",
            selected_topic_guidance="Group items by tone.",
            selected_topic_generated_at_turn=2,
        )

    app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster next?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 1,
        }
    )
    app.invoke(
        {
            "messages": [AIMessage(content="Anything else before I save?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
        }
    )

    trigger_turn_system_messages = [
        str(message.content)
        for message in human_llm.invocations[0]
        if isinstance(message, SystemMessage)
    ]
    later_turn_system_messages = [
        str(message.content)
        for message in human_llm.invocations[1]
        if isinstance(message, SystemMessage)
    ]

    assert any(
        f"Injected instruction:\n{resolved_prompt_text}" in message
        for message in trigger_turn_system_messages
    )
    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in later_turn_system_messages
    )
    assert not any(
        "A clustering topic has already been selected for this continuation." in message
        for message in trigger_turn_system_messages
    )
    assert not any(
        "A clustering topic has already been selected for this continuation." in message
        for message in later_turn_system_messages
    )


def test_build_human_app_keeps_selected_topic_guidance_for_coarsen() -> None:
    turn_injection = _granularity_turn_injection(2, edit_primitive="coarsen")
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Please merge these into broader tone buckets.",
                thought="Keep the same topic but ask for a coarser grouping.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Keep the broader tone buckets and summarize them.",
                thought="Continue the same coarse-grained topic on the next turn.",
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            turn_injection=turn_injection,
            selected_topic_label="Tone",
            selected_topic_guidance="Group items by tone.",
            selected_topic_generated_at_turn=2,
        )

    app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster next?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 1,
        }
    )
    app.invoke(
        {
            "messages": [AIMessage(content="Anything else before I save?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
        }
    )

    trigger_turn_system_messages = [
        str(message.content)
        for message in human_llm.invocations[0]
        if isinstance(message, SystemMessage)
    ]
    later_turn_system_messages = [
        str(message.content)
        for message in human_llm.invocations[1]
        if isinstance(message, SystemMessage)
    ]

    assert any(
        f"Injected instruction:\n{resolved_prompt_text}" in message
        for message in trigger_turn_system_messages
    )
    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in later_turn_system_messages
    )
    assert any(
        "A clustering topic has already been selected for this continuation." in message
        for message in trigger_turn_system_messages
    )
    assert any(
        "Keep steering toward the topic 'Tone'." in message
        for message in trigger_turn_system_messages
    )
    assert any(
        "A clustering topic has already been selected for this continuation." in message
        for message in later_turn_system_messages
    )
    assert any(
        "Keep steering toward the topic 'Tone'." in message
        for message in later_turn_system_messages
    )


def test_build_human_app_preserves_consumed_theme_switch_guidance_in_continuations() -> None:
    turn_injection = _theme_switch_turn_injection(2)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Keep refining the ticket-category clusters before saving.",
                thought="Continue the post-switch topic on this branch.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Can you summarize the ticket-category clusters now?",
                thought="Preserve the continuation topic on later turns too.",
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            turn_injection=turn_injection,
            selected_topic_label="Tone",
            selected_topic_guidance="Group items by tone.",
            selected_topic_generated_at_turn=2,
        )

    app.invoke(
        {
            "messages": [AIMessage(content="Anything else before I save?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
            "turn_injection_consumed_turn_index": 2,
        }
    )
    app.invoke(
        {
            "messages": [AIMessage(content="One more adjustment?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 3,
            "turn_injection_consumed_turn_index": 2,
        }
    )

    first_continuation_system_messages = [
        str(message.content)
        for message in human_llm.invocations[0]
        if isinstance(message, SystemMessage)
    ]
    later_continuation_system_messages = [
        str(message.content)
        for message in human_llm.invocations[1]
        if isinstance(message, SystemMessage)
    ]

    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in first_continuation_system_messages
    )
    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in later_continuation_system_messages
    )
    assert any(
        "A clustering topic has already been selected for this continuation." in message
        for message in first_continuation_system_messages
    )
    assert any(
        "Keep steering toward the topic 'Tone'." in message
        for message in first_continuation_system_messages
    )
    assert any(
        "A clustering topic has already been selected for this continuation." in message
        for message in later_continuation_system_messages
    )
    assert any(
        "Keep steering toward the topic 'Tone'." in message
        for message in later_continuation_system_messages
    )


def test_build_human_app_keeps_pre_switch_selected_topic_suppressed_after_theme_switch() -> None:
    turn_injection = _theme_switch_turn_injection(2)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Let's keep exploring the new direction before saving.",
                thought="Stay on the switched theme without reviving the old branch topic.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Can you summarize the new grouping now?",
                thought="Do not reintroduce the pre-switch selected topic later either.",
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            turn_injection=turn_injection,
            selected_topic_label="Tone",
            selected_topic_guidance="Group items by tone.",
            selected_topic_generated_at_turn=1,
        )

    app.invoke(
        {
            "messages": [AIMessage(content="Anything else before I save?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
            "turn_injection_consumed_turn_index": 2,
        }
    )
    app.invoke(
        {
            "messages": [AIMessage(content="One more adjustment?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 3,
            "turn_injection_consumed_turn_index": 2,
        }
    )

    first_continuation_system_messages = [
        str(message.content)
        for message in human_llm.invocations[0]
        if isinstance(message, SystemMessage)
    ]
    later_continuation_system_messages = [
        str(message.content)
        for message in human_llm.invocations[1]
        if isinstance(message, SystemMessage)
    ]

    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in first_continuation_system_messages
    )
    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in later_continuation_system_messages
    )
    assert not any(
        "A clustering topic has already been selected for this continuation." in message
        for message in first_continuation_system_messages
    )
    assert not any(
        "A clustering topic has already been selected for this continuation." in message
        for message in later_continuation_system_messages
    )


def test_build_human_app_preserves_turn_injection_after_topic_handoff() -> None:
    turn_injection = _theme_switch_turn_injection(2)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    discovery_human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I should enumerate topics before the handoff.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly in the discussion.",
                    }
                ],
                input_tokens=1,
                output_tokens=1,
            )
        ]
    )
    empty_verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=discovery_human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=empty_verifier_llm),
    ):
        discovery_app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
            ),
            turn_injection=turn_injection,
        )

    handoff_state = discovery_app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster next?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 1,
        }
    )

    handoff_state["turn_injection_consumed_turn_index"] = 2

    continuation_human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Let's keep clustering around tone.",
                thought="Continue from the selected topic while preserving the injected edit.",
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Can you describe the tone-based clusters before saving?",
                thought="Keep following both the selected topic and the injected edit later.",
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    continuation_verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 1, "output_tokens": 1}},
                ),
            },
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=continuation_human_llm),
        mock.patch.object(
            human_agent, "build_human_verifier_llm", return_value=continuation_verifier_llm
        ),
    ):
        continuation_app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            turn_injection=turn_injection,
            selected_topic_label="Tone",
            selected_topic_guidance="Group items by tone.",
            selected_topic_generated_at_turn=2,
        )

    continuation_app.invoke(handoff_state)
    continuation_app.invoke(
        {
            "messages": [AIMessage(content="Anything else before I save?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
            "turn_injection_consumed_turn_index": 2,
        }
    )

    first_continuation_system_messages = [
        str(message.content)
        for message in continuation_human_llm.invocations[0]
        if isinstance(message, SystemMessage)
    ]
    later_continuation_system_messages = [
        str(message.content)
        for message in continuation_human_llm.invocations[1]
        if isinstance(message, SystemMessage)
    ]

    assert first_continuation_system_messages[0] == "Base system prompt."
    assert later_continuation_system_messages[0] == "Base system prompt."
    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in first_continuation_system_messages
    )
    assert any(
        f"Original injected instruction:\n{resolved_prompt_text}" in message
        for message in later_continuation_system_messages
    )
    assert any(
        "Keep steering toward the topic 'Tone'." in message
        for message in first_continuation_system_messages
    )
    assert any(
        "Keep steering toward the topic 'Tone'." in message
        for message in later_continuation_system_messages
    )


def test_build_human_app_enumerates_topics_via_tool_when_configured() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I have enough context to propose multiple viable directions.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone and communication style.",
                        "rationale": "Different tones appeared repeatedly in the summaries.",
                    },
                    {
                        "label": "Audience",
                        "guidance": "Group items by intended audience.",
                        "rationale": "The dataset seems to separate novice and expert material.",
                    },
                ],
                input_tokens=2,
                output_tokens=3,
            ),
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I should avoid duplicates and keep only novel ideas.",
                topics=[
                    {
                        "label": "tone",
                        "guidance": "Group items by tone and communication style.",
                        "rationale": "This repeats an existing topic.",
                    }
                ],
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
            ),
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
        }
    )

    assert result["topic_candidates"] == [
        {
            "topic_id": "tone",
            "label": "Tone",
            "guidance": "Group items by tone and communication style.",
            "rationale": "Different tones appeared repeatedly in the summaries.",
        },
        {
            "topic_id": "audience",
            "label": "Audience",
            "guidance": "Group items by intended audience.",
            "rationale": "The dataset seems to separate novice and expert material.",
        },
    ]
    assert result["topic_candidates_metadata"] == {
        "trigger_mode": "human_tool",
        "generated_at_turn": 2,
        "num_rounds": 2,
        "new_topics_by_round": [2, 0],
    }
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "What would you like to cluster?"
    assert len(verifier_llm.invocations) == 0


def test_build_human_app_keeps_turn_injection_across_topic_discovery_rounds() -> None:
    turn_injection = _theme_switch_turn_injection(3)
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="First pass with the injected direction.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly in the discussion.",
                    },
                    {
                        "label": "Audience",
                        "guidance": "Group items by audience.",
                        "rationale": "Audience segments also appear repeatedly.",
                    },
                ],
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="Second pass should still see the injected direction.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "This repeats the first round on purpose.",
                    }
                ],
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Base system prompt.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
            ),
            turn_injection=turn_injection,
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
        }
    )

    first_system_messages = [
        str(message.content)
        for message in human_llm.invocations[0]
        if isinstance(message, SystemMessage)
    ]
    second_system_messages = [
        str(message.content)
        for message in human_llm.invocations[1]
        if isinstance(message, SystemMessage)
    ]

    assert any(
        f"Injected instruction:\n{resolved_prompt_text}" in message
        for message in first_system_messages
    )
    assert any(
        f"Injected instruction:\n{resolved_prompt_text}" in message
        for message in second_system_messages
    )
    assert result["topic_candidates_metadata"] == {
        "trigger_mode": "human_tool",
        "generated_at_turn": 2,
        "num_rounds": 2,
        "new_topics_by_round": [2, 0],
    }


def test_build_human_app_accumulates_topics_across_rounds_until_novelty_gate() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="First pass to gather broad options.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly in dataset summaries.",
                    },
                    {
                        "label": "Audience",
                        "guidance": "Group items by audience.",
                        "rationale": "Audience levels are visibly mixed.",
                    },
                ],
                input_tokens=1,
                output_tokens=2,
            ),
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="Second pass to add only missing ideas.",
                topics=[
                    {
                        "label": "tone",
                        "guidance": "Group items by overall tone.",
                        "rationale": "This repeats a prior dimension.",
                    },
                    {
                        "label": "Genre",
                        "guidance": "Group items by genre patterns.",
                        "rationale": "Genre cues appear in many item descriptions.",
                    },
                ],
                input_tokens=3,
                output_tokens=4,
            ),
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore first.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
            ),
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
        }
    )

    assert [topic["topic_id"] for topic in result["topic_candidates"]] == [
        "tone",
        "audience",
        "genre",
    ]
    assert result["topic_candidates_metadata"] == {
        "trigger_mode": "human_tool",
        "generated_at_turn": 2,
        "num_rounds": 2,
        "new_topics_by_round": [2, 1],
    }
    assert result["human_turn_token_usage"] == {
        "input_tokens": 4,
        "output_tokens": 6,
        "total_tokens": 10,
    }
    assert len(human_llm.invocations) == 2
    second_invocation_last_message = human_llm.invocations[1][-1]
    assert isinstance(second_invocation_last_message, HumanMessage)
    assert "Generate only novel topic candidates" in str(second_invocation_last_message.content)


def test_build_human_app_stops_topic_enumeration_at_max_rounds() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="Round 1.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly.",
                    },
                    {
                        "label": "Audience",
                        "guidance": "Group items by audience.",
                        "rationale": "Audience varies widely.",
                    },
                ],
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="Round 2.",
                topics=[
                    {
                        "label": "Genre",
                        "guidance": "Group items by genre.",
                        "rationale": "Genre cues are explicit.",
                    },
                    {
                        "label": "Pacing",
                        "guidance": "Group items by pacing.",
                        "rationale": "Pacing differences are visible.",
                    },
                ],
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="Round 3.",
                topics=[
                    {
                        "label": "Format",
                        "guidance": "Group items by format.",
                        "rationale": "Formats differ across entries.",
                    },
                    {
                        "label": "Intent",
                        "guidance": "Group items by intent.",
                        "rationale": "User intents vary by item.",
                    },
                ],
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore first.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
            ),
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 1,
        }
    )

    assert result["topic_candidates_metadata"] == {
        "trigger_mode": "human_tool",
        "generated_at_turn": 1,
        "num_rounds": 3,
        "new_topics_by_round": [2, 2, 2],
    }
    assert len(result["topic_candidates"]) == 6
    assert len(human_llm.invocations) == 3


def test_build_human_app_uses_configurable_topic_discovery_round_thresholds() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="Round 1.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly.",
                    },
                    {
                        "label": "Audience",
                        "guidance": "Group items by audience.",
                        "rationale": "Audience varies widely.",
                    },
                ],
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="Round 2.",
                topics=[
                    {
                        "label": "Genre",
                        "guidance": "Group items by genre.",
                        "rationale": "Genre cues are explicit.",
                    }
                ],
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore first.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
                max_rounds=2,
                min_new_topics_per_round=1,
            ),
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 1,
        }
    )

    assert result["topic_candidates_metadata"] == {
        "trigger_mode": "human_tool",
        "generated_at_turn": 1,
        "num_rounds": 2,
        "new_topics_by_round": [2, 1],
    }
    assert [topic["topic_id"] for topic in result["topic_candidates"]] == [
        "tone",
        "audience",
        "genre",
    ]
    assert len(human_llm.invocations) == 2


def test_build_human_app_retries_when_topic_discovery_is_disabled() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I want to enumerate topics even though the tool is unavailable.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears often.",
                    }
                ],
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Could you show me a few representative samples first?",
                thought="I should ask a normal question instead.",
                input_tokens=2,
                output_tokens=2,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 3, "output_tokens": 4}},
                ),
            }
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 0,
        }
    )

    final_message = result["messages"][-1]
    assert isinstance(final_message, HumanMessage)
    assert final_message.content == "Could you show me a few representative samples first?"
    assert len(human_llm.invocations) == 2


def test_build_human_app_raises_when_topic_discovery_is_disabled_and_retry_budget_exhausted() -> (
    None
):
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I want to enumerate topics even though the tool is unavailable.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears often.",
                    }
                ],
                input_tokens=1,
                output_tokens=1,
            )
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
            max_verification_retries=0,
        )

    with pytest.raises(ValueError, match="Topic discovery is unavailable"):
        app.invoke(
            {
                "messages": [AIMessage(content="What would you like to cluster?")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 0,
            }
        )


def test_build_human_app_retries_after_failed_topic_enumeration() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I think I have enough context.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears often.",
                    },
                    {
                        "label": "tone",
                        "guidance": "Group items by style.",
                        "rationale": "Style also appears often.",
                    },
                ],
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                message="Could you show me a few representative samples first?",
                thought="I need more context after the failed topic attempt.",
                input_tokens=2,
                output_tokens=2,
            ),
        ]
    )
    verifier_llm = FakeLLM(
        responses=[
            {
                "parsed": {"decision": "allow", "feedback": ""},
                "raw": AIMessage(
                    content='{"decision":"allow","feedback":""}',
                    additional_kwargs={"token_usage": {"input_tokens": 3, "output_tokens": 4}},
                ),
            }
        ]
    )

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
            ),
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 0,
        }
    )

    final_message = result["messages"][-1]
    assert isinstance(final_message, HumanMessage)
    assert final_message.content == "Could you show me a few representative samples first?"
    assert not any(
        "Could not store topic candidates" in str(message.content)
        for message in result["messages"]
        if isinstance(message, HumanMessage | AIMessage)
    )
    assert len(human_llm.invocations) == 2


def test_build_human_app_raises_when_topics_already_exist_and_retry_budget_exhausted() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I want to enumerate topics again.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears often.",
                    }
                ],
                input_tokens=1,
                output_tokens=1,
            )
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
            ),
            max_verification_retries=0,
        )

    with pytest.raises(ValueError, match="already been stored"):
        app.invoke(
            {
                "messages": [AIMessage(content="What would you like to cluster?")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 0,
                "topic_candidates": [
                    {
                        "topic_id": "tone",
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears often.",
                    }
                ],
            }
        )


def test_build_human_app_retries_when_topic_tool_is_required() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Could you show me a few more samples first?",
                thought="I want more context before picking a topic.",
                input_tokens=1,
                output_tokens=2,
            ),
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I need to enumerate topics now that the turn budget is exhausted.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly in the summaries.",
                    }
                ],
                input_tokens=2,
                output_tokens=3,
            ),
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="turn_budget",
                turn_budget=2,
            ),
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
            "topic_discovery_force_generation": True,
        }
    )

    assert result["topic_candidates"] == [
        {
            "topic_id": "tone",
            "label": "Tone",
            "guidance": "Group items by tone.",
            "rationale": "Tone appears repeatedly in the summaries.",
        }
    ]
    assert result["verifier_decision"] == "allow"
    assert len(human_llm.invocations) == 2


def test_build_human_app_raises_when_topic_tool_is_required_and_retry_budget_exhausted() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                message="Could you show me a few more samples first?",
                thought="I want more context before picking a topic.",
                input_tokens=1,
                output_tokens=2,
            )
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="turn_budget",
                turn_budget=2,
            ),
            max_verification_retries=0,
        )

    with pytest.raises(ValueError, match="Topic discovery is required now"):
        app.invoke(
            {
                "messages": [AIMessage(content="What would you like to cluster?")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 2,
                "topic_discovery_force_generation": True,
            }
        )


def test_build_human_app_keeps_forced_topic_generation_after_invalid_topic_payload() -> None:
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I need to enumerate topics now.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly in the summaries.",
                    },
                    {
                        "label": "tone",
                        "guidance": "Group items by style.",
                        "rationale": "Style also appears repeatedly in the summaries.",
                    },
                ],
                input_tokens=1,
                output_tokens=1,
            ),
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I need to retry with a valid topic list.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly in the summaries.",
                    }
                ],
                input_tokens=2,
                output_tokens=2,
            ),
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="turn_budget",
                turn_budget=2,
            ),
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
            "topic_discovery_force_generation": True,
        }
    )

    assert result["topic_candidates"] == [
        {
            "topic_id": "tone",
            "label": "Tone",
            "guidance": "Group items by tone.",
            "rationale": "Tone appears repeatedly in the summaries.",
        }
    ]
    assert len(human_llm.invocations) == 2
    second_invocation_last_message = human_llm.invocations[1][-1]
    assert isinstance(second_invocation_last_message, HumanMessage)
    assert "Call the enumerate_topics tool now" in str(second_invocation_last_message.content)


def test_build_human_app_clears_stale_verifier_feedback_after_successful_topic_enumeration() -> (
    None
):
    human_llm = FakeLLM(
        responses=[
            _human_draft_response(
                action="enumerate_topics",
                message="",
                thought="I have enough context to enumerate topics now.",
                topics=[
                    {
                        "label": "Tone",
                        "guidance": "Group items by tone.",
                        "rationale": "Tone appears repeatedly in the summaries.",
                    }
                ],
                input_tokens=1,
                output_tokens=1,
            )
        ]
    )
    verifier_llm = FakeLLM(responses=[])

    with (
        mock.patch.object(human_agent, "build_human_llm", return_value=human_llm),
        mock.patch.object(human_agent, "build_human_verifier_llm", return_value=verifier_llm),
    ):
        app = build_human_app(
            guidance=None,
            system_prompt="Explore the dataset before choosing a basis.",
            topic_discovery_config=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="human_tool",
            ),
        )

    result = app.invoke(
        {
            "messages": [AIMessage(content="What would you like to cluster?")],
            "dataset_metadata": {"size": 1},
            "iteration_count": 0,
            "verifier_feedback": "Write a more natural follow-up.",
            "verifier_retry_count": 1,
            "human_turn_token_usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
    )

    assert result["verifier_feedback"] == ""
    assert result["verifier_decision"] == "allow"
