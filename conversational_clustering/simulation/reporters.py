"""Reporter interfaces and implementations for simulation output handling."""

from collections.abc import Mapping
from typing import Protocol


class SimulationReporter(Protocol):
    def on_agent_chunk(self, content: str, *, is_start: bool) -> None: ...

    def on_agent_message(self, content: str, *, streamed: bool) -> None: ...

    def on_tool_call(self, tool_name: str) -> None: ...

    def on_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        is_error: bool,
        state: Mapping[str, object] | None = None,
    ) -> None: ...

    def on_tool_error(self, tool_name: str) -> None: ...

    def on_tokens(self, token_usage: dict, clustering_agent_session_tokens: dict) -> None: ...

    def on_human_tokens(self, token_usage: dict, human_agent_session_tokens: dict) -> None: ...

    def on_system_message(self, message: str) -> None: ...

    def on_human_message(self, message: str, *, simulated: bool) -> None: ...

    def on_simulation_start(self, max_turns: int, guidance: str | None) -> None: ...

    def on_simulation_end(
        self,
        reason: str,
        *,
        clustering_agent_session_tokens: dict | None = None,
        human_agent_session_tokens: dict | None = None,
    ) -> None: ...


class NullReporter:
    def on_agent_chunk(self, content: str, *, is_start: bool) -> None:
        return None

    def on_agent_message(self, content: str, *, streamed: bool) -> None:
        return None

    def on_tool_call(self, tool_name: str) -> None:
        return None

    def on_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        is_error: bool,
        state: Mapping[str, object] | None = None,
    ) -> None:
        _ = tool_name, content, is_error, state
        return None

    def on_tool_error(self, tool_name: str) -> None:
        return None

    def on_tokens(self, token_usage: dict, clustering_agent_session_tokens: dict) -> None:
        return None

    def on_human_tokens(self, token_usage: dict, human_agent_session_tokens: dict) -> None:
        return None

    def on_system_message(self, message: str) -> None:
        return None

    def on_human_message(self, message: str, *, simulated: bool) -> None:
        return None

    def on_simulation_start(self, max_turns: int, guidance: str | None) -> None:
        return None

    def on_simulation_end(
        self,
        reason: str,
        *,
        clustering_agent_session_tokens: dict | None = None,
        human_agent_session_tokens: dict | None = None,
    ) -> None:
        return None


class ConsoleReporter:
    def on_agent_chunk(self, content: str, *, is_start: bool) -> None:
        if is_start:
            print("Agent: ", end="", flush=True)
        print(content, end="", flush=True)

    def on_agent_message(self, content: str, *, streamed: bool) -> None:
        if streamed:
            print()
        else:
            print(f"Agent: {content}")

    def on_tool_call(self, tool_name: str) -> None:
        print(f"System: Executing tool: {tool_name}")

    def on_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        is_error: bool,
        state: Mapping[str, object] | None = None,
    ) -> None:
        _ = is_error, state
        if content.strip():
            print(f"System: Tool result from {tool_name}: {content}")

    def on_tool_error(self, tool_name: str) -> None:
        print(f"System: Tool {tool_name} failed.")

    def on_tokens(self, token_usage: dict, clustering_agent_session_tokens: dict) -> None:
        print(
            "Clustering Tokens: "
            f"in={token_usage['input_tokens']} "
            f"out={token_usage['output_tokens']} "
            f"total={token_usage['total_tokens']} "
            "| "
            "Clustering Agent Session Tokens: "
            f"in={clustering_agent_session_tokens['input_tokens']} "
            f"out={clustering_agent_session_tokens['output_tokens']} "
            f"total={clustering_agent_session_tokens['total_tokens']}"
        )

    def on_human_tokens(self, token_usage: dict, human_agent_session_tokens: dict) -> None:
        print(
            "Human Tokens: "
            f"in={token_usage['input_tokens']} "
            f"out={token_usage['output_tokens']} "
            f"total={token_usage['total_tokens']} "
            "| "
            "Human Agent Session Tokens: "
            f"in={human_agent_session_tokens['input_tokens']} "
            f"out={human_agent_session_tokens['output_tokens']} "
            f"total={human_agent_session_tokens['total_tokens']}"
        )

    def on_system_message(self, message: str) -> None:
        print(message)

    def on_human_message(self, message: str, *, simulated: bool) -> None:
        if simulated:
            print(f"Human (simulated): {message}")
        else:
            print(f"User: {message}")

    def on_simulation_start(self, max_turns: int, guidance: str | None) -> None:
        print(f"Simulation mode: turns={max_turns}")
        if guidance:
            print(f"Simulation guidance: {guidance}")

    def on_simulation_end(
        self,
        reason: str,
        *,
        clustering_agent_session_tokens: dict | None = None,
        human_agent_session_tokens: dict | None = None,
    ) -> None:
        if not clustering_agent_session_tokens and not human_agent_session_tokens:
            return None
        clustering_total = (
            clustering_agent_session_tokens.get("total_tokens", 0)
            if clustering_agent_session_tokens
            else 0
        )
        human_total = (
            human_agent_session_tokens.get("total_tokens", 0) if human_agent_session_tokens else 0
        )
        combined_total = clustering_total + human_total
        print(
            "Total Tokens Used: "
            f"clustering={clustering_total} "
            f"human={human_total} "
            f"combined={combined_total}"
        )
