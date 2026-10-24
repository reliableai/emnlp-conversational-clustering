"""Typed API schemas for the standalone demo session and streaming bridge."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolTraceMetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool_trace"] = "tool_trace"
    tool_name: str
    display_name: str
    step_index: int = Field(ge=1)
    stage: Literal["running", "failed"]
    details: list[str] = Field(default_factory=list)


class TranscriptEntryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    status: Literal["complete", "streaming", "error"] = "complete"
    cumulative_token_total: int | None = Field(default=None, ge=0)
    metadata: ToolTraceMetadataModel | None = None


class ProposalClusterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    size: int = Field(ge=0)
    summary: str
    representative_points: list[str]


class ProposalSnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    embedding_instruction: str
    num_clusters: int = Field(ge=1)
    clusters: list[ProposalClusterModel]


class TokenTotalsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class SessionSnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    transcript: list[TranscriptEntryModel]
    latest_proposal: ProposalSnapshotModel | None = None
    proposal_history: list[ProposalSnapshotModel] = Field(default_factory=list)
    is_turn_in_progress: bool
    token_totals: TokenTotalsModel


class CreateSessionResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: SessionSnapshotModel


class SendMessageRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped
