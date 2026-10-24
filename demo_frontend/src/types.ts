export type TranscriptRole = "user" | "assistant" | "system";
export type TranscriptStatus = "complete" | "streaming" | "error";

export interface ToolTraceMetadata {
  kind: "tool_trace";
  tool_name: string;
  display_name: string;
  step_index: number;
  stage: "running" | "failed";
  details: string[];
}

export interface TranscriptEntry {
  id: string;
  role: TranscriptRole;
  content: string;
  status: TranscriptStatus;
  cumulative_token_total?: number | null;
  metadata?: ToolTraceMetadata | null;
}

export interface ProposalCluster {
  name: string;
  size: number;
  summary: string;
  representative_points: string[];
}

export interface ProposalSnapshot {
  proposal_id: string;
  embedding_instruction: string;
  num_clusters: number;
  clusters: ProposalCluster[];
}

export interface TokenTotals {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface SessionSnapshot {
  session_id: string;
  transcript: TranscriptEntry[];
  latest_proposal: ProposalSnapshot | null;
  proposal_history: ProposalSnapshot[];
  is_turn_in_progress: boolean;
  token_totals: TokenTotals;
}

export interface CreateSessionResponse {
  session: SessionSnapshot;
}

export interface EntryEnvelope {
  entry: TranscriptEntry;
}

export interface AgentChunkEvent {
  entry_id: string;
  content: string;
  is_start: boolean;
}

export interface AgentMessageEvent {
  entry: TranscriptEntry;
  streamed: boolean;
}

export interface ProposalUpdatedEvent {
  proposal: ProposalSnapshot;
}

export interface TokenUsageEvent {
  token_totals: TokenTotals;
  entry?: TranscriptEntry;
}

export interface TurnFinishedEvent {
  session: SessionSnapshot;
}
