import { FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";

import { connectToSessionEvents, createSession, fetchSessionState, sendMessage } from "./api";
import type {
  AgentChunkEvent,
  AgentMessageEvent,
  EntryEnvelope,
  ProposalSnapshot,
  ProposalUpdatedEvent,
  SessionSnapshot,
  TokenUsageEvent,
  ToolTraceMetadata,
  TranscriptEntry,
  TurnFinishedEvent,
} from "./types";

const DEFAULT_PROPOSAL_PANEL_WIDTH = 640;
const MIN_PROPOSAL_PANEL_WIDTH = 420;
const MAX_PROPOSAL_PANEL_WIDTH = 820;
const STARTER_PROMPTS = [
  "What kind of data is this?",
  "What are some interesting POVs in this data?",
  "What is unique about this data?",
];

function upsertTranscriptEntry(
  transcript: TranscriptEntry[],
  nextEntry: TranscriptEntry,
): TranscriptEntry[] {
  const existingIndex = transcript.findIndex((entry) => entry.id === nextEntry.id);
  if (existingIndex === -1) {
    return [...transcript, nextEntry];
  }
  return transcript.map((entry, index) => (index === existingIndex ? nextEntry : entry));
}

function appendAgentChunk(
  transcript: TranscriptEntry[],
  payload: AgentChunkEvent,
): TranscriptEntry[] {
  const existingIndex = transcript.findIndex((entry) => entry.id === payload.entry_id);
  if (existingIndex === -1) {
    return [
      ...transcript,
      {
        id: payload.entry_id,
        role: "assistant",
        content: payload.content,
        status: "streaming",
      },
    ];
  }
  return transcript.map((entry, index) =>
    index === existingIndex
      ? {
          ...entry,
          content: payload.is_start ? payload.content : `${entry.content}${payload.content}`,
          status: "streaming",
        }
      : entry,
  );
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const pattern = /(\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*)/g;
  const nodes: ReactNode[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0;
    if (start > lastIndex) {
      nodes.push(text.slice(lastIndex, start));
    }

    if (match[2] && match[3]) {
      nodes.push(
        <a
          key={`link-${start}`}
          href={match[3]}
          rel="noreferrer"
          target="_blank"
        >
          {match[2]}
        </a>,
      );
    } else if (match[4]) {
      nodes.push(<code key={`code-${start}`}>{match[4]}</code>);
    } else if (match[5]) {
      nodes.push(<strong key={`strong-${start}`}>{match[5]}</strong>);
    } else if (match[6]) {
      nodes.push(<em key={`em-${start}`}>{match[6]}</em>);
    }

    lastIndex = start + match[0].length;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

function normalizeMarkdownContent(content: string): string {
  return content
    .replace(/:\s+-\s+/g, ":\n- ")
    .replace(/\s+-\s+(?=[A-Za-z][A-Za-z/& ]{1,40}:)/g, "\n- ")
    .replace(/:\s+(?=\d+(?:[.)])?\s+[a-z])/g, ":\n")
    .replace(/,\s+(?:or|and)\s+(?=\d+(?:[.)])?\s+)/gi, "\n");
}

function startsMarkdownBlock(line: string): boolean {
  const trimmed = line.trim();
  return (
    trimmed.startsWith("```") ||
    trimmed.startsWith(">") ||
    /^#{1,3}\s+/.test(trimmed) ||
    /^[-*]\s+/.test(trimmed) ||
    /^\d+(?:[.)])?\s+/.test(trimmed)
  );
}

function renderMarkdown(content: string): ReactNode {
  const lines = normalizeMarkdownContent(content).split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let index = 0;
  let blockKey = 0;

  while (index < lines.length) {
    const current = lines[index];
    const trimmed = current.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push(
        <pre key={`block-${blockKey++}`}>
          <code>{codeLines.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const headingContent = renderInlineMarkdown(headingMatch[2]);
      if (level === 1) {
        blocks.push(<h1 key={`block-${blockKey++}`}>{headingContent}</h1>);
      } else if (level === 2) {
        blocks.push(<h2 key={`block-${blockKey++}`}>{headingContent}</h2>);
      } else {
        blocks.push(<h3 key={`block-${blockKey++}`}>{headingContent}</h3>);
      }
      index += 1;
      continue;
    }

    if (trimmed.startsWith(">")) {
      const quoteLines: string[] = [];
      while (index < lines.length && lines[index].trim().startsWith(">")) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }
      blocks.push(
        <blockquote key={`block-${blockKey++}`}>
          <p>{renderInlineMarkdown(quoteLines.join(" "))}</p>
        </blockquote>,
      );
      continue;
    }

    const unorderedListMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (unorderedListMatch) {
      const items: ReactNode[] = [];
      while (index < lines.length) {
        const listMatch = lines[index].trim().match(/^[-*]\s+(.+)$/);
        if (!listMatch) {
          break;
        }
        items.push(<li key={`item-${blockKey}-${index}`}>{renderInlineMarkdown(listMatch[1])}</li>);
        index += 1;
      }
      blocks.push(<ul key={`block-${blockKey++}`}>{items}</ul>);
      continue;
    }

    const orderedListMatch = trimmed.match(/^\d+(?:[.)])?\s+(.+)$/);
    if (orderedListMatch) {
      const items: ReactNode[] = [];
      while (index < lines.length) {
        if (!lines[index].trim()) {
          const nextLine = lines[index + 1];
          if (nextLine !== undefined && /^\d+(?:[.)])?\s+/.test(nextLine.trim())) {
            index += 1;
            continue;
          }
          break;
        }
        const listMatch = lines[index].trim().match(/^\d+(?:[.)])?\s+(.+)$/);
        if (!listMatch) {
          break;
        }
        items.push(<li key={`item-${blockKey}-${index}`}>{renderInlineMarkdown(listMatch[1])}</li>);
        index += 1;
      }
      blocks.push(<ol key={`block-${blockKey++}`}>{items}</ol>);
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length && lines[index].trim()) {
      if (paragraphLines.length > 0 && startsMarkdownBlock(lines[index])) {
        break;
      }
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    blocks.push(
      <p key={`block-${blockKey++}`}>{renderInlineMarkdown(paragraphLines.join(" "))}</p>,
    );
  }

  return blocks;
}

function isToolTraceEntry(entry: TranscriptEntry): boolean {
  return entry.role === "system" && entry.metadata?.kind === "tool_trace";
}

function shouldRenderEntry(entry: TranscriptEntry): boolean {
  if (entry.role === "assistant" && !entry.content.trim()) {
    return false;
  }
  if (entry.role === "system" && !entry.content.trim()) {
    return false;
  }
  return true;
}

function getToolTraceMetadata(entry: TranscriptEntry): ToolTraceMetadata | null {
  return entry.metadata?.kind === "tool_trace" ? entry.metadata : null;
}

function getPendingProposalTrace(
  transcript: TranscriptEntry[],
  isTurnInProgress: boolean,
): ToolTraceMetadata | null {
  if (!isTurnInProgress) {
    return null;
  }

  for (let index = transcript.length - 1; index >= 0; index -= 1) {
    const metadata = getToolTraceMetadata(transcript[index]);
    if (metadata === null || metadata.tool_name !== "propose_clustering") {
      continue;
    }
    if (metadata.stage !== "running" || metadata.details.length > 0) {
      return null;
    }
    return metadata;
  }

  return null;
}

function hasStreamingAssistantContent(transcript: TranscriptEntry[]): boolean {
  return transcript.some(
    (entry) =>
      entry.role === "assistant" &&
      entry.status === "streaming" &&
      entry.content.trim().length > 0,
  );
}

function clampProposalPanelWidth(width: number, viewportWidth: number): number {
  const maxWidth = Math.min(MAX_PROPOSAL_PANEL_WIDTH, Math.floor(viewportWidth * 0.55));
  const minWidth = Math.min(MIN_PROPOSAL_PANEL_WIDTH, maxWidth);
  return Math.min(Math.max(width, minWidth), maxWidth);
}

function proposalSortValue(proposalId: string): number {
  const match = proposalId.match(/^proposal_(\d+)$/);
  if (match === null) {
    return Number.MAX_SAFE_INTEGER;
  }
  return Number.parseInt(match[1], 10);
}

function upsertProposalSnapshot(
  proposals: ProposalSnapshot[],
  nextProposal: ProposalSnapshot,
): ProposalSnapshot[] {
  const existingIndex = proposals.findIndex(
    (proposal) => proposal.proposal_id === nextProposal.proposal_id,
  );
  const nextProposals =
    existingIndex === -1
      ? [...proposals, nextProposal]
      : proposals.map((proposal, index) => (index === existingIndex ? nextProposal : proposal));
  return nextProposals.sort(
    (left, right) => proposalSortValue(left.proposal_id) - proposalSortValue(right.proposal_id),
  );
}

function formatTokenNote(tokenTotal: number): string {
  return `${tokenTotal.toLocaleString()} tokens total`;
}

function ClusteringAnimation(): ReactNode {
  return (
    <div
      className="proposal-pending__animation"
      data-testid="clustering-animation"
      aria-hidden="true"
    >
      <svg
        className="clustering-animation"
        viewBox="0 0 240 132"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="cluster-line-gradient" x1="37" y1="20" x2="203" y2="112">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.15" />
            <stop offset="50%" stopColor="#818cf8" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.15" />
          </linearGradient>
        </defs>
        <path
          className="clustering-animation__flow clustering-animation__flow--left"
          d="M32 35C54 35 59 38 79 50C95 60 104 65 120 66"
          pathLength="1"
          stroke="url(#cluster-line-gradient)"
        />
        <path
          className="clustering-animation__flow clustering-animation__flow--upper"
          d="M92 24C110 32 128 38 156 43C174 46 186 46 206 42"
          pathLength="1"
          stroke="url(#cluster-line-gradient)"
        />
        <path
          className="clustering-animation__flow clustering-animation__flow--lower"
          d="M38 100C64 100 75 94 96 84C117 74 131 72 152 78C174 84 186 97 204 105"
          pathLength="1"
          stroke="url(#cluster-line-gradient)"
        />

        <g className="clustering-animation__cluster clustering-animation__cluster--1">
          <circle className="clustering-animation__halo" cx="82" cy="50" r="18" />
          <circle className="clustering-animation__center" cx="82" cy="50" r="6" />
        </g>
        <g className="clustering-animation__cluster clustering-animation__cluster--2">
          <circle className="clustering-animation__halo" cx="170" cy="42" r="16" />
          <circle className="clustering-animation__center" cx="170" cy="42" r="6" />
        </g>
        <g className="clustering-animation__cluster clustering-animation__cluster--3">
          <circle className="clustering-animation__halo" cx="160" cy="86" r="20" />
          <circle className="clustering-animation__center" cx="160" cy="86" r="7" />
        </g>

        <g className="clustering-animation__particle clustering-animation__particle--1">
          <circle cx="82" cy="50" r="3.5" />
        </g>
        <g className="clustering-animation__particle clustering-animation__particle--2">
          <circle cx="82" cy="50" r="2.75" />
        </g>
        <g className="clustering-animation__particle clustering-animation__particle--3">
          <circle cx="82" cy="50" r="3" />
        </g>
        <g className="clustering-animation__particle clustering-animation__particle--4">
          <circle cx="170" cy="42" r="3" />
        </g>
        <g className="clustering-animation__particle clustering-animation__particle--5">
          <circle cx="170" cy="42" r="2.5" />
        </g>
        <g className="clustering-animation__particle clustering-animation__particle--6">
          <circle cx="170" cy="42" r="2.75" />
        </g>
        <g className="clustering-animation__particle clustering-animation__particle--7">
          <circle cx="160" cy="86" r="3.5" />
        </g>
        <g className="clustering-animation__particle clustering-animation__particle--8">
          <circle cx="160" cy="86" r="3" />
        </g>
        <g className="clustering-animation__particle clustering-animation__particle--9">
          <circle cx="160" cy="86" r="2.75" />
        </g>
      </svg>
    </div>
  );
}

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [latestProposal, setLatestProposal] = useState<ProposalSnapshot | null>(null);
  const [proposalHistory, setProposalHistory] = useState<ProposalSnapshot[]>([]);
  const [selectedProposalId, setSelectedProposalId] = useState<string | null>(null);
  const [isProposalPanelVisible, setIsProposalPanelVisible] = useState(false);
  const [expandedClusterSummaries, setExpandedClusterSummaries] = useState<Record<string, boolean>>(
    {},
  );
  const [inputValue, setInputValue] = useState("");
  const [isTurnInProgress, setIsTurnInProgress] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [proposalPanelWidth, setProposalPanelWidth] = useState(DEFAULT_PROPOSAL_PANEL_WIDTH);
  const eventSourceRef = useRef<EventSource | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const composerInputRef = useRef<HTMLTextAreaElement | null>(null);
  const appFrameRef = useRef<HTMLDivElement | null>(null);
  const resizeStateRef = useRef<{
    frameLeft: number;
    frameWidth: number;
  } | null>(null);
  const hasProposalRef = useRef(false);
  const hasPendingProposalRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    async function bootstrap(): Promise<void> {
      try {
        const response = await createSession();
        if (!isMounted) {
          return;
        }
        applySnapshot(response.session);
        setSessionId(response.session.session_id);
      } catch (error) {
        if (!isMounted) {
          return;
        }
        setErrorMessage(error instanceof Error ? error.message : "Failed to initialize the demo.");
      }
    }

    void bootstrap();

    return () => {
      isMounted = false;
      eventSourceRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (sessionId === null) {
      return;
    }

    const eventSource = connectToSessionEvents(sessionId);
    eventSourceRef.current = eventSource;
    const handleEntryEvent = (event: MessageEvent<string>) => {
      const payload = JSON.parse(event.data) as EntryEnvelope;
      setTranscript((current) => upsertTranscriptEntry(current, payload.entry));
    };

    eventSource.addEventListener("open", () => {
      setErrorMessage(null);
    });
    eventSource.addEventListener("user_message", handleEntryEvent);
    eventSource.addEventListener("system_message", handleEntryEvent);
    eventSource.addEventListener("agent_chunk", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as AgentChunkEvent;
      setTranscript((current) => appendAgentChunk(current, payload));
    });
    eventSource.addEventListener("agent_message", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as AgentMessageEvent;
      setTranscript((current) => upsertTranscriptEntry(current, payload.entry));
    });
    eventSource.addEventListener("proposal_updated", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as ProposalUpdatedEvent;
      setLatestProposal(payload.proposal);
      setProposalHistory((current) => upsertProposalSnapshot(current, payload.proposal));
      setSelectedProposalId(payload.proposal.proposal_id);
    });
    eventSource.addEventListener("turn_started", () => {
      setIsTurnInProgress(true);
    });
    eventSource.addEventListener("turn_finished", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as TurnFinishedEvent;
      applySnapshot(payload.session);
    });
    eventSource.addEventListener("token_usage", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as TokenUsageEvent;
      if (payload.entry !== undefined) {
        setTranscript((current) => upsertTranscriptEntry(current, payload.entry!));
      }
    });
    eventSource.addEventListener("session_error", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as { message: string };
      setErrorMessage(payload.message);
    });
    eventSource.onerror = () => {
      void hydrateSession(sessionId);
    };

    return () => {
      eventSource.close();
      if (eventSourceRef.current === eventSource) {
        eventSourceRef.current = null;
      }
    };
  }, [sessionId]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [transcript, isTurnInProgress]);

  useEffect(() => {
    if (sessionId === null || isTurnInProgress) {
      return;
    }
    composerInputRef.current?.focus();
  }, [sessionId, isTurnInProgress]);

  useEffect(() => {
    function handleWindowResize(): void {
      setProposalPanelWidth((current) =>
        clampProposalPanelWidth(current, window.innerWidth),
      );
    }

    handleWindowResize();
    window.addEventListener("resize", handleWindowResize);
    return () => {
      window.removeEventListener("resize", handleWindowResize);
    };
  }, []);

  const hasProposal = latestProposal !== null || proposalHistory.length > 0;

  useEffect(() => {
    if (!hasProposalRef.current && hasProposal) {
      setIsProposalPanelVisible(true);
    }
    hasProposalRef.current = hasProposal;
  }, [hasProposal]);

  const pendingProposalTrace = getPendingProposalTrace(transcript, isTurnInProgress);

  useEffect(() => {
    const hasPendingProposal = pendingProposalTrace !== null;
    if (!hasPendingProposalRef.current && hasPendingProposal) {
      setIsProposalPanelVisible(true);
    }
    hasPendingProposalRef.current = hasPendingProposal;
  }, [pendingProposalTrace]);

  function applySnapshot(snapshot: SessionSnapshot): void {
    setSessionId(snapshot.session_id);
    setTranscript(snapshot.transcript);
    setLatestProposal(snapshot.latest_proposal);
    setProposalHistory(snapshot.proposal_history);
    setSelectedProposalId((current) => {
      if (
        current !== null &&
        snapshot.proposal_history.some((proposal) => proposal.proposal_id === current)
      ) {
        return current;
      }
      return (
        snapshot.latest_proposal?.proposal_id ??
        snapshot.proposal_history.at(-1)?.proposal_id ??
        null
      );
    });
    setIsTurnInProgress(snapshot.is_turn_in_progress);
  }

  async function hydrateSession(activeSessionId: string): Promise<void> {
    try {
      const snapshot = await fetchSessionState(activeSessionId);
      applySnapshot(snapshot);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Connection lost.");
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    await submitCurrentMessage();
  }

  async function submitCurrentMessage(nextMessage?: string): Promise<void> {
    if (sessionId === null || isTurnInProgress) {
      return;
    }
    const trimmed = (nextMessage ?? inputValue).trim();
    if (!trimmed) {
      return;
    }

    try {
      setErrorMessage(null);
      await sendMessage(sessionId, trimmed);
      setInputValue("");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to send the message.");
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }
    event.preventDefault();
    void submitCurrentMessage();
  }

  function handleResizeStart(): void {
    if (window.innerWidth <= 1024 || appFrameRef.current === null) {
      return;
    }
    const rect = appFrameRef.current.getBoundingClientRect();
    resizeStateRef.current = {
      frameLeft: rect.left,
      frameWidth: rect.width,
    };

    const handlePointerMove = (event: PointerEvent): void => {
      const resizeState = resizeStateRef.current;
      if (resizeState === null) {
        return;
      }
      const nextWidth = resizeState.frameWidth - (event.clientX - resizeState.frameLeft);
      setProposalPanelWidth(clampProposalPanelWidth(nextWidth, window.innerWidth));
    };

    const handlePointerUp = (): void => {
      resizeStateRef.current = null;
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp, { once: true });
  }

  function toggleClusterSummary(summaryId: string): void {
    setExpandedClusterSummaries((current) => ({
      ...current,
      [summaryId]: !current[summaryId],
    }));
  }

  function applyStarterPrompt(prompt: string): void {
    void submitCurrentMessage(prompt);
  }

  const activeProposal =
    proposalHistory.find((proposal) => proposal.proposal_id === selectedProposalId) ??
    latestProposal;
  const hasMultipleProposals = proposalHistory.length > 1;
  const hasProposalPanelContent = hasProposal || pendingProposalTrace !== null;
  const isDesktopViewport = window.innerWidth > 1024;
  const shouldRenderProposalPanel = isProposalPanelVisible && hasProposalPanelContent;
  const shouldShowThreadWorkingIndicator =
    isTurnInProgress && !hasStreamingAssistantContent(transcript);
  const threadWorkingLabel =
    pendingProposalTrace !== null ? "Generating clustering proposal" : "Agent is working";

  return (
    <div
      ref={appFrameRef}
      className="app-frame"
      style={{
        gridTemplateColumns:
          isDesktopViewport && shouldRenderProposalPanel
            ? `minmax(0, 1fr) 20px minmax(${MIN_PROPOSAL_PANEL_WIDTH}px, ${proposalPanelWidth}px)`
            : "minmax(0, 1fr)",
      }}
    >
      <div className="bg-orb-3" aria-hidden="true" />
      <section className="chat-panel">
        {hasProposalPanelContent ? (
          <div className="chat-panel__toolbar">
            <button
              type="button"
              className="proposal-panel-toggle"
              aria-controls="proposal-panel"
              aria-label={shouldRenderProposalPanel ? "Hide proposal panel" : "Show proposal panel"}
              aria-expanded={shouldRenderProposalPanel}
              title={shouldRenderProposalPanel ? "Hide proposal panel" : "Show proposal panel"}
              onClick={() => setIsProposalPanelVisible((current) => !current)}
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              >
                <rect x="3.5" y="4.5" width="17" height="15" rx="3" />
                <path d="M13.5 4.5V19.5" strokeLinecap="round" />
                {shouldRenderProposalPanel ? (
                  <rect x="13.5" y="4.5" width="7" height="15" rx="3" fill="currentColor" opacity="0.2" stroke="none" />
                ) : (
                  <>
                    <path d="M16.5 8.25V15.75" strokeLinecap="round" opacity="0.95" />
                    <path d="M18.75 8.25V15.75" strokeLinecap="round" opacity="0.55" />
                  </>
                )}
              </svg>
            </button>
          </div>
        ) : null}

        <div className="chat-thread" aria-live="polite">
          {transcript.length === 0 ? (
            <div className="thread-empty">
              <p>Ask for a clustering proposal and the conversation will build here.</p>
              <div className="thread-empty__actions">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className="thread-empty__prompt"
                    onClick={() => applyStarterPrompt(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            transcript.filter(shouldRenderEntry).map((entry) => (
              (() => {
                const toolTrace = getToolTraceMetadata(entry);
                if (toolTrace !== null) {
                  return (
                    <article
                      key={entry.id}
                      className={`trace-row trace-row--${toolTrace.stage}`}
                    >
                      <div className="trace-row__rail" />
                      <div className="trace-row__content">
                        <p className="trace-row__title-text">{toolTrace.display_name}</p>
                        {toolTrace.details.length > 0 ? (
                          <div className="trace-row__details">
                            {toolTrace.details.map((detail) => (
                              <span key={`${entry.id}-${detail}`}>{detail}</span>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    </article>
                  );
                }

                return (
                  <article
                    key={entry.id}
                    className={[
                      "message-row",
                      `message-row--${entry.role}`,
                      `message-row--${entry.status}`,
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    {entry.role === "assistant" ? (
                      <div className="message-row__avatar" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                           <path strokeLinecap="round" strokeLinejoin="round" d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z" />
                        </svg>
                      </div>
                    ) : null}
                    <div
                      className={[
                        "message-bubble",
                        `message-bubble--${entry.role}`,
                      ]
                        .filter(Boolean)
                        .join(" ")}
                    >
                      {entry.role === "assistant" ? (
                        <div className="message-bubble__markdown">{renderMarkdown(entry.content)}</div>
                      ) : (
                        <p className="message-bubble__text">{entry.content}</p>
                      )}
                      {entry.role === "assistant" &&
                      typeof entry.cumulative_token_total === "number" ? (
                        <p className="message-bubble__note">
                          {formatTokenNote(entry.cumulative_token_total)}
                        </p>
                      ) : null}
                    </div>
                  </article>
                );
              })()
            ))
          )}
          {shouldShowThreadWorkingIndicator ? (
            <div className="thread-working" role="status" aria-live="polite">
              <span className="working-indicator" aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
              <span>{threadWorkingLabel}</span>
            </div>
          ) : null}
          <div ref={transcriptEndRef} />
        </div>

        {errorMessage ? <p className="error-banner">{errorMessage}</p> : null}

        <form className="chat-composer" onSubmit={handleSubmit}>
          <div className="chat-composer__shell">
            <textarea
              ref={composerInputRef}
              id="message-input"
              className="chat-composer__input"
              rows={3}
              placeholder="Ask for a clustering proposal, refine the embedding lens, or request a new number of clusters."
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              disabled={sessionId === null}
              readOnly={isTurnInProgress}
              aria-disabled={sessionId === null || isTurnInProgress}
            />
            <button
              className="chat-composer__button"
              type="submit"
              aria-label="Send message"
              disabled={sessionId === null || isTurnInProgress || inputValue.trim().length === 0}
            >
              <svg
                aria-hidden="true"
                className="chat-composer__button-icon"
                viewBox="0 0 24 24"
                fill="none"
              >
                <path
                  d="M5 12H18M18 12L12.75 6.75M18 12L12.75 17.25"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="1.8"
                />
              </svg>
            </button>
          </div>
        </form>
      </section>

      {isDesktopViewport && shouldRenderProposalPanel ? (
        <div
          className="proposal-resize-handle"
          aria-hidden="true"
          onPointerDown={handleResizeStart}
        />
      ) : null}

      {shouldRenderProposalPanel ? (
        <aside id="proposal-panel" className="proposal-panel">
          {activeProposal === null && pendingProposalTrace === null ? (
            <div className="proposal-empty">
              <p>No proposal yet.</p>
              <span>The next `propose_clustering` result will appear here automatically.</span>
            </div>
          ) : (
            <div className="proposal-rail">
              {pendingProposalTrace !== null ? (
                <div className="proposal-pending" role="status" aria-live="polite">
                  <div className="proposal-pending__header">
                    <ClusteringAnimation />
                    <div>
                      <p className="proposal-pending__title">Generating clustering proposal</p>
                    </div>
                  </div>
                  <p className="proposal-pending__copy">
                    The panel will populate as soon as `propose_clustering` finishes.
                  </p>
                </div>
              ) : null}

              {hasMultipleProposals ? (
                <div className="proposal-selector" role="tablist" aria-label="Proposal history">
                  {proposalHistory.map((proposal, index) => {
                    const isActive = proposal.proposal_id === activeProposal?.proposal_id;
                    return (
                      <button
                        key={proposal.proposal_id}
                        type="button"
                        role="tab"
                        aria-selected={isActive}
                        className={[
                          "proposal-selector__button",
                          isActive ? "proposal-selector__button--active" : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        onClick={() => setSelectedProposalId(proposal.proposal_id)}
                      >
                        {`Proposal ${index + 1}`}
                      </button>
                    );
                  })}
                </div>
              ) : null}

              <div className="proposal-summary">
                {activeProposal !== null && hasMultipleProposals ? (
                  <div className="proposal-summary__stats">
                    <span>{activeProposal.num_clusters} clusters</span>
                  </div>
                ) : null}
              </div>

              {activeProposal !== null ? (
                <div className="proposal-grid">
                  {activeProposal.clusters.map((cluster) => {
                    const summaryId = `${activeProposal.proposal_id}-${cluster.name}`;
                    const isExpanded = expandedClusterSummaries[summaryId] ?? false;

                    return (
                      <article className="cluster-card" key={summaryId}>
                        <div className="cluster-card__header">
                          <h4>{cluster.name}</h4>
                          <span>{cluster.size} items</span>
                        </div>
                        <p
                          id={`summary-${summaryId}`}
                          className={[
                            "cluster-card__summary",
                            !isExpanded ? "cluster-card__summary--collapsed" : "",
                          ]
                            .filter(Boolean)
                            .join(" ")}
                        >
                          {cluster.summary}
                        </p>
                        <button
                          type="button"
                          className="cluster-card__summary-toggle"
                          aria-expanded={isExpanded}
                          aria-controls={`summary-${summaryId}`}
                          onClick={() => toggleClusterSummary(summaryId)}
                        >
                          {isExpanded ? "Show less" : "Show more"}
                        </button>
                        <div className="cluster-card__points">
                          {cluster.representative_points.map((point) => (
                            <span key={point}>{point}</span>
                          ))}
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="proposal-grid proposal-grid--pending" aria-hidden="true">
                  {Array.from({ length: 3 }, (_, index) => (
                    <article className="cluster-card cluster-card--skeleton" key={`pending-${index}`}>
                      <div className="cluster-card__header">
                        <div className="skeleton-line skeleton-line--title" />
                        <div className="skeleton-line skeleton-line--stat" />
                      </div>
                      <div className="cluster-card__summary cluster-card__summary--skeleton">
                        <div className="skeleton-line skeleton-line--body" />
                        <div className="skeleton-line skeleton-line--body skeleton-line--short" />
                      </div>
                      <div className="cluster-card__points">
                        <span className="skeleton-pill" />
                        <span className="skeleton-pill skeleton-pill--short" />
                        <span className="skeleton-pill" />
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </div>
          )}
        </aside>
      ) : null}
      <p className="demo-disclaimer" role="note">
        Demo only: this is a lightweight proof of concept to demonstrate clustering behavior, not a polished product UI or project deliverable.
      </p>
    </div>
  );
}

export default App;
