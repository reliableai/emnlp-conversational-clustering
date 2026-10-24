import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { SessionSnapshot } from "./types";

const snapshot: SessionSnapshot = {
  session_id: "session-1",
  transcript: [],
  latest_proposal: null,
  proposal_history: [],
  is_turn_in_progress: false,
  token_totals: {
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
  },
};

let mockEventSource: MockEventSource | null = null;

class MockEventSource {
  url: string;
  onerror: (() => void) | null = null;
  private listeners = new Map<string, Array<(event: MessageEvent<string>) => void>>();

  constructor(url: string) {
    this.url = url;
    mockEventSource = this;
  }

  addEventListener(type: string, listener: (event: MessageEvent<string>) => void): void {
    const current = this.listeners.get(type) ?? [];
    current.push(listener);
    this.listeners.set(type, current);
  }

  close(): void {}

  emit(type: string, data: unknown): void {
    const listeners = this.listeners.get(type) ?? [];
    const event = { data: JSON.stringify(data) } as MessageEvent<string>;
    listeners.forEach((listener) => listener(event));
  }
}

describe("App", () => {
  beforeEach(() => {
    mockEventSource = null;
    HTMLElement.prototype.scrollIntoView = vi.fn();
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 1280,
    });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/session") && init?.method === "POST") {
        return new Response(JSON.stringify({ session: snapshot }), { status: 201 });
      }
      if (url.endsWith(`/api/session/${snapshot.session_id}/messages`) && init?.method === "POST") {
        return new Response(JSON.stringify({ status: "accepted" }), { status: 202 });
      }
      if (url.endsWith(`/api/session/${snapshot.session_id}/state`)) {
        return new Response(JSON.stringify(snapshot), { status: 200 });
      }
      return new Response(null, { status: 404 });
    }));
    vi.stubGlobal("EventSource", MockEventSource);
  });

  afterEach(() => {
    cleanup();
  });

  it("keeps the proposal panel hidden after bootstrapping a session without proposals", async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    });
    expect(screen.queryByText("No proposal yet.")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show proposal panel" })).not.toBeInTheDocument();
    expect(screen.queryByTestId("clustering-animation")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
  });

  it("sends a starter prompt immediately on click", async () => {
    render(<App />);

    const starterPrompt = "What kind of data is this?";
    await waitFor(() => {
      expect(screen.getByRole("button", { name: starterPrompt })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: starterPrompt }));

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        expect.stringMatching(new RegExp(`/api/session/${snapshot.session_id}/messages$`)),
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ message: starterPrompt }),
        }),
      );
    });
    expect(
      screen.getByRole("button", { name: "What are some interesting POVs in this data?" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "What is unique about this data?" }),
    ).toBeInTheDocument();
  });

  it("shows the proposal panel with loading placeholders while the first proposal is still running", async () => {
    render(<App />);

    await waitFor(() => {
      expect(mockEventSource).not.toBeNull();
    });

    mockEventSource?.emit("turn_started", { session_id: snapshot.session_id });
    mockEventSource?.emit("system_message", {
      entry: {
        id: "entry_1",
        role: "system",
        content: "Propose Clustering",
        status: "complete",
        metadata: {
          kind: "tool_trace",
          tool_name: "propose_clustering",
          display_name: "Propose Clustering",
          step_index: 1,
          stage: "running",
          details: [],
        },
      },
    });

    await waitFor(() => {
      expect(screen.getByText("Generating clustering proposal")).toBeInTheDocument();
    });
    expect(screen.getByTestId("clustering-animation")).toBeInTheDocument();
    await waitFor(() => {
      expect(
        screen.getByText("The panel will populate as soon as `propose_clustering` finishes."),
      ).toBeInTheDocument();
    });
    expect(screen.queryByText("Step 1 in progress")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hide proposal panel" })).toBeInTheDocument();
  });

  it("renders proposal results immediately when a proposal update arrives mid-turn and lets the user hide the panel", async () => {
    render(<App />);

    await waitFor(() => {
      expect(mockEventSource).not.toBeNull();
    });

    mockEventSource?.emit("turn_started", { session_id: snapshot.session_id });
    mockEventSource?.emit("proposal_updated", {
      proposal: {
        proposal_id: "proposal_1",
        embedding_instruction: "support theme",
        num_clusters: 3,
        clusters: [
          {
            name: "Access issues",
            size: 6,
            summary: "Authentication and login work.",
            representative_points: ["VPN issue", "SSO failure"],
          },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByText("Access issues")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("clustering-animation")).not.toBeInTheDocument();
    expect(screen.queryByText("proposal_1")).not.toBeInTheDocument();
    expect(screen.queryByText("3 clusters")).not.toBeInTheDocument();
    expect(screen.queryByRole("tablist", { name: "Proposal history" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hide proposal panel" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Hide proposal panel" }));
    expect(screen.queryByText("Access issues")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show proposal panel" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show proposal panel" }));
    expect(screen.getByText("Access issues")).toBeInTheDocument();
  });

  it("shows proposal history controls only after a second proposal arrives", async () => {
    render(<App />);

    await waitFor(() => {
      expect(mockEventSource).not.toBeNull();
    });

    mockEventSource?.emit("proposal_updated", {
      proposal: {
        proposal_id: "proposal_1",
        embedding_instruction: "support theme",
        num_clusters: 3,
        clusters: [
          {
            name: "Access issues",
            size: 6,
            summary: "Authentication and login work.",
            representative_points: ["VPN issue", "SSO failure"],
          },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByText("Access issues")).toBeInTheDocument();
    });
    expect(screen.queryByRole("tablist", { name: "Proposal history" })).not.toBeInTheDocument();
    expect(screen.queryByText("proposal_1")).not.toBeInTheDocument();
    expect(screen.queryByText("3 clusters")).not.toBeInTheDocument();

    mockEventSource?.emit("proposal_updated", {
      proposal: {
        proposal_id: "proposal_2",
        embedding_instruction: "platform",
        num_clusters: 5,
        clusters: [
          {
            name: "Web issues",
            size: 8,
            summary: "Browser-specific regressions and broken workflows.",
            representative_points: ["Safari checkout failure", "Chrome search lag"],
          },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByRole("tablist", { name: "Proposal history" })).toBeInTheDocument();
    });
    expect(screen.queryByText("proposal_2")).not.toBeInTheDocument();
    expect(screen.getByText("5 clusters")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Proposal 1" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Proposal 2" })).toHaveAttribute("aria-selected", "true");
  });
});
