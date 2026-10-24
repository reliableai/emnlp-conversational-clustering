import type { CreateSessionResponse, SessionSnapshot } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

function buildUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export async function createSession(): Promise<CreateSessionResponse> {
  const response = await fetch(buildUrl("/api/session"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Session creation failed with status ${response.status}.`);
  }
  return (await response.json()) as CreateSessionResponse;
}

export async function fetchSessionState(sessionId: string): Promise<SessionSnapshot> {
  const response = await fetch(buildUrl(`/api/session/${sessionId}/state`));
  if (!response.ok) {
    throw new Error(`Session refresh failed with status ${response.status}.`);
  }
  return (await response.json()) as SessionSnapshot;
}

export async function sendMessage(sessionId: string, message: string): Promise<void> {
  const response = await fetch(buildUrl(`/api/session/${sessionId}/messages`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Message submission failed with status ${response.status}.`);
  }
}

export function connectToSessionEvents(sessionId: string): EventSource {
  return new EventSource(buildUrl(`/api/session/${sessionId}/events`));
}
