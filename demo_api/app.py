"""FastAPI application factory that exposes a streaming bridge for the demo frontend."""

from __future__ import annotations

import queue
from collections.abc import Iterator
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import DemoSettings, load_demo_settings
from .schemas import CreateSessionResponseModel, SendMessageRequestModel, SessionSnapshotModel
from .sessions import (
    DemoSessionStore,
    SessionNotFoundError,
    TurnInProgressError,
    format_sse_event,
)


def create_app(
    settings: DemoSettings | None = None,
    *,
    session_store: DemoSessionStore | None = None,
) -> FastAPI:
    resolved_settings = settings or load_demo_settings()
    store = session_store or DemoSessionStore(resolved_settings)

    app = FastAPI(title="Conversational Clustering Demo API")
    app.add_middleware(
        cast(Any, CORSMiddleware),
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_session_snapshot(session_id: str) -> SessionSnapshotModel:
        try:
            return store.get_session(session_id).snapshot()
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
            ) from exc

    def event_stream(session_id: str) -> Iterator[str]:
        try:
            session = store.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
            ) from exc
        subscriber_id, subscriber_queue = session.add_subscriber()
        try:
            while True:
                try:
                    event = subscriber_queue.get(timeout=15.0)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    break
                yield format_sse_event(event)
        finally:
            session.remove_subscriber(subscriber_id)

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/api/session",
        response_model=CreateSessionResponseModel,
        status_code=status.HTTP_201_CREATED,
    )
    def create_session() -> CreateSessionResponseModel:
        session = store.create_session()
        return CreateSessionResponseModel(session=session.snapshot())

    @app.get("/api/session/{session_id}/state", response_model=SessionSnapshotModel)
    def get_session_state(session_id: str) -> SessionSnapshotModel:
        return get_session_snapshot(session_id)

    @app.get("/api/session/{session_id}/events")
    def stream_session_events(session_id: str) -> StreamingResponse:
        return StreamingResponse(
            event_stream(session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/session/{session_id}/messages", status_code=status.HTTP_202_ACCEPTED)
    def submit_message(session_id: str, request: SendMessageRequestModel) -> dict[str, str]:
        try:
            store.submit_user_message(session_id, request.message)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
            ) from exc
        except TurnInProgressError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return {"status": "accepted"}

    @app.delete("/api/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str) -> Response:
        store.delete_session(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
