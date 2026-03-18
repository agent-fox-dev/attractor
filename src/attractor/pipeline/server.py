"""HTTP server mode per Section 9.5 of the Attractor spec.

Provides a lightweight REST + SSE server for running and monitoring
pipelines remotely. Uses only the Python standard library (no
framework dependency).

Endpoints:
    POST   /pipelines              — Start a new pipeline run
    GET    /pipelines              — List running/completed pipelines
    GET    /pipelines/{id}         — Get pipeline status
    GET    /pipelines/{id}/events  — SSE event stream
    POST   /pipelines/{id}/cancel  — Cancel a running pipeline
    GET    /pipelines/{id}/context — Get pipeline context snapshot
    POST   /pipelines/{id}/questions/{qid} — Answer a human gate question
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse, parse_qs

from .engine import PipelineConfig, PipelineRunner, run
from .events import EventEmitter, PipelineEvent, PipelineEventKind
from .graph import Outcome, StageStatus
from .interviewer import (
    Answer,
    AnswerValue,
    Interviewer,
    Question,
    QueueInterviewer,
)
from .parser import parse_dot
from .transforms import DEFAULT_TRANSFORMS, prepare_pipeline
from .validation import Severity


# ---------------------------------------------------------------------------
# Pipeline run state
# ---------------------------------------------------------------------------


@dataclass
class PipelineRun:
    """Tracks the state of a single pipeline execution."""

    id: str
    name: str = ""
    status: str = "running"
    outcome: Outcome | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    event_queues: list[queue.Queue] = field(default_factory=list)
    thread: threading.Thread | None = None
    interviewer: QueueInterviewer | None = None
    pending_questions: dict[str, Question] = field(default_factory=dict)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    dot_source: str = ""
    checkpoint_data: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Pipeline Manager
# ---------------------------------------------------------------------------


class PipelineManager:
    """Manages multiple concurrent pipeline runs."""

    def __init__(self) -> None:
        self._runs: dict[str, PipelineRun] = {}
        self._lock = threading.Lock()

    def start_run(self, dot_source: str, config: PipelineConfig | None = None) -> PipelineRun:
        run_id = str(uuid.uuid4())
        interviewer = QueueInterviewer()
        pipeline_run = PipelineRun(
            id=run_id,
            interviewer=interviewer,
        )

        with self._lock:
            self._runs[run_id] = pipeline_run

        def _event_callback(event: PipelineEvent) -> None:
            event_dict = {
                "kind": event.kind.value,
                "timestamp": time.time(),
                "data": event.data,
            }
            pipeline_run.events.append(event_dict)
            # Push to all SSE subscribers
            for q in pipeline_run.event_queues:
                try:
                    q.put_nowait(event_dict)
                except queue.Full:
                    pass

        def _run_thread() -> None:
            try:
                runner = PipelineRunner(
                    interviewer=interviewer,
                    event_callback=_event_callback,
                )
                cfg = config or PipelineConfig()
                outcome = runner.run(dot_source, config=cfg)
                pipeline_run.outcome = outcome
                pipeline_run.status = "completed" if outcome.status == StageStatus.SUCCESS else "failed"
            except Exception as exc:
                pipeline_run.status = "failed"
                pipeline_run.error = str(exc)

        thread = threading.Thread(target=_run_thread, daemon=True)
        pipeline_run.thread = thread
        pipeline_run.name = f"pipeline-{run_id[:8]}"
        pipeline_run.dot_source = dot_source
        thread.start()

        return pipeline_run

    def get_run(self, run_id: str) -> PipelineRun | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id,
                "name": r.name,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in self._runs.values()
        ]

    def cancel_run(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.status != "running":
            return False
        run.status = "cancelled"
        return True


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------


class PipelineHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for the pipeline REST API."""

    manager: PipelineManager  # Set by the server factory

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _parse_path(self) -> tuple[str, list[str]]:
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        return parsed.path, parts

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path, parts = self._parse_path()

        # GET /pipelines
        if parts == ["pipelines"]:
            self._send_json(self.manager.list_runs())
            return

        # GET /pipelines/{id}
        if len(parts) == 2 and parts[0] == "pipelines":
            run = self.manager.get_run(parts[1])
            if run is None:
                self._send_error("Pipeline not found", 404)
                return
            self._send_json({
                "id": run.id,
                "name": run.name,
                "status": run.status,
                "created_at": run.created_at,
                "events_count": len(run.events),
                "error": run.error,
                "outcome": run.outcome.model_dump() if run.outcome else None,
            })
            return

        # GET /pipelines/{id}/events — SSE stream
        if len(parts) == 3 and parts[0] == "pipelines" and parts[2] == "events":
            run = self.manager.get_run(parts[1])
            if run is None:
                self._send_error("Pipeline not found", 404)
                return
            self._handle_sse(run)
            return

        # GET /pipelines/{id}/context
        if len(parts) == 3 and parts[0] == "pipelines" and parts[2] == "context":
            run = self.manager.get_run(parts[1])
            if run is None:
                self._send_error("Pipeline not found", 404)
                return
            self._send_json(run.context_snapshot)
            return

        # GET /pipelines/{id}/checkpoint
        if len(parts) == 3 and parts[0] == "pipelines" and parts[2] == "checkpoint":
            run = self.manager.get_run(parts[1])
            if run is None:
                self._send_error("Pipeline not found", 404)
                return
            checkpoint = getattr(run, "checkpoint_data", None)
            if checkpoint is None:
                self._send_json({"status": "no_checkpoint"})
            else:
                self._send_json(checkpoint)
            return

        # GET /pipelines/{id}/questions
        if len(parts) == 3 and parts[0] == "pipelines" and parts[2] == "questions":
            run = self.manager.get_run(parts[1])
            if run is None:
                self._send_error("Pipeline not found", 404)
                return
            questions = [
                {"id": qid, "text": str(q)}
                for qid, q in run.pending_questions.items()
            ]
            self._send_json(questions)
            return

        # GET /pipelines/{id}/graph
        if len(parts) == 3 and parts[0] == "pipelines" and parts[2] == "graph":
            run = self.manager.get_run(parts[1])
            if run is None:
                self._send_error("Pipeline not found", 404)
                return
            dot_source = getattr(run, "dot_source", "")
            if not dot_source:
                self._send_json({"format": "dot", "content": ""})
                return
            # Try to render SVG via Graphviz
            try:
                proc = subprocess.run(
                    ["dot", "-Tsvg"],
                    input=dot_source,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and "<svg" in proc.stdout:
                    body = proc.stdout.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "image/svg+xml")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            # Fallback: return raw DOT
            self._send_json({"format": "dot", "content": dot_source})
            return

        self._send_error("Not found", 404)

    def do_POST(self) -> None:
        path, parts = self._parse_path()

        # POST /pipelines — start a new run
        if parts == ["pipelines"]:
            body = self._read_body()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_error("Invalid JSON")
                return

            dot_source = data.get("dot_source", "")
            if not dot_source:
                self._send_error("Missing 'dot_source' field")
                return

            config = PipelineConfig()
            if "logs_root" in data:
                config.logs_root = data["logs_root"]
            if "context_values" in data:
                config.context_values = data["context_values"]

            run = self.manager.start_run(dot_source, config)
            self._send_json({"id": run.id, "status": run.status}, 201)
            return

        # POST /pipelines/{id}/cancel
        if len(parts) == 3 and parts[0] == "pipelines" and parts[2] == "cancel":
            success = self.manager.cancel_run(parts[1])
            if success:
                self._send_json({"status": "cancelled"})
            else:
                self._send_error("Cannot cancel pipeline", 400)
            return

        # POST /pipelines/{id}/questions/{qid} — answer a human gate
        if len(parts) == 4 and parts[0] == "pipelines" and parts[2] == "questions":
            run = self.manager.get_run(parts[1])
            if run is None:
                self._send_error("Pipeline not found", 404)
                return
            body = self._read_body()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_error("Invalid JSON")
                return

            if run.interviewer:
                answer = Answer(
                    value=AnswerValue(data.get("value", "yes")),
                    text=data.get("text", ""),
                    selected_option=data.get("selected_option", ""),
                )
                run.interviewer.answer_queue.put(answer)
                self._send_json({"status": "answered"})
            else:
                self._send_error("No interviewer configured", 400)
            return

        self._send_error("Not found", 404)

    def _handle_sse(self, run: PipelineRun) -> None:
        """Stream pipeline events as SSE."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send historical events first
        for event in run.events:
            self._write_sse_event(event)

        # If already finished, close
        if run.status != "running":
            self._write_sse_event({"kind": "stream_end", "data": {"status": run.status}})
            return

        # Subscribe to new events
        event_queue: queue.Queue = queue.Queue(maxsize=1000)
        run.event_queues.append(event_queue)

        try:
            while run.status == "running":
                try:
                    event = event_queue.get(timeout=1.0)
                    self._write_sse_event(event)
                except queue.Empty:
                    # Send keepalive
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
        finally:
            if event_queue in run.event_queues:
                run.event_queues.remove(event_queue)

        self._write_sse_event({"kind": "stream_end", "data": {"status": run.status}})

    def _write_sse_event(self, event: dict[str, Any]) -> None:
        try:
            data = json.dumps(event, default=str)
            self.wfile.write(f"event: {event.get('kind', 'message')}\n".encode())
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server(host: str = "0.0.0.0", port: int = 8080) -> HTTPServer:
    """Create and return an HTTP server for the pipeline API."""
    manager = PipelineManager()
    PipelineHTTPHandler.manager = manager

    server = HTTPServer((host, port), PipelineHTTPHandler)
    return server


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the pipeline HTTP server (blocking)."""
    server = create_server(host, port)
    print(f"Attractor pipeline server listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
