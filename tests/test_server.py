"""Tests for the HTTP pipeline server."""

import json
import threading
import time
from http.client import HTTPConnection

import pytest

from attractor.pipeline.server import PipelineManager, create_server


@pytest.fixture()
def server_url():
    """Start a test server on a random port and yield (host, port)."""
    server = create_server("127.0.0.1", 0)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield host, port
    server.shutdown()


def _request(host, port, method, path, body=None):
    conn = HTTPConnection(host, port, timeout=10)
    headers = {"Content-Type": "application/json"} if body else {}
    conn.request(method, path, body=json.dumps(body).encode() if body else None, headers=headers)
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    return resp.status, json.loads(data) if data else {}


SIMPLE_DOT = """
digraph T {
    start [shape=Mdiamond]
    exit  [shape=Msquare]
    a [shape=box, prompt="Hello"]
    start -> a -> exit
}
"""


def test_list_pipelines_empty(server_url):
    status, data = _request(*server_url, "GET", "/pipelines")
    assert status == 200
    assert data == []


def test_start_pipeline(server_url):
    status, data = _request(*server_url, "POST", "/pipelines", {"dot_source": SIMPLE_DOT})
    assert status == 201
    assert "id" in data
    assert data["status"] in ("running", "completed")


def test_get_pipeline_status(server_url):
    host, port = server_url
    _, create_data = _request(host, port, "POST", "/pipelines", {"dot_source": SIMPLE_DOT})
    run_id = create_data["id"]

    # Wait for completion
    for _ in range(50):
        time.sleep(0.1)
        status, data = _request(host, port, "GET", f"/pipelines/{run_id}")
        if data.get("status") != "running":
            break

    assert status == 200
    assert data["status"] == "completed"


def test_pipeline_not_found(server_url):
    status, data = _request(*server_url, "GET", "/pipelines/nonexistent-id")
    assert status == 404


def test_missing_dot_source(server_url):
    status, data = _request(*server_url, "POST", "/pipelines", {"dot_source": ""})
    assert status == 400
    assert "error" in data


def test_cancel_pipeline(server_url):
    host, port = server_url
    _, create_data = _request(host, port, "POST", "/pipelines", {"dot_source": SIMPLE_DOT})
    run_id = create_data["id"]
    # Try to cancel (may already be done)
    status, _ = _request(host, port, "POST", f"/pipelines/{run_id}/cancel")
    assert status in (200, 400)  # 400 if already completed


class TestPipelineManager:
    def test_start_and_list(self):
        mgr = PipelineManager()
        run = mgr.start_run(SIMPLE_DOT)
        assert run.id
        assert run.status in ("running", "completed")
        runs = mgr.list_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == run.id

    def test_get_nonexistent(self):
        mgr = PipelineManager()
        assert mgr.get_run("fake") is None

    def test_cancel_nonexistent(self):
        mgr = PipelineManager()
        assert mgr.cancel_run("fake") is False


def test_questions_endpoint(server_url):
    host, port = server_url
    _, create_data = _request(host, port, "POST", "/pipelines", {"dot_source": SIMPLE_DOT})
    run_id = create_data["id"]

    # Wait for completion
    for _ in range(50):
        time.sleep(0.1)
        _, data = _request(host, port, "GET", f"/pipelines/{run_id}")
        if data.get("status") != "running":
            break

    status, data = _request(host, port, "GET", f"/pipelines/{run_id}/questions")
    assert status == 200
    assert isinstance(data, list)  # Empty list since no human gates in SIMPLE_DOT


def test_questions_not_found(server_url):
    status, data = _request(*server_url, "GET", "/pipelines/nonexistent/questions")
    assert status == 404
