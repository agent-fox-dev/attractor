"""Tests for the unified LLM type system."""

from attractor.llm.types import (
    Message, Role, ContentPart, ContentKind, Usage,
    ToolDefinition, Request, Response, FinishReason,
    ModelInfo, ProviderError, AuthenticationError, RateLimitError,
    ServerError, NetworkError, ContentFilterError, InvalidRequestError,
    RetryPolicy, ResponseFormat, RateLimitInfo, ResponseWarning,
    ThinkingData,
)
from attractor.llm.high_level import GenerateResult, StepResult
from attractor.llm.catalog import get_model_info, list_models, get_latest_model


def test_message_system():
    m = Message.system("You are helpful.")
    assert m.role == Role.SYSTEM
    assert m.text == "You are helpful."


def test_message_user():
    m = Message.user("Hello")
    assert m.role == Role.USER
    assert m.text == "Hello"


def test_message_assistant():
    m = Message.assistant("Hi there")
    assert m.role == Role.ASSISTANT
    assert m.text == "Hi there"


def test_message_tool_result():
    m = Message.tool_result("call_1", "output data")
    assert m.role == Role.TOOL
    parts = [p for p in m.content if p.kind == ContentKind.TOOL_RESULT]
    assert len(parts) == 1
    assert parts[0].tool_result.content == "output data"


def test_usage_addition():
    u1 = Usage(input_tokens=10, output_tokens=5)
    u2 = Usage(input_tokens=20, output_tokens=10)
    total = u1 + u2
    assert total.input_tokens == 30
    assert total.output_tokens == 15


def test_response_text_and_tool_calls():
    r = Response(
        id="resp_1",
        model="test",
        content=[
            ContentPart(kind=ContentKind.TEXT, text="Hello"),
            ContentPart(kind=ContentKind.TOOL_CALL, tool_call={
                "id": "call_1", "name": "shell", "arguments": {"command": "ls"},
            }),
        ],
        usage=Usage(),
        finish_reason=FinishReason.TOOL_CALLS,
    )
    assert r.text == "Hello"
    assert len(r.tool_calls) == 1


def test_catalog_known_model():
    info = get_model_info("claude-opus-4-6")
    assert info is not None
    assert info.provider == "anthropic"
    assert info.supports_tools is True


def test_catalog_unknown_model():
    assert get_model_info("nonexistent-model") is None


def test_catalog_list_by_provider():
    anthropic_models = list_models("anthropic")
    assert len(anthropic_models) >= 2
    assert all(m.provider == "anthropic" for m in anthropic_models)


def test_catalog_latest_model():
    latest = get_latest_model("openai")
    assert latest is not None
    assert latest.provider == "openai"


# ---------------------------------------------------------------------------
# Error hierarchy tests
# ---------------------------------------------------------------------------


def test_error_hierarchy_base():
    err = ProviderError("test error", status_code=500, retryable=True)
    assert str(err) == "test error"
    assert err.status_code == 500
    assert err.retryable is True
    assert isinstance(err, Exception)


def test_authentication_error():
    err = AuthenticationError(provider="anthropic")
    assert err.status_code == 401
    assert err.retryable is False
    assert err.provider == "anthropic"
    assert isinstance(err, ProviderError)


def test_rate_limit_error():
    err = RateLimitError(retry_after=5.0, provider="openai")
    assert err.status_code == 429
    assert err.retryable is True
    assert err.retry_after == 5.0


def test_server_error():
    err = ServerError(status_code=503, provider="gemini")
    assert err.status_code == 503
    assert err.retryable is True


def test_network_error():
    err = NetworkError("DNS failed")
    assert err.retryable is True
    assert "DNS failed" in str(err)


def test_content_filter_error():
    err = ContentFilterError()
    assert err.retryable is False


def test_invalid_request_error():
    err = InvalidRequestError("bad model")
    assert err.status_code == 400
    assert err.retryable is False


# ---------------------------------------------------------------------------
# New types tests
# ---------------------------------------------------------------------------


def test_retry_policy_defaults():
    policy = RetryPolicy()
    assert policy.max_retries == 2
    assert policy.initial_delay == 1.0
    assert policy.jitter is True


def test_response_format():
    fmt = ResponseFormat(type="json_schema", json_schema={"type": "object"}, strict=True)
    assert fmt.type == "json_schema"
    assert fmt.strict is True


def test_rate_limit_info():
    info = RateLimitInfo(requests_remaining=100, tokens_remaining=50000)
    assert info.requests_remaining == 100


def test_response_warnings():
    r = Response(
        id="r1", model="test",
        warnings=[ResponseWarning(code="deprecated", message="old model")],
    )
    assert len(r.warnings) == 1
    assert r.warnings[0].code == "deprecated"


def test_response_rate_limit():
    r = Response(
        id="r1", model="test",
        rate_limit=RateLimitInfo(requests_remaining=5),
    )
    assert r.rate_limit is not None
    assert r.rate_limit.requests_remaining == 5


def test_response_reasoning_property():
    r = Response(
        id="r1", model="test",
        content=[
            ContentPart(kind=ContentKind.THINKING, thinking=ThinkingData(text="step 1")),
            ContentPart(kind=ContentKind.TEXT, text="answer"),
            ContentPart(kind=ContentKind.THINKING, thinking=ThinkingData(text=" step 2")),
        ],
    )
    assert r.reasoning == "step 1 step 2"
    assert r.text == "answer"


def test_response_provider_field():
    r = Response(id="r1", model="test", provider="anthropic")
    assert r.provider == "anthropic"


def test_request_top_p_and_metadata():
    req = Request(model="test", top_p=0.9, metadata={"user_id": "123"})
    assert req.top_p == 0.9
    assert req.metadata["user_id"] == "123"


# ---------------------------------------------------------------------------
# GenerateResult tests
# ---------------------------------------------------------------------------


def test_generate_result():
    resp = Response(
        id="r1", model="test",
        content=[ContentPart(kind=ContentKind.TEXT, text="hello")],
    )
    result = GenerateResult(
        response=resp,
        steps=[StepResult(response=resp)],
        total_usage=Usage(input_tokens=100, output_tokens=50),
    )
    assert result.text == "hello"
    assert result.total_usage.input_tokens == 100
    assert len(result.steps) == 1


def test_message_developer():
    msg = Message.developer("You are a helpful assistant.")
    assert msg.role == Role.DEVELOPER
    assert msg.text == "You are a helpful assistant."


def test_image_data_from_path(tmp_path):
    from attractor.llm.types import ImageData
    import base64

    # Create a tiny PNG file
    png_path = tmp_path / "test.png"
    # Minimal valid PNG (1x1 transparent)
    png_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
        0x54, 0x78, 0x9C, 0x62, 0x00, 0x00, 0x00, 0x02,
        0x00, 0x01, 0xE5, 0x27, 0xDE, 0xFC, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
        0x60, 0x82,
    ])
    png_path.write_bytes(png_bytes)

    img = ImageData.from_path(str(png_path))
    assert img.media_type == "image/png"
    assert img.data == png_bytes
    assert img.url.startswith("data:image/png;base64,")


def test_image_data_from_path_not_found():
    from attractor.llm.types import ImageData
    import pytest

    with pytest.raises(FileNotFoundError):
        ImageData.from_path("/nonexistent/image.png")


def test_stream_result_partial_text():
    from attractor.llm.high_level import StreamResult

    sr = StreamResult.__new__(StreamResult)
    sr._text_parts = ["Hello", " world"]
    sr._tool_calls = []
    sr._usage = Usage()
    sr._finish_reason = None
    sr._done = False
    assert sr.partial_text == "Hello world"


def test_stream_result_response_assembly():
    from attractor.llm.high_level import StreamResult

    sr = StreamResult.__new__(StreamResult)
    sr._text_parts = ["answer"]
    sr._tool_calls = []
    sr._usage = Usage(input_tokens=10, output_tokens=5)
    sr._finish_reason = FinishReason.STOP
    sr._done = True
    resp = sr.response()
    assert resp.text == "answer"
    assert resp.usage.input_tokens == 10


def test_openai_compatible_adapter_init():
    from attractor.llm.adapters.openai_compatible import OpenAICompatibleAdapter

    adapter = OpenAICompatibleAdapter(
        base_url="http://localhost:11434/v1",
        provider_name="ollama",
    )
    assert adapter.name == "ollama"


def test_abort_signal_and_controller():
    from attractor.llm.types import AbortSignal, AbortController, AbortError
    import pytest

    controller = AbortController()
    assert controller.signal.aborted is False
    controller.signal.check()  # should not raise

    controller.abort()
    assert controller.signal.aborted is True
    with pytest.raises(AbortError):
        controller.signal.check()


def test_abort_signal_on_request():
    from attractor.llm.types import AbortController

    controller = AbortController()
    req = Request(model="test", abort_signal=controller.signal)
    assert req.abort_signal is controller.signal
    assert req.abort_signal.aborted is False


def test_client_complete_checks_abort():
    import pytest
    import asyncio
    from attractor.llm.types import AbortController, AbortError
    from attractor.llm.client import Client

    controller = AbortController()
    controller.abort()  # pre-abort

    client = Client(providers={}, default_provider=None)
    req = Request(model="test", provider="fake", abort_signal=controller.signal)

    with pytest.raises(AbortError):
        asyncio.run(client.complete(req))
