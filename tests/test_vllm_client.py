from gateway.models import InferenceRequest
from gateway.vllm_client import MockBackendClient, VLLMClient


def test_build_payload_minimal():
    client = VLLMClient(base_url="http://localhost:8000")

    req = InferenceRequest(
        request_id="r1",
        tenant="tenant-a",
        model="base-model",
        arrival_ts=0.0,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32,
    )

    payload = client.build_payload(req)

    assert payload["model"] == "base-model"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0.0


def test_build_payload_with_extra_metadata():
    client = VLLMClient(base_url="http://localhost:8000")

    req = InferenceRequest(
        request_id="r2",
        tenant="tenant-b",
        model="tenant-b",
        arrival_ts=0.0,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=16,
        metadata={
            "stream": False,
            "top_p": 0.9,
            "extra_body": {"seed": 123},
        },
    )

    payload = client.build_payload(req)

    assert payload["stream"] is False
    assert payload["top_p"] == 0.9
    assert payload["seed"] == 123


def test_mock_backend_health_and_models():
    client = MockBackendClient()

    assert client.health() is True
    assert client.ping() is True
    assert client.version() == "mock-backend"
    assert "base-model" in client.list_models()


def test_mock_backend_complete_returns_result():
    client = MockBackendClient(fixed_delay_s=0.0)

    req = InferenceRequest(
        request_id="r3",
        tenant="tenant-a",
        model="tenant-a",
        arrival_ts=0.0,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=24,
        prompt_text="Say hello",
    )

    result = client.complete(req)

    assert result.request_id == "r3"
    assert result.tenant == "tenant-a"
    assert result.model == "tenant-a"
    assert result.status == "completed"
    assert result.response_text is not None
    assert result.generated_tokens is not None