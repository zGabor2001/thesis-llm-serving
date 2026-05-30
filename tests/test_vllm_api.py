import os
import time

import pytest
import requests

BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
TIMEOUT = int(os.getenv("VLLM_TEST_TIMEOUT", "120"))
EXPECTED_MODELS = [
    m.strip()
    for m in os.getenv(
        "VLLM_EXPECTED_MODELS",
        "base-model,tenant-a,tenant-b,tenant-c",
    ).split(",")
    if m.strip()
]
TEST_PROMPT = os.getenv("VLLM_TEST_PROMPT", "Introduce yourself in one short sentence.")


@pytest.fixture(scope="session")
def session():
    s = requests.Session()
    yield s
    s.close()


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL.rstrip("/")


@pytest.fixture(scope="session")
def models_response(session, base_url):
    return session.get(f"{base_url}/v1/models", timeout=TIMEOUT)


@pytest.fixture(scope="session")
def discovered_models(models_response):
    models_response.raise_for_status()
    body = models_response.json()
    return [item["id"] for item in body.get("data", [])]


def test_health(session, base_url):
    r = session.get(f"{base_url}/health", timeout=TIMEOUT)
    assert r.status_code == 200, r.text


def test_ping(session, base_url):
    r = session.get(f"{base_url}/ping", timeout=TIMEOUT)
    assert r.status_code == 200, r.text


def test_version(session, base_url):
    r = session.get(f"{base_url}/version", timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    assert r.text.strip()


def test_models_endpoint_ok(models_response):
    assert models_response.status_code == 200, models_response.text


def test_expected_models_present(discovered_models):
    missing = [m for m in EXPECTED_MODELS if m not in discovered_models]
    assert not missing, f"Missing models: {missing}; discovered={discovered_models}"


@pytest.mark.parametrize("model_name", EXPECTED_MODELS)
def test_chat_completion_per_model(session, base_url, model_name):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 48,
        "temperature": 0.0,
    }
    start = time.time()
    r = session.post(f"{base_url}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    latency = time.time() - start

    assert r.status_code == 200, f"status={r.status_code}, body={r.text}"

    body = r.json()
    assert "choices" in body and body["choices"], f"Unexpected response: {body}"

    text = body["choices"][0]["message"]["content"]
    assert isinstance(text, str) and text.strip(), f"Empty response text for {model_name}"
    assert latency < TIMEOUT, f"Request for {model_name} exceeded timeout window"


def test_invalid_model_returns_error(session, base_url):
    payload = {
        "model": "does-not-exist",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    r = session.post(f"{base_url}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    assert r.status_code >= 400, f"Expected error status, got {r.status_code} with body={r.text}"


@pytest.mark.parametrize("model_name", EXPECTED_MODELS)
def test_model_response_summary(session, base_url, model_name, capsys):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 48,
        "temperature": 0.0,
    }
    r = session.post(f"{base_url}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    assert r.status_code == 200, f"status={r.status_code}, body={r.text}"

    body = r.json()
    text = body["choices"][0]["message"]["content"].strip()

    print(f"MODEL={model_name}\\nRESPONSE={text}\\n")
    captured = capsys.readouterr()
    assert f"MODEL={model_name}" in captured.out