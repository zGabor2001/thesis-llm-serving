# vLLM CPU Experiment Setup

```bash
docker compose up -d --build
docker logs thesis-vllm-cpu
```

```bash
uvicorn gateway.proxy:app --reload --port 8080
```

## Testing

```bash
pytest -v -s tests/test_models.py
pytest -v -s tests/test_proxy.py
pytest -v -s tests/test_vllm_api.py
pytest -v -s tests/test_vllm_client.py
pytest -v -s tests/test_dispatcher.py
```
