# vLLM CPU Experiment Setup

## Start

```bash
docker compose up -d --build
```

## Testing

vLLM API functionality test
```bash
pytest -v -s tests/test_vllm_api.py
```
