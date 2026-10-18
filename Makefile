SHELL := /bin/bash
PY := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn

GATEWAY_PORT ?= 8080
VLLM_PORT ?= 8000

.PHONY: help setup fetch-models up-vllm gateway gateway-mock smoke smoke-multi demo demo-multi test test-integration down

help: ## Show available targets
	@echo "make setup           Create .venv and install dev dependencies"
	@echo "make fetch-models    Download the base model into models/"
	@echo "make up-vllm         Start the vLLM CPU backend via docker compose"
	@echo "make gateway         Run the gateway in dev mode (BACKEND_MODE=vllm)"
	@echo "make gateway-mock    Run the gateway against the mock backend"
	@echo "make smoke           Run loadgen against a running gateway"
	@echo "make smoke-multi     Run a mixed-tenant load test against the gateway"
	@echo "make demo            Zero-dependency loop: mock gateway + smoke test"
	@echo "make demo-multi      Zero-dependency loop: mixed-tenant smoke test"
	@echo "make test            Run hermetic unit tests (skips integration)"
	@echo "make test-integration Run live vLLM integration tests"
	@echo "make down            Stop the vLLM container"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

fetch-models:
	bash scripts/fetch-models.sh

up-vllm:
	docker compose up -d --build
	@echo "Waiting for vLLM health at :$(VLLM_PORT) ..."
	@for i in $$(seq 1 120); do \
		curl -fsS http://localhost:$(VLLM_PORT)/health >/dev/null 2>&1 && break; \
		sleep 2; \
	done
	@curl -fsS http://localhost:$(VLLM_PORT)/health >/dev/null 2>&1 \
		&& echo "vLLM ready" || { echo "vLLM did not become ready"; exit 1; }

gateway:
	BACKEND_MODE=vllm VLLM_BASE_URL=http://localhost:$(VLLM_PORT) \
		$(UVICORN) gateway.proxy:app --reload --port $(GATEWAY_PORT)

gateway-mock:
	BACKEND_MODE=mock $(UVICORN) gateway.proxy:app --reload --port $(GATEWAY_PORT)

smoke:
	$(PY) scripts/loadgen.py --url http://localhost:$(GATEWAY_PORT) \
		--requests 50 --concurrency 4 --tenant tenant-a

smoke-multi:
	$(PY) scripts/loadgen.py --url http://localhost:$(GATEWAY_PORT) \
		--requests 90 --concurrency 9 \
		--tenants tenant-a:2,tenant-b:1,tenant-c:1 --variable-prompt

demo:
	BACKEND_MODE=mock $(UVICORN) gateway.proxy:app --port $(GATEWAY_PORT) & \
	GW_PID=$$!; \
	for i in $$(seq 1 30); do \
		curl -fsS http://localhost:$(GATEWAY_PORT)/health >/dev/null 2>&1 && break; \
		sleep 1; \
	done; \
	$(PY) scripts/loadgen.py --url http://localhost:$(GATEWAY_PORT) \
		--requests 50 --concurrency 4 --tenant tenant-a; \
	kill $$GW_PID 2>/dev/null || true

demo-multi:
	BACKEND_MODE=mock $(UVICORN) gateway.proxy:app --port $(GATEWAY_PORT) & \
	GW_PID=$$!; \
	for i in $$(seq 1 30); do \
		curl -fsS http://localhost:$(GATEWAY_PORT)/health >/dev/null 2>&1 && break; \
		sleep 1; \
	done; \
	$(PY) scripts/loadgen.py --url http://localhost:$(GATEWAY_PORT) \
		--requests 90 --concurrency 9 \
		--tenants tenant-a:2,tenant-b:1,tenant-c:1 --variable-prompt; \
	kill $$GW_PID 2>/dev/null || true

test:
	$(PY) -m pytest -v

test-integration: up-vllm
	$(PY) -m pytest -v -m integration

down:
	docker compose down
