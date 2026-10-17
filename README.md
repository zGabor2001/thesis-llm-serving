# vLLM Serving Experiment

Multi-tenant LLM serving stack: a vLLM backend (base model + LoRA adapters), a
scheduling gateway (FIFO / WRR / DRR), and a concurrency load generator.

## Local development

### Prerequisites

- Python 3.12+
- Docker (optional — only for the real vLLM-CPU backend)

### Quick start (zero dependencies, mock backend)

```bash
make setup   # create .venv and install dev dependencies
make demo    # start the gateway in mock mode, run a smoke test, stop it
```

### Full local stack (real vLLM-CPU backend)

```bash
make setup
make fetch-models   # download the base model into models/ (LoRAs are local artifacts)
make up-vllm        # start the vLLM CPU container and wait for health
make gateway        # terminal 1: gateway in dev mode (BACKEND_MODE=vllm)
make smoke          # terminal 2: load generator against the gateway
make down           # stop the vLLM container
```

### Mock vs real backend

Set `BACKEND_MODE` in `.env` (copy from `.env.example`) or export it in the shell.

| Mode   | Requires                          | Use for                        |
|--------|-----------------------------------|--------------------------------|
| `mock` | nothing                           | offline development, unit tests |
| `vllm` | running vLLM at `VLLM_BASE_URL`   | end-to-end experiments          |

`make gateway-mock` runs the gateway with `BACKEND_MODE=mock` explicitly.

### Configuration

Copy `.env.example` to `.env` and edit it. Shell-exported variables take
precedence over `.env`. Key variables:

- `BACKEND_MODE` — `mock` | `vllm`
- `VLLM_BASE_URL` — vLLM server URL (default `http://localhost:8000`)
- `SCHEDULER_MODE` — `fifo` | `wrr` | `drr`
- `GATEWAY_PORT` — gateway listen port (default `8080`)

Per-request timing is appended to `logs/requests.jsonl` (root-relative; override
with `LOG_DIR`).

### Testing

```bash
make test               # hermetic unit tests (no services needed)
make test-integration   # live vLLM integration tests (starts vLLM first)
```

## SLURM Cluster Deployment (GPU)

Professional deployment for a SLURM-managed GPU cluster using Apptainer (no Docker on compute nodes). vLLM and the gateway run as two decoupled jobs and discover each other via state files in `run/`.

```bash
# 1. Build container images once (vLLM pulled from Docker Hub; gateway built via Docker)
scripts/build-apptainer.sh

# 2. Start the vLLM backend (long-running GPU job)
sbatch scripts/slurm/serve-vllm.sbatch

# 3. Start the gateway (CPU job) pointed at the vLLM endpoint
SCHEDULER_MODE=drr sbatch scripts/slurm/run-gateway.sbatch

# 4. Or run the whole experiment end-to-end (vLLM + gateway + load test)
SCHEDULER_MODE=drr CONCURRENCY=16 REQUESTS=200 scripts/run-experiment.sh
```

- Endpoints are written to `run/vllm.url` and `run/gateway.url`.
- Per-request timing lands in `logs/requests.jsonl`.
- `SCHEDULER_MODE` = `fifo` | `wrr` | `drr`.
- Adjust `--partition`, `--gres`, `--account`, `--qos`, `--time` in the `.sbatch` files for your site.
