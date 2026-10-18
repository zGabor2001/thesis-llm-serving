# vLLM Serving Experiment

Multi-tenant LLM serving stack: a vLLM backend (base model + LoRA adapters), a
scheduling gateway (FIFO / WRR / DRR), a concurrency load generator, and a log
analyzer.

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

The LoRA adapters (`models/loras/`) are fixed input artifacts — see
`models/loras/README.md` for their provenance.

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
- `TENANT_WEIGHTS` — WRR weights, e.g. `tenant-a:2,tenant-b:1` (optional)
- `TENANT_QUANTA` — DRR token quanta, e.g. `tenant-a:16,tenant-b:16` (optional)
- `MAX_CONCURRENCY` — gateway worker pool size (default `32`)
- `TOKENIZER_PATH` — tokenizer.json path for DRR token counting (optional; falls back to a heuristic)
- `LORA_TENANTS` — tenants mapped to a LoRA adapter of the same name (default `tenant-a,tenant-b,tenant-c`)
- `GATEWAY_PORT` — gateway listen port (default `8080`)

Per-request timing is appended to `logs/requests.jsonl` (root-relative; override
with `LOG_DIR`).

### Testing

```bash
make test               # hermetic unit tests (no services needed)
make test-integration   # live vLLM integration tests (starts vLLM first)
```

### Running the scheduler comparison

```bash
# terminal 1 — run the gateway with a chosen scheduler (mock backend)
SCHEDULER_MODE=fifo make gateway-mock   # or wrr / drr

# terminal 2 — mixed-tenant load against it
make smoke-multi
```

For a self-contained single-shot mixed-tenant smoke test (starts and stops its
own mock gateway), run `make demo-multi`.

To compare all three schedulers from one log, run each scheduler, send the same
mixed workload, then analyze the combined log:

```bash
python scripts/analyze.py --log logs/requests.jsonl
```

`analyze.py` prints per-scheduler and per-tenant latency/TTFT/throughput and
Jain's fairness index, and writes `logs/analysis.json`.

### Open-loop (trace-driven) experiments

Generate Taha-style arrival traces and replay them open-loop (the same protocol
as the baseline thesis `thesis.pdf`):

```bash
python scripts/gen-trace.py --type sparse_tail --requests 200
python scripts/loadgen.py --url http://localhost:8080 --trace traces/sparse_tail.jsonl
python scripts/analyze.py
```

Trace types: `sparse_tail`, `bursty_tail`, `popularity_shift`, `balanced_mix`.
`analyze.py` additionally reports cold/warm TTFT and adapter switch counts for
comparison with the baseline.

## SLURM Cluster Deployment (GPU)

Professional deployment for a SLURM-managed GPU cluster using Apptainer (no
Docker on compute nodes). vLLM and the gateway run as decoupled jobs and
discover each other via state files in `run/`.

Build the images **once, on a compute node** (interactive `srun` session, not
the login node):

```bash
scripts/build-apptainer.sh
# vLLM .sif is pulled from Docker Hub; the gateway is built from gateway/Dockerfile
# (or scripts/slurm/gateway.def if Docker is unavailable).
```

Run the experiment:

```bash
# 1. vLLM backend (long-running GPU job)
sbatch scripts/slurm/serve-vllm.sbatch

# 2. Gateway (CPU job) pointed at the vLLM endpoint
SCHEDULER_MODE=drr sbatch scripts/slurm/run-gateway.sbatch

# 3. Load test (CPU job — never on the login node)
CONCURRENCY=16 REQUESTS=200 sbatch scripts/slurm/run-loadgen.sbatch

# ...or all three end-to-end from the login node (submission/polling only):
SCHEDULER_MODE=drr CONCURRENCY=16 REQUESTS=200 scripts/run-experiment.sh
```

- Endpoints are written to `run/vllm.url` and `run/gateway.url`.
- Per-request timing lands in `logs/requests.jsonl`; loadgen results in `logs/loadgen.jsonl`.
- `SCHEDULER_MODE` = `fifo` | `wrr` | `drr`.
- `TENANT_WEIGHTS`, `TENANT_QUANTA`, `LORA_TENANTS`, `MAX_CONCURRENCY`, `TOKENIZER_PATH` are passed through to the gateway.
- Adjust `--partition`, `--gres`, `--account`, `--qos`, `--time` in the `.sbatch` files for your site.
