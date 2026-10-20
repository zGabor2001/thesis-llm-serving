# vLLM Serving Experiment

Multi-tenant LLM serving stack: a vLLM backend (base model + LoRA adapters), a
scheduling gateway (`fifo` / `wrr` / `drr`), a load generator (closed-loop and
open-loop trace replay), and a log analyzer.

The gateway schedules requests across tenants and measures latency, TTFT
(cold/warm), throughput, per-tenant fairness (Jain's index), and adapter-switch
statistics — the same protocol as the baseline thesis (`thesis.pdf`).

## Project layout

```
gateway/          FastAPI proxy, dispatchers (FIFO/WRR/DRR), tokenizer, residency tracker
scripts/          loadgen.py, analyze.py, gen-trace.py, fetch-models.sh, build-apptainer.sh
scripts/slurm/    sbatch jobs, gateway.def, common.sh
tests/            unit + integration tests
models/           base model + LoRA adapters (gitignored; see models/loras/README.md)
traces/           generated arrival traces
logs/             per-request log + loadgen/analysis results (gitignored)
run/              cluster state files (vllm.url, gateway.url, …)
```

## Local development

### Prerequisites

- Python 3.12+
- Docker (optional — only for the real vLLM-CPU backend)

### Quick start (zero dependencies, mock backend)

```bash
make setup      # create .venv and install dev dependencies
make demo       # start the gateway in mock mode, run a smoke test, stop it
make demo-multi # same, with a mixed-tenant workload
```

### Full local stack (real vLLM-CPU backend)

```bash
make setup
make fetch-models   # download the base model into models/
make up-vllm        # start the vLLM CPU container and wait for health
make gateway        # terminal 1: gateway in dev mode (BACKEND_MODE=vllm)
make smoke          # terminal 2: load generator against the gateway
make down           # stop the vLLM container
```

The LoRA adapters (`models/loras/`) are fixed input artifacts — see
`models/loras/README.md` for their provenance.

### Makefile targets

| Target | Description |
|---|---|
| `setup` | create `.venv` + install dev dependencies |
| `fetch-models` | download the base model |
| `up-vllm` | start the vLLM-CPU container, wait for health |
| `gateway` | run the gateway (`BACKEND_MODE=vllm`, hot reload) |
| `gateway-mock` | run the gateway (`BACKEND_MODE=mock`) |
| `smoke` | single-tenant load test against a running gateway |
| `smoke-multi` | mixed-tenant load test against a running gateway |
| `demo` / `demo-multi` | self-contained mock gateway + load test |
| `test` | hermetic unit tests |
| `test-integration` | live vLLM tests (starts vLLM first) |
| `down` | stop the vLLM container |

### Mock vs real backend

Set `BACKEND_MODE` in `.env` (copy from `.env.example`) or export it in the shell.

| Mode | Requires | Use for |
|---|---|---|
| `mock` | nothing | offline development, unit tests |
| `vllm` | running vLLM at `VLLM_BASE_URL` | end-to-end experiments |

### Configuration

Copy `.env.example` to `.env` and edit it. Shell-exported variables take
precedence over `.env`.

| Variable | Meaning |
|---|---|
| `BACKEND_MODE` | `mock` \| `vllm` |
| `VLLM_BASE_URL` | vLLM server URL (default `http://localhost:8000`) |
| `DEFAULT_MODEL` | model used when none is specified (default `base-model`) |
| `SCHEDULER_MODE` | `fifo` \| `wrr` \| `drr` |
| `TENANT_WEIGHTS` | WRR weights, e.g. `tenant-a:2,tenant-b:1` (optional) |
| `TENANT_QUANTA` | DRR token quanta, e.g. `tenant-a:16,tenant-b:16` (optional) |
| `LORA_TENANTS` | tenants mapped to a LoRA adapter of the same name (default `tenant-a,tenant-b,tenant-c`) |
| `MAX_CONCURRENCY` | gateway worker-pool size (default `32`) |
| `TOKENIZER_PATH` | tokenizer.json path for DRR token counting (optional; falls back to a heuristic) |
| `GATEWAY_PORT` | gateway listen port (default `8080`) |
| `REQUEST_TIMEOUT_S` | backend request timeout (default `60`) |
| `LOG_DIR` | override the per-request log directory |

Per-request timing is appended to `logs/requests.jsonl` (root-relative).

## Testing

```bash
make test               # hermetic unit tests (no services needed)
make test-integration   # live vLLM integration tests (starts vLLM first)
```

`pytest` runs hermetic tests by default; live-vLLM tests are marked
`integration` and are deselected unless run with `-m integration`.

## Running experiments locally

### Scheduler comparison

```bash
# terminal 1 — gateway with a chosen scheduler (mock backend)
SCHEDULER_MODE=fifo make gateway-mock   # or wrr / drr

# terminal 2 — mixed-tenant load against it
make smoke-multi
```

Run each scheduler, send the same workload, then analyze the combined log.

### Multi-tenant load (closed-loop)

```bash
python scripts/loadgen.py --url http://localhost:8080 \
  --requests 100 --concurrency 8 \
  --tenants tenant-a:2,tenant-b:1,tenant-c:1 --variable-prompt
```

### Open-loop (trace-driven) experiments

Generate Taha-style arrival traces and replay them open-loop (same protocol as
`thesis.pdf`):

```bash
python scripts/gen-trace.py --type sparse_tail --requests 200
python scripts/loadgen.py --url http://localhost:8080 --trace traces/sparse_tail.jsonl
```

Trace types: `sparse_tail`, `bursty_tail`, `popularity_shift`, `balanced_mix`.
Use `--interarrival-scale` to shrink inter-arrival times for quick local runs:

```bash
python scripts/gen-trace.py --type bursty_tail --requests 60 --interarrival-scale 0.05
```

### Analysis

```bash
python scripts/analyze.py --log logs/requests.jsonl
```

Reports per-scheduler and per-tenant:
- latency / TTFT (p50/p95/p99), throughput (req/s and tokens/s)
- Jain's fairness index (generated tokens and tokens/s)
- cold/warm TTFT split, cold count, switch count, switch gap
- cold TTFT including model load time (`--load-time-s` or `run/vllm.load_time_s`)

Flags: `--out` (default `logs/analysis.json`), `--json`, `--no-write`.

## SLURM Cluster Deployment (GPU)

Deployment for a SLURM-managed GPU cluster using Apptainer (no Docker on compute
nodes). vLLM, the gateway, and the load test run as **decoupled batch jobs** and
discover each other via state files in `run/`. No interactive sessions and no
login-node work are required.

### 0. Cluster access (dataLAB)

One-time setup:

1. Log in once via the dataLAB SSO to populate your user profile.
2. Generate an SSH key and upload the public key to your dataLAB profile:

   ```bash
   ssh-keygen -t ed25519 -C "user@tuwien.ac.at" -f ~/.ssh/id_datalab
   ```

3. Add to `~/.ssh/config`:

   ```text
   Host datalab
       Hostname cluster.datalab.tuwien.ac.at
       User USERNAME
       AddKeysToAgent yes
       IdentityFile ~/.ssh/id_datalab
   ```

4. Connect: `ssh datalab`

Check partitions and your jobs:

```bash
sinfo                # partitions / GPU types
squeue -u $USER      # your queued/running jobs
```

(`USERNAME` and the email are placeholders — replace with your TU Wien identity.)

Stage the repo (code + models, nothing else):

```bash
scripts/upload-to-cluster.sh          # from your laptop
scripts/upload-to-cluster.sh --dry-run  # preview what would be uploaded
```

### 1. Build the container images (batch job)

```bash
sbatch scripts/slurm/build-images.sbatch
# vLLM .sif is pulled from Docker Hub; the gateway is built from gateway/Dockerfile
# (or scripts/slurm/gateway.def when Docker is unavailable).
```

### 2. Run the experiment (all batch jobs)

```bash
# vLLM backend (long-running GPU job)
sbatch scripts/slurm/serve-vllm.sbatch

# Gateway (CPU job) pointed at the vLLM endpoint
SCHEDULER_MODE=drr sbatch scripts/slurm/run-gateway.sbatch

# Load test (CPU job — never on the login node)
CONCURRENCY=16 REQUESTS=200 sbatch scripts/slurm/run-loadgen.sbatch

# ...or all three end-to-end (submission/polling only, no compute on login node):
SCHEDULER_MODE=drr CONCURRENCY=16 REQUESTS=200 scripts/run-experiment.sh

# ...or a full scheduler × concurrency sweep:
SCHEDULERS="fifo,wrr,drr" CONCURRENCIES="1,4,8,16" REQUESTS=200 scripts/run-sweep.sh
```

### State files & results

- Endpoints: `run/vllm.url`, `run/gateway.url`.
- Cold model-load time: `run/vllm.load_time_s`.
- Per-request timing: `logs/requests.jsonl`; loadgen results: `logs/loadgen.jsonl`.

### Cluster etiquette

- Builds, load tests, and calculations run as **batch jobs** (never on the login node).
- `SCHEDULER_MODE`, `TENANT_WEIGHTS`, `TENANT_QUANTA`, `LORA_TENANTS`,
  `MAX_CONCURRENCY`, `TOKENIZER_PATH` are passed through to the gateway.
- Partitions: `GPU-a40` for the vLLM job; `CPU-amd` for the gateway/loadgen/build jobs.
- Adjust `--account`, `--qos`, `--time` for your site.
