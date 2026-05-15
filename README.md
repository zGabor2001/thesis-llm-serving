# LLM Serving Thesis Repo

## What it does
- Simulates multi-tenant LLM serving workloads
- Compares WRR vs DRR scheduling
- Models warm adapter reuse vs cold adapter switching
- Computes TTFT, P95/P99, throughput, cold starts, and Jain's fairness index
- Writes CSV results you can later turn into thesis figures

## Files
- `main.py` — entry point
- `llm_serving_mvp/config.py` — experiment config
- `llm_serving_mvp/models.py` — request and result data structures
- `llm_serving_mvp/workload.py` — synthetic workload generator
- `llm_serving_mvp/scheduler.py` — WRR and DRR
- `llm_serving_mvp/simulator.py` — service-time and residency model
- `llm_serving_mvp/metrics.py` — metrics and summaries
- `requirements.txt` — lightweight dependencies

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run default experiment
```bash
python main.py
```

## Run with custom arguments
```bash
python main.py --requests 1000 --burstiness 0.5 --whale-rate 0.12 --quantum 256
```

## Output
Results are written to `results/`:
- `request_log.csv`
- `tenant_summary.csv`
- `run_summary.json`

## Suggested first comparison
Run twice:
1. `--scheduler WRR --residency cold`
2. `--scheduler DRR --residency warm`

That gives you a simple first thesis result: DRR + warm residency should improve fairness and reduce tail latency for short interactive traffic under bursty mixed workloads.
