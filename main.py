from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_serving.config import ExperimentConfig
from llm_serving.workload import generate_workload
from llm_serving.simulator import run_simulation
from llm_serving.metrics import build_run_summary, build_tenant_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM serving thesis MVP")
    p.add_argument("--scheduler", choices=["WRR", "DRR"], default="DRR")
    p.add_argument("--residency", choices=["warm", "cold"], default="warm")
    p.add_argument("--requests", type=int, default=600)
    p.add_argument("--burstiness", type=float, default=0.45)
    p.add_argument("--whale-rate", type=float, default=0.12)
    p.add_argument("--quantum", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        scheduler=args.scheduler,
        residency=args.residency,
        request_count=args.requests,
        burstiness=args.burstiness,
        whale_rate=args.whale_rate,
        quantum=args.quantum,
        seed=args.seed,
    )

    workload = generate_workload(cfg)
    sim_result = run_simulation(cfg, workload)
    run_summary = build_run_summary(cfg, sim_result)
    tenant_summary = build_tenant_summary(cfg, sim_result)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([r.to_dict() for r in sim_result.completed_requests]).to_csv(outdir / "request_log.csv", index=False)
    tenant_summary.to_csv(outdir / "tenant_summary.csv", index=False)
    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(json.dumps(run_summary, indent=2))
    print("\nSaved:")
    print(f"- {outdir / 'request_log.csv'}")
    print(f"- {outdir / 'tenant_summary.csv'}")
    print(f"- {outdir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
