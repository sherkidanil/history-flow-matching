"""Derive the Egg FM generator acceptance table from raw evaluation reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from fmgeo.artifacts import sha256_file
from fmgeo.metrics.egg import assess_egg_generator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"declaration_status", "limits"}:
        raise ValueError("acceptance configuration must contain status and limits only")
    status = payload["declaration_status"]
    limits = payload["limits"]
    if not isinstance(status, str) or not status or not isinstance(limits, dict):
        raise ValueError("invalid generator acceptance configuration")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for report_path in args.report:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        strategy = str(report["strategy"])
        if strategy in seen:
            raise ValueError(f"duplicate evaluation report for strategy {strategy}")
        seen.add(strategy)
        assessment = assess_egg_generator(
            report["metrics"],
            roundtrip_relative_error=float(report["roundtrip_relative_error"]),
            limits={str(name): float(value) for name, value in limits.items()},
        )
        criteria = assessment["criteria"]
        for metric, criterion in criteria.items():
            rows.append(
                {
                    "declaration_status": status,
                    "strategy": strategy,
                    "accepted": assessment["accepted"],
                    "metric": metric,
                    "value": criterion["value"],
                    "limit": criterion["limit"],
                    "passed": criterion["passed"],
                    "source_report": report_path.as_posix(),
                    "source_report_sha256": sha256_file(report_path),
                    "sample_artifact_sha256": report["artifact"]["sha256"],
                }
            )
    if seen != {"augmentation", "mps", "procedural"}:
        raise ValueError("table requires augmentation, mps, and procedural reports")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
