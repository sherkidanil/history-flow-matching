from __future__ import annotations

import re
import subprocess
from pathlib import Path

RESULT_PATH = re.compile(r"results/[A-Za-z0-9_./*?-]+")


def _tracked_paths(repo: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(completed.stdout.splitlines())


def test_every_report_result_reference_resolves_to_tracked_files() -> None:
    repo = Path.cwd()
    report = (repo / "REPORT.md").read_text(encoding="utf-8")
    tracked = _tracked_paths(repo)
    references = sorted(set(RESULT_PATH.findall(report)))

    assert references
    missing: list[str] = []
    for reference in references:
        if "*" in reference or "?" in reference:
            if not any(repo.glob(reference)):
                missing.append(reference)
        elif (repo / reference).is_dir():
            prefix = reference.rstrip("/") + "/"
            if not any(path.startswith(prefix) for path in tracked):
                missing.append(reference)
        elif reference not in tracked:
            missing.append(reference)
    assert missing == []


def test_raw_json_is_public_but_heavy_raw_data_remains_ignored() -> None:
    repo = Path.cwd()
    json_result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "results/raw/example.json"],
        cwd=repo,
        check=False,
    )
    h5_result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "results/raw/example.h5"],
        cwd=repo,
        check=False,
    )

    assert json_result.returncode == 1
    assert h5_result.returncode == 0


def test_report_publishes_session3_remedy_evidence() -> None:
    report = Path("REPORT.md").read_text(encoding="utf-8")

    assert "## Assimilation-step remedy" in report
    for reference in (
        "results/tables/f2_remedies.csv",
        "results/tables/egg_stagewise_remedies.csv",
        "results/tables/egg_matched_misfit_v2.csv",
        "results/tables/g2_anomaly_diagnostics.csv",
        "results/figures/egg_misfit_geology_tradeoff.svg",
    ):
        assert reference in report
