from __future__ import annotations

import csv
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


def _markdown_table(report: str, header: str) -> list[list[str]]:
    lines = report.splitlines()
    start = lines.index(header)
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_g4_report_table_matches_published_csv() -> None:
    report = Path("REPORT.md").read_text(encoding="utf-8")
    header = (
        "| Variant | N | Bounds | Localization | Final misfit | Field spread | "
        "Effective members | Effective fraction | Cells clipped | FOPT covers truth |"
    )
    published = _markdown_table(report, header)
    display_to_variant = {
        "Raw unbounded": "raw_na8",
        "PCA unbounded": "pca_na8",
        "FM unbounded": "fm_na8",
        "Raw bounded": "raw_na8_bounded",
        "PCA bounded": "pca_na8_bounded",
        "FM bounded": "fm_na8_bounded",
        "Raw bounded + localized": "raw_na8_localized",
        "PCA bounded + localized": "pca_na8_localized",
        "FM bounded + localized": "fm_na8_localized",
        "Raw bounded, N=200": "raw_na8_n200",
        "FM bounded, N=200": "fm_na8_n200",
    }
    with Path("results/tables/g4_bounded_controls.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        source = {row["variant"]: row for row in csv.DictReader(stream)}

    assert {display_to_variant[row[0]] for row in published} == set(source)
    for row in published:
        variant = display_to_variant[row[0]]
        values = source[variant]
        clipped = values["clipped_fraction"]
        expected = [
            row[0],
            str(int(values["ensemble_size"])),
            "none"
            if not values["logk_lower_bound"]
            else (
                f'[{float(values["logk_lower_bound"]):.0f}, '
                f'{float(values["logk_upper_bound"]):.0f}]'
            ),
            "64 m" if values["localization_enabled"] == "True" else "no",
            f'{float(values["mean_normalized_data_misfit"]):.5f}',
            f'{float(values["mean_field_spread"]):.5f}',
            f'{float(values["effective_ensemble_members"]):.2f}',
            f'{float(values["effective_ensemble_fraction"]):.3f}',
            "n/a" if not clipped else f"{100.0 * float(clipped):.4f}%",
            "yes" if values["covered"] == "True" else "no",
        ]
        assert row == expected
