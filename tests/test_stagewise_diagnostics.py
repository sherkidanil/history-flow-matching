from __future__ import annotations

import numpy as np
import pytest

from fmgeo.metrics.egg import (
    bimodality_coefficient,
    closest_stage_to_target,
    summarize_egg_stage,
)


def test_summarize_egg_stage_reuses_declared_scientific_metrics() -> None:
    fields = np.array(
        [
            [[[1.0, 1.0]]],
            [[[1.0, 0.0]]],
        ]
    )
    simulated = np.array([[1.0], [3.0]])
    active = np.ones((1, 1, 2), dtype=bool)
    wells = {"INJECT1": (0, 0), "PROD1": (0, 1)}

    summary = summarize_egg_stage(
        fields,
        simulated_data=simulated,
        observation=np.array([1.0]),
        observation_covariance=np.eye(1),
        fopt=np.array([0.0, 10.0]),
        truth_fopt=5.0,
        active_mask=active,
        threshold=0.5,
        wells=wells,
        truth_connectivity={("INJECT1", "PROD1"): 1.0},
    )

    assert summary == pytest.approx(
        {
            "mean_normalized_data_misfit": 2.0,
            "bimodality_coefficient": bimodality_coefficient(fields[:, active]),
            "connectivity_mae_to_truth": 0.5,
            "largest_component_fraction_p50": 1.0,
            "fopt_p10": 1.0,
            "fopt_p50": 5.0,
            "fopt_p90": 9.0,
            "covered": True,
        }
    )


def test_summarize_egg_stage_rejects_incomplete_truth_connectivity() -> None:
    fields = np.ones((2, 1, 1, 2))

    with pytest.raises(ValueError, match="connectivity pairs"):
        summarize_egg_stage(
            fields,
            simulated_data=np.ones((2, 1)),
            observation=np.ones(1),
            observation_covariance=np.eye(1),
            fopt=np.ones(2),
            truth_fopt=1.0,
            active_mask=np.ones((1, 1, 2), dtype=bool),
            threshold=0.5,
            wells={"INJECT1": (0, 0), "PROD1": (0, 1)},
            truth_connectivity={},
        )


def test_closest_stage_to_target_breaks_ties_by_earliest_stage() -> None:
    stage, gap = closest_stage_to_target({0: 70.0, 1: 50.0, 2: 30.0}, target=60.0)

    assert stage == 0
    assert gap == 10.0


@pytest.mark.parametrize(
    ("misfits", "target", "message"),
    [({}, 1.0, "at least one"), ({0: np.nan}, 1.0, "finite"), ({0: 1.0}, np.inf, "finite")],
)
def test_closest_stage_to_target_validates_inputs(
    misfits: dict[int, float], target: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        closest_stage_to_target(misfits, target=target)
