# G4 Review Fixes Design

## Goal

Make the Egg G4 comparison methodologically interpretable and reproducible before
running the separate multi-seed study. The revision addresses review findings 1–19
while keeping the existing single-truth, single-seed scope explicit.

## Alternatives considered

1. **Editorial patch only.** Fast, but it leaves the bounded-state inconsistency and
   confounded G4 comparison in place.
2. **Controlled G4 repair (selected).** Correct the bounded decoder, complete the
   necessary factorial controls, normalize diagnostics, regenerate evidence from
   one analysis commit, and narrow the claims.
3. **Full multi-seed paper study now.** Strongest evidence, but it mixes a large new
   study into a repair PR and delays resolution of the known protocol defects.

## Bounded state semantics

'logk_bounds' defines a deterministic bounded decoder. Every parameterization
maps its ES-MDA parameters to an effective field through the same wrapper:

    parameters -> parameterization decoder -> active-cell clamp -> simulator

The stage field stored in HDF5 is exactly the effective field returned by that
decoder and sent to OPM Flow. The unclamped decoded field is used only to compute
the activation fraction. This makes the state used by the simulator explicit and
prevents 'run_egg_esmda' from silently changing the result of its supplied
decoder. Provenance records the policy as 'decoded_field_clamp'.

The clamp remains a nonlinear, non-invertible decoder operation. The report will
state this limitation and will not describe it as a constrained ES-MDA update.

## Control matrix

At N=100, use a three-by-three comparison where data exist or can be generated:

| Bounds | Localization | Raw | PCA | FM |
|---|---|---:|---:|---:|
| none | none | existing | existing | existing |
| [2,11] | none | new | new | new |
| [2,11] | 64 m | existing rerun | new | existing rerun |

The failed unbounded Raw-localized run remains failure evidence, not a comparable
row. The bounded-unlocalized rows separate bound activation from localization.

At N=200, compare bounded-unlocalized Raw and FM under identical settings.
These rows test ensemble-size sensitivity without mixing bound policies.

For PCA, project the cell-to-observation taper to coefficient space. Each PCA
component receives the squared-loading-weighted mean of the cell taper. The
configuration and report identify this as 'pca_squared_loading_projection'.

## Derived evidence

The G4 builder will:

- represent clipping as missing for unbounded variants;
- add 'effective_ensemble_fraction';
- include the complete control matrix in a stable order;
- use a descriptive figure title;
- validate bound/localization metadata against the variant specification.

The F2 linearity diagnostic will use common rank 79, the largest rank supported
by every 80-member training fold. Its output will identify both available and
used rank.

All F2/G4 derived tables and figures will be regenerated from one committed
analysis revision. Source artifact commits remain separate provenance fields.

## Reporting and reproducibility

'REPORT.md', 'BLOCKERS.md', 'README.md', and 'REPRODUCE.md' will:

- distinguish bound effects from localization effects;
- call unbounded clipping 'n/a';
- report effective-member fractions alongside counts;
- compare FM connectivity with final PCA as well as stage-matched rows;
- describe matched-stage comparisons as diagnostic only;
- remove claims that unchanged FM bimodality proves geological preservation;
- state the single-truth, single-seed limitation;
- include exact commands for every new inversion and derived artifact;
- use consistent rounding and full commit hashes.

Markdown-table tests will parse published values and compare them with their CSV
sources instead of checking only for substrings.

## Verification

Behavioral changes follow test-first development. Unit tests cover bounded decode
semantics, PCA localization projection, nullable clipping, normalized effective
members, common regression rank, report-table consistency, and provenance
validation. The full fast test suite, Ruff, and mypy must pass before experiment
launch. Each generated result is checksum-verified before tables and prose are
updated.
