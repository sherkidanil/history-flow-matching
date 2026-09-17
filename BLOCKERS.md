# Blockers and assumptions

This log records blocked work, attempted remedies, fallbacks, and assumptions.
Entries must be dated and must distinguish measured facts from hypotheses.

## Environment constraints

### Bundled Egg MPS binaries are not portable

- **Observed (2026-09-15):** `scikit-mps` installs on both machines, but its
  bundled `mps_genesim` executable is x86-only on the arm64 Mac (`Exec format
  error`). On the Linux x86_64 cluster, a minimal 6x6x1 run with NumPy 1.26.4
  exits with signal 11 before producing a realization. With the project's
  NumPy 2.x environment, the wrapper also uses the removed `np.NaN` alias.
- **Impact:** The bundled wheel cannot be used for accepted MPS-derived Egg
  data. The arm64 Mac remains unsuitable for this strategy.
- **Attempted remedies:** Tested the documented `mps_genesim` Python interface,
  isolated output directories, an explicit output directory, NumPy below 2,
  and direct executable invocation on Linux; the latter confirmed return code
  `-11` rather than a wrapper path/read error.
- **Resolved capability path (2026-09-15):** Building official MPSlib source on
  the Linux cluster produced a working executable. The same deterministic test
  then returned two `(6, 6, 1)` realizations successfully. Accepted runs must
  use that explicit executable directory with `scikit-mps` and NumPy below 2;
  no substitute prior will be labeled as MPS.

## Open blockers

### The non-monotone FM assimilation-step and ensemble-size controls are only partly explained

- **Configuration audit (2026-09-17):** The `fm`, `fm_na8`, `fm_na16`, and
  `fm_na8_n200` reports use the same observation-noise seed (`20260915`), prior
  SHA-256, truth case, 160 observations, FM checkpoint, FM configuration, and
  training data. Every inflation schedule satisfies `sum(1 / alpha_i) = 1`.
  The 200-member inverse transform is finite for all members. Its first 96
  stage-0 members match the 100-member run exactly; members 96--99 differ only
  at the final inference-batch boundary (stage-0 field relative Frobenius
  difference `5.71e-8`, maximum absolute difference `3.48e-5`). This is not
  large enough to invalidate the control.
- **Measured saturation:** Final prior-relative field distances are `0.15249`
  for `fm_na8`, `0.15426` for `fm_na16`, and `0.14120` for
  `fm_na8_n200`. Per-stage relative field shifts decline to `0.05425`,
  `0.04627`, and `0.06746`, respectively, but do not become zero.
- **Measured collapse and conditioning:** After stage 9, `fm_na16` oscillates
  in misfit while effective ensemble membership falls from `33.96` to `22.43`;
  its normalized covariance-system condition number also rebounds from `65.6`
  to values as high as `141`. This is consistent with, but does not prove,
  noisy covariance updates after decoder-limited field displacement.
- **Unresolved ensemble-size anomaly:** `fm_na8_n200` finishes with more
  effective members (`64.63`) and more ensemble-explained normalized data
  variance (`0.770`) than `fm_na8` (`27.83` and `0.379`), yet its misfit is
  worse (`32.107` versus `6.628`). Ensemble collapse alone therefore cannot
  explain why 200 members are worse than 100. Treat the result as a measured
  non-monotone stochastic ES-MDA outcome pending replicated-seed or localized
  controls, not as evidence that larger ensembles are intrinsically harmful.
- **Source:** `results/tables/g2_anomaly_diagnostics.csv`.

### FM assimilation failure is not explained by the in-sample linear-response fit

- **Measured (2026-09-16, before remedy runs):** On the common 100-member Egg
  stage-0 ensemble, the truncated-SVD linear-response diagnostic gives
  `R²_lin = 1.0000` for raw, `0.98830` for PCA, and `1.0000` for FM. Thus the
  proposed low-FM-`R²` mechanism is not supported by this in-sample diagnostic;
  with up to 99 anomaly directions, it can interpolate the ensemble response.
- **Measured decoder nonlinearity:** Across 200 fixed random member pairs, the
  FM midpoint-affinity error has P10/P50/P90
  `0.07577 / 0.08427 / 0.09331`; raw is exactly zero and PCA is below
  `1.62e-8`.
- **Measured update attenuation:** Across the four original FM assimilation
  steps, relative latent shifts are `0.7324, 0.6012, 0.5069, 0.4561`, while
  corresponding field shifts are only `0.1156, 0.1020, 0.09381, 0.09029`.
  The field/latent shift ratios are `0.1579–0.1980`.
- **Interpretation:** Decoder nonlinearity and attenuation are directly
  observed, but they do not by themselves prove that nonlinearity caused the
  high final data misfit. Remedy comparisons must therefore be treated as
  controlled empirical tests, not confirmation of a preselected mechanism.
- **Sources:** `results/tables/f2_linearity.csv` and
  `results/raw/f2_linearity.json`.

### OPM Flow does not reproduce the official PUNQ-S3 truth forecast

- **Observed (2026-09-15):** The official deck needed explicit `TABDIMS`,
  `AQUDIMS`, `NUMRES`, a terminated `AQUCT` record set, unified summary output,
  and an initial `ORAT` control before Flow 2026.04 could run it. With those
  compatibility changes, the 16.5-year run completed and all six wells exposed
  `WBHP`, `WGOR`, and `WWCT` summary vectors.
- **Measured discrepancy:** OPM produced `FOPT = 2,085,424.375 m3` at 6025 days,
  while the official benchmark description reports `3.87e6 m3`. The relative
  error is 46.1131%, which is not acceptable for publication experiments.
- **Known semantic gap:** `WCUTBACK` is present in the official deck. OPM's own
  reference manual states that this keyword is ignored, and Flow 2026.04 treats
  it as an unsupported critical keyword under strict parsing.
- **Impact:** The 100-realization forward-coverage gate, the full 30,000-member
  prior, and PUNQ history-matching experiments are paused. No result using this
  mismatched forward model will be presented as benchmark-comparable.
- **Attempted remedies:** Added only parser/output compatibility records,
  preserved the official PVT, relative permeability, aquifer, schedule, grid,
  and truth properties, and ran a single storage-safe diagnostic truth case.
- **Resolution needed:** Reproduce the commercial simulator's well cutback and
  aquifer/schedule semantics in a validated OPM-compatible schedule, or obtain
  an authoritative OPM-compatible PUNQ deck/reference curve. The corrected
  truth run must pass before the paused gates resume.

### PUNQ-S3 redistribution terms are not stated

- **Observed:** The Coventry University source page provides the official
  archive but does not state a data license.
- **Impact:** The downloaded archive and extracted files remain ignored and
  will not be redistributed through this repository.
- **Current fallback:** Track only source URLs, measured hashes, validation
  tooling, and acquisition instructions.
- **Resolution needed:** Obtain explicit redistribution terms before any
  benchmark files are published with the code.

## Assumptions

- The mandated object-based prior differs from the official truth generator,
  which used conditional Gaussian random fields. Channel widths and azimuths
  come from `GeolDescr.htm`; unsourced sinusoid amplitudes, wavelengths, width
  dispersion, and net-to-gross multipliers are marked `ASSUMED` in the config.
- The official hard data are continuous normalized well properties, not facies
  labels. For channel layers, normalized values greater than or equal to zero
  are classified as sand; layers 2 and 4 are forced to background per the task
  design. This threshold is an explicit assumption.
- Axis-aligned property correlation lengths and the `porosity >= 0.20` truth
  sand definition are operational validation choices, documented in the config
  and validation output rather than represented as published facts.
- The Egg archive does not prescribe a history/forecast split or observation
  error model for this experiment. The held-out inversion therefore uses the
  first 1,800 of 3,600 days, 90-day WOPR/WWPR observations, 5% independent
  relative errors, and a 1 m3/day standard-deviation floor. These choices are
  labelled `ASSUMED` in `configs/egg/inversion.yaml` and are held identical for
  raw, PCA, and FM.
- Numerical Egg generator acceptance limits were not fixed before the first
  three complete pilots. They are retained as a
  `retrospective_after_complete_pilots` engineering gate in
  `configs/egg/generator_acceptance.yaml`, not presented as confirmatory
  pre-registration. Under this gate augmentation and MPS pass; procedural
  generation fails the KS, X/Y variogram, and spanning-connectivity criteria,
  so no procedural inversion is run.
