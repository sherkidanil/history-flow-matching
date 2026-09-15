# Blockers and assumptions

This log records blocked work, attempted remedies, fallbacks, and assumptions.
Entries must be dated and must distinguish measured facts from hypotheses.

## Open blockers

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
