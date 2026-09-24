# G4 Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair the bounded Egg ES-MDA semantics, complete the controlled G4 comparison, and publish claims and provenance that match the evidence.

**Architecture:** Bounds are implemented by a shared decoder wrapper so the stored field and simulator input are identical. Spatial localization remains cell-based for Raw/FM and is projected to PCA coordinates by squared PCA loadings. Builders validate an explicit variant specification and derive all tables and figures from one analysis commit.

**Tech Stack:** Python 3.11, NumPy, SciPy, h5py, PyTorch, pytest, Ruff, mypy, OPM Flow in Docker.

---

### Task 1: Establish the clean baseline

**Files:**
- Read: 'pyproject.toml'
- Read: 'tests/'
- Modify: none

**Step 1: Install the locked environment**

Run:

    uv sync --frozen

Expected: environment resolves without lockfile changes.

**Step 2: Run the fast baseline**

Run:

    uv run pytest -m 'not slow' -q
    uv run ruff check .
    uv run mypy src scripts

Expected: all commands pass. If an existing failure appears, record it before any implementation.

### Task 2: Make bounds part of decoding

**Files:**
- Modify: 'src/fmgeo/inverse/egg.py'
- Modify: 'scripts/m8_invert_egg.py'
- Test: 'tests/test_egg_inversion.py'

**Step 1: Write the failing decoder test**

Add a test for the desired helper:

    bounded, fraction = decode_bounded_fields(
        parameters,
        decode=decode,
        bounds=(0.0, 10.0),
        active=active,
    )
    np.testing.assert_array_equal(bounded[:, active], [[0.0, 5.0], [3.0, 10.0]])
    assert fraction == pytest.approx(0.5)

Also assert that 'run_egg_esmda' stores the exact field returned by its supplied decoder and does no independent post-decode mutation.

**Step 2: Verify RED**

Run:

    uv run pytest tests/test_egg_inversion.py -k 'bounded or exact_decoder' -q

Expected: FAIL because 'decode_bounded_fields' does not exist and the old API clips inside 'run_egg_esmda'.

**Step 3: Implement the bounded decoder**

Create a small helper that validates shapes, calls the base decoder, clamps only active cells, and returns the effective field plus activation fraction. Change 'run_egg_esmda' to accept a decoder returning effective fields and optional per-stage activation metadata rather than 'field_bounds' and 'active'.

In 'm8_invert_egg.py', wrap the Raw/PCA/FM decoder once when 'logk_bounds' is configured. Record:

    bound_policy = 'decoded_field_clamp'

Ensure '_write_stage' stores the effective decoded field and activation fraction.

**Step 4: Verify GREEN**

Run:

    uv run pytest tests/test_egg_inversion.py -q

Expected: PASS.

**Step 5: Commit**

    git add src/fmgeo/inverse/egg.py scripts/m8_invert_egg.py tests/test_egg_inversion.py
    git commit -m 'fix: make Egg bounds part of decoding'

### Task 3: Add PCA localization projection

**Files:**
- Modify: 'src/fmgeo/inverse/localization.py'
- Modify: 'scripts/m8_invert_egg.py'
- Test: 'tests/test_localization.py'
- Test: 'tests/test_egg_inversion.py'

**Step 1: Write failing projection tests**

Test the desired API:

    projected = project_localization_to_pca(cell_taper, pca.components)

Assert shape '(pca_rank, observation_count)', values in '[0,1]', invariance to component sign, and squared-loading weighted averages for a hand-computed matrix.

Change the validation test to accept 'method=pca' with localization.

**Step 2: Verify RED**

Run:

    uv run pytest tests/test_localization.py tests/test_egg_inversion.py -k 'pca or projection' -q

Expected: FAIL because projection is absent and PCA localization is rejected.

**Step 3: Implement minimal projection**

Compute:

    weights = components**2
    weights /= weights.sum(axis=1, keepdims=True)
    projected = weights @ cell_taper

Use the projected taper for PCA and record 'pca_squared_loading_projection' in provenance. Raw and FM retain the cell taper.

**Step 4: Verify GREEN**

Run:

    uv run pytest tests/test_localization.py tests/test_egg_inversion.py -q

Expected: PASS.

**Step 5: Commit**

    git add src/fmgeo/inverse/localization.py scripts/m8_invert_egg.py tests/test_localization.py tests/test_egg_inversion.py
    git commit -m 'feat: project Egg localization to PCA coordinates'

### Task 4: Define and validate the full G4 control matrix

**Files:**
- Create: 'configs/egg/inversion_na8_bounded.yaml'
- Create: 'configs/egg/inversion_na8_pca_localized.yaml'
- Create: 'configs/egg/inversion_na8_ensemble200_bounded.yaml' if adjustment is required
- Modify: 'scripts/g4_build_controls.py'
- Modify: 'scripts/f2_build_remedies.py'
- Test: 'tests/test_g4_controls_script.py'
- Test: 'tests/test_f2_remedies_script.py'

**Step 1: Write failing variant and nullable-clipping tests**

Define the expected ordered variants:

    raw_na8
    pca_na8
    fm_na8
    raw_na8_bounded
    pca_na8_bounded
    fm_na8_bounded
    raw_na8_localized
    pca_na8_localized
    fm_na8_localized
    raw_na8_n200
    fm_na8_n200

Assert that unbounded stage rows carry 'clipped_fraction=None', bounded rows require explicit clipping metadata, and report metadata must match expected bounds, localization, and ensemble size.

Assert 'effective_ensemble_fraction == effective_ensemble_members / ensemble_size'.

**Step 2: Verify RED**

Run:

    uv run pytest tests/test_g4_controls_script.py tests/test_f2_remedies_script.py -q

Expected: FAIL on the old six-variant schema and zero default clipping.

**Step 3: Implement the schema and configs**

Introduce a typed G4 variant specification and validate each report. Emit blank CSV fields and missing plot values for unbounded clipping. Use a descriptive figure title such as 'Egg G4 bound, localization, and ensemble-size controls'.

**Step 4: Verify GREEN**

Run:

    uv run pytest tests/test_g4_controls_script.py tests/test_f2_remedies_script.py -q

Expected: PASS.

**Step 5: Commit**

    git add configs/egg scripts/g4_build_controls.py scripts/f2_build_remedies.py tests/test_g4_controls_script.py tests/test_f2_remedies_script.py
    git commit -m 'feat: define complete G4 control matrix'

### Task 5: Use a common F2 regression rank

**Files:**
- Modify: 'scripts/f2_linearity_diagnostics.py'
- Modify: 'scripts/f2_build_remedies.py'
- Test: 'tests/test_f2_linearity_script.py'
- Test: 'tests/test_f2_remedies_script.py'

**Step 1: Write the failing rank test**

For five-fold evaluation with 100 samples, assert that every parameterization uses rank 79 even if its available rank is 85 or 99. Preserve 'available_rank' separately.

**Step 2: Verify RED**

Run:

    uv run pytest tests/test_f2_linearity_script.py tests/test_f2_remedies_script.py -q

Expected: FAIL because each method currently uses its own retained rank.

**Step 3: Implement the common-rank option**

Add a required or explicit 'common_rank' path to the diagnostic builder and set the published protocol to 79. Reject requests above the smallest fold-training rank or available parameter rank.

**Step 4: Verify GREEN**

Run:

    uv run pytest tests/test_f2_linearity_script.py tests/test_f2_remedies_script.py -q

Expected: PASS.

**Step 5: Commit**

    git add scripts/f2_linearity_diagnostics.py scripts/f2_build_remedies.py tests/test_f2_linearity_script.py tests/test_f2_remedies_script.py
    git commit -m 'fix: compare F2 linearity at common rank'

### Task 6: Add numeric report-table verification

**Files:**
- Modify: 'tests/test_report_links.py'
- Modify later: 'REPORT.md'
- Source: 'results/tables/egg_matched_misfit_v2.csv'
- Source: 'results/tables/f2_linearity.csv'
- Source: 'results/tables/g4_bounded_controls.csv'

**Step 1: Write the failing parser test**

Parse the named Markdown tables and assert that displayed values equal their CSV source after documented rounding. Assert that unbounded clipping is rendered as 'n/a'.

**Step 2: Verify RED**

Run:

    uv run pytest tests/test_report_links.py -q

Expected: FAIL on current prose/table values.

**Step 3: Add only reusable test helpers**

Implement a small Markdown table parser in the test module and keep source-to-column mappings explicit.

**Step 4: Leave RED until Task 9**

The test intentionally remains failing until generated evidence and prose are updated.

### Task 7: Launch the missing controlled inversions

**Files:**
- Generate ignored HDF5 artifacts under 'artifacts/'
- Generate tracked JSON reports under 'results/raw/'
- Generate manifests under 'artifacts/'
- Update: 'REPRODUCE.md'

**Step 1: Commit the analysis code revision**

Record the full 40-character commit used for every run.

**Step 2: Run bounded-unlocalized N=100**

Run Raw, PCA, and FM with 'configs/egg/inversion_na8_bounded.yaml'.

Expected: nine stages, zero failed members, explicit bound activation at every stage.

**Step 3: Run bounded-localized N=100**

Rerun Raw and FM with the corrected decoder and run PCA with projected localization.

Expected: nine stages or an explicit, retained failure artifact. Do not convert a failed run into a positive result.

**Step 4: Run bounded-unlocalized FM N=200**

Use the same bounded N=200 protocol as Raw.

Expected: nine stages and zero failed members, or a retained failure report.

**Step 5: Verify artifacts**

For each successful artifact, validate HDF5 stage count, report checksum, full source commit, parameterization label, bounds, localization policy, and ensemble size.

**Step 6: Commit tracked reports and exact commands**

    git add REPRODUCE.md results/raw
    git commit -m 'results: add controlled G4 inversion reports'

### Task 8: Regenerate F2 and G4 evidence from one revision

**Files:**
- Modify generated: 'results/raw/f2_linearity.json'
- Modify generated: 'results/tables/f2_linearity.csv'
- Modify generated: 'results/tables/f2_remedies.csv'
- Modify generated: 'results/tables/g4_bounded_controls.csv'
- Modify generated: 'results/figures/g4_bounded_controls.svg'
- Modify generated: 'results/figures/g4_bounded_controls.pdf'
- Test: relevant builder tests

**Step 1: Record one analysis commit**

Use the same full commit for every 'derived_git_commit' field in this section.

**Step 2: Regenerate F2**

Run the documented diagnostics and remedy builders with common rank 79.

**Step 3: Regenerate G4**

Run 'g4_build_controls.py' with the complete ordered control matrix.

**Step 4: Validate provenance**

Assert all derived outputs name the same analysis commit and all source hashes match local artifacts/reports.

**Step 5: Commit**

    git add results
    git commit -m 'results: rebuild F2 and G4 evidence'

### Task 9: Align the paper text with the evidence

**Files:**
- Modify: 'REPORT.md'
- Modify: 'BLOCKERS.md'
- Modify: 'README.md'
- Modify: 'REPRODUCE.md'
- Modify: 'results/raw/g4_localization_failure.json'
- Test: 'tests/test_report_links.py'

**Step 1: Update claims and numbers**

Use the generated CSV values. Describe FM bimodality as decoder-stable rather than preserved by assimilation; show final PCA connectivity beside FM; carry the matched-stage limitation into prose; report N-normalized effective members; replace causal localization language with the factorial result; fix all rounding and full hashes.

**Step 2: Document exact run commands**

Include every inversion command before the builder command and identify expected outputs and checksums.

**Step 3: Verify GREEN**

Run:

    uv run pytest tests/test_report_links.py tests/test_g4_controls_script.py tests/test_f2_linearity_script.py -q

Expected: PASS, including numeric Markdown-to-CSV checks.

**Step 4: Commit**

    git add REPORT.md BLOCKERS.md README.md REPRODUCE.md results/raw/g4_localization_failure.json tests/test_report_links.py
    git commit -m 'docs: align Egg conclusions with controlled evidence'

### Task 10: Final verification

**Files:**
- Modify: none unless verification finds a defect

**Step 1: Run all automated checks**

    uv run pytest -m 'not slow' -q
    uv run ruff check .
    uv run mypy src scripts

Expected: all pass with no warnings attributable to the change.

**Step 2: Audit generated provenance**

Check clean status, full hashes, source/derived commits, table row order, and report links.

**Step 3: Review the branch diff**

    git diff --check origin/codex/g4-final-report...HEAD
    git diff --stat origin/codex/g4-final-report...HEAD

Expected: no whitespace errors; changes limited to the approved design.

**Step 4: Commit any verification-only correction**

Use a focused commit message and rerun the affected checks.
