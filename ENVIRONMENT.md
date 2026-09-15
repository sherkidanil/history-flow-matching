# Environment

Measured platform, dependency, simulator, and runtime information is recorded
here by milestone tooling. Values must not be estimated or copied from an
unverified environment.

## Verified environments

M0 completed on 2026-09-15 on the Linux cluster with the official
`openporousmedia/opmreleases:latest` container. The measured image was Flow
2026.04, image ID
`sha256:a3aedae0558b137cbe6c0204414969670dd49834d8ef8d69a49a4254c4d90105`,
created 2026-05-20, with an unpacked size of 1,136,346,890 bytes. The SPE1
input came from `OPM/opm-tests` commit
`7486607ca47722cf6be73e914bb80cd10088a508`.

The cluster home filesystem had only 5,436,391,424 free bytes, so simulator
scratch stayed in Docker's filesystem on `/mnt/local`; only `.SMSPEC` and
`.UNSMRY` were copied back. The transient filesystem had 32,324,386,816 free
bytes before the accepted run, above the mandatory 20 GiB reserve. The
container wrapper discards all other simulator files by default. Set
`OPM_KEEP_SIMULATOR_FILES=1` only for the small number of explicitly selected
visualization runs.

The macOS development environment was re-probed after adding PyTorch: Torch
2.14.0 detects and selects the Apple MPS device. Both the 3D U-Net and UNO
models completed a `(1, 1, 5, 28, 19)` forward pass on MPS. A deterministic
two-step CPU training smoke test on eight reduced-prior realizations completed
with finite losses; this is an implementation check, not a trained scientific
model.

For Egg MPS, the `scikit-mps` wheel's bundled executable failed on both
available platforms: it is not arm64-compatible on the Mac and segfaulted on
Linux. MPSlib built from official source commit
`a47718fc0e2c7c6f3411de429e51f1267b5d7f7c` succeeded on the Linux x86_64
cluster with `scikit-mps` and NumPy 1.26.4. A 16-realization, full-grid
`mps_snesim_tree` pilot using 16 processes completed in 52.46 seconds wall
time (404.30 seconds aggregate user CPU). The accepted path therefore uses an
explicit source-built executable directory; the package binary is rejected.

Egg FM training uses the existing
`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` cluster image with dependencies
added to an ephemeral environment by `uv`. On an A100 80 GB, a full-grid
batch-16 step with the final coarse-attention-only U-Net (base width 16, 772,241
parameters) took 1.83 seconds and peaked at 1,044,154,880 allocated CUDA bytes.
The shared 16-epoch budget is therefore 5,008 optimizer steps per strategy;
this budget was fixed before fitting any of the three scientific models.

The complete 5,000-member MPS generation used 32 worker processes and the same
source-built MPSlib commit. The elapsed time between the launch PID-file mtime
and final HDF5 mtime was 5,576.82 seconds (92.95 minutes). The result occupied
295,252,077 bytes. Its subsequent FM training ran on an A100 80 GB with Torch
2.4.1+cu121, 772,241 parameters, and 5,008 optimizer steps; the first, final,
and minimum recorded losses were 2.07466, 0.29491, and 0.20995.

The restart-free Egg realization-100 truth run took approximately 32 seconds
from measured launch/output timestamps and produced only a 1,400-byte SMSPEC
and 25,920-byte UNSMRY. It completed all 120 monthly output points through day
3,600 with terminal FOPT 505,246.84375 m3. A 100-member, 16-worker Egg forward
batch completed without failures; individual measured runtimes summed to
3,934.83 seconds while wall time was reduced by concurrency.

<!-- BEGIN AUTO-GENERATED M0 PROBE -->
## Latest measured M0 probe

- Operating system: Linux 5.15.0-151-generic
- Machine architecture: x86_64
- Python: 3.11.16
- CPU count: 64
- Memory bytes: 540825333760
- Free disk bytes: 5436391424
- OPM Flow available: True
- OPM Flow version: flow 2026.04
- Torch installed: False
- Selected Torch device: cpu
- SPE1 smoke status: ok
- SPE1 runtime seconds: 2.2177659198641777
- SPE1 final FOPT: 45898384.0
- SPE1 SMSPEC bytes: 1820
- SPE1 UNSMRY bytes: 33840
- SPE1 total output bytes: 35660
- Transient simulator disk free bytes before run: 32324386816
- SPE1 restart output present: False
<!-- END AUTO-GENERATED M0 PROBE -->
