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
