# make_rocm_kit.sh -- usage and post-build notes

This document holds the long-form guidance that the script prints at the end
of a successful run. Paths below use `<tree>` for the `{torch}` checkout root.

Last change: v25 -- CI migration (phase 0+1): `--ci`, fatal wheel/SDK HIP
mismatch gate, `rocm-kit.yaml` + `build-rocm-lantern` action. v24 added
`--install` (a plain install_torch() fetches a CPU/CUDA wheel: ABI chimera).

## Aligned tree

```
build-lantern/libtorch/     -> staged libtorch        (TORCH_URL candidate)
build-lantern/hip-compat/   -> cuda->HIP shim headers + HIP SDK math-header
                               snapshot (hipsparse/hipblas/hipblaslt: the
                               wheel bundles only the hip/ runtime subset;
                               compile-time only) + the -fsyntax-only probe
                               (compat-probe.cpp / flags-probe.log)
build-lantern/liblantern.so -> built library
inst/lib/liblantern.so      -> dev loading (load_all)
inst/lib/libcaffe2_nvrtc.so -> nvrtc bridge (device RNG / JIT)
torch-rocm-kit/lantern/     -> install.R layout       (LANTERN_URL candidate)
torch-rocm-kit/lantern/lib/libcaffe2_nvrtc.so
                            -> nvrtc bridge staged for install_torch()
src/lantern/CMakeLists.txt  -> [hip-shim-rocm] block appended at the end
                               (remove with: ./tools/make_rocm_kit.sh --unpatch)
```

## Load-time note

The staged tree is HERMETIC: every wheel library carries `DT_RPATH=$ORIGIN`
and `liblantern` carries `DT_RPATH='$ORIGIN:<tree>/build-lantern/libtorch/lib'`.
`DT_RPATH` has priority over `LD_LIBRARY_PATH` and the ld.so cache
(`DT_RUNPATH` does not), so neither a shell exporting
`LD_LIBRARY_PATH=/opt/rocm/lib` nor ROCm's cache entries can swap a shared
soname (`libhsa-runtime64.so.1`, ...) onto the 7.x chain. The older
`~/.Renviron` cure is no longer needed (harmless if already set).

If loading still fails, run `./tools/make_rocm_kit.sh --doctor`
(dlopen probe + LD_DEBUG culprit extraction -- prints WHO pulls `/opt/rocm`
in).

## Failure handling

Any failing gate QUARANTINES `inst/lib/liblantern.so` and
`torch-rocm-kit/lantern/lib/liblantern.so` before exiting. `R CMD INSTALL .`
goes through `{torch}`'s dummylantern target ("*** Skip building lantern.")
and reinstalls the PREVIOUS `inst/lib/liblantern.so`: a stale pre-alignment
binary is what resurrected the 6.4/7.2 chimera (undefined symbol ...
version ROCR_1). After a quarantine the failure is loud (`liblantern.so`
missing), never silent.

## nvrtc bridge

The installed `libtorch_cpu.so` dlopens `libcaffe2_nvrtc.so` by bare name
(device RNG state and JIT kernel compilation). The kit copies the wheel's
`libcaffe2_nvrtc.so` next to `liblantern.so` in BOTH layouts: the dev copy
(`inst/lib`) gets `DT_RPATH='$ORIGIN:<tree>/build-lantern/libtorch/lib'` so
its own NEEDED chain stays reachable, and the kit copy keeps
`DT_RPATH='$ORIGIN'` because `install_torch()` drops it next to the full
wheel. A failed kit run quarantines both copies along with `liblantern.so`.

## Path A -- dev loop (recommended for #907)

```r
devtools::load_all("<tree>")        # libs loaded from inst/
# or: R CMD INSTALL "<tree>"
```

## Path B -- clean install (installed R package)

```sh
R CMD INSTALL "<tree>"                   # R side (incl. the R/cuda.R patch)
"<tree>/tools/make_rocm_kit.sh" --install  # ROCm libtorch + kit lantern
# restart R, then:
Rscript "<tree>/tools/test_torch_rocm.R"
```

`--install` is the missing half of the kit: it copies the clean ROCm libtorch
layout (11 GB, `lib/include/share/bin` only) and `torch-rocm-kit/lantern` into
`torch_install_path()` with `reinstall = TRUE`. Equivalent manual form:

```r
Sys.setenv(TORCH_URL   = "<tree>/torch-rocm-kit/libtorch",
           LANTERN_URL = "<tree>/torch-rocm-kit/lantern")
torch::install_torch(reinstall = TRUE)   # then restart R
```

`torch-rocm-kit/libtorch` is built by the kit (hardlinked to
`build-lantern/libtorch`, no extra space) because the staging root is the
WHEEL root: pointing `TORCH_URL` there would copy the whole Python `torch`
package (105 entries, ~1.5 GB) into the R package.

Do NOT rely on a bare `torch::install_torch()`: without `TORCH_URL` it fetches
a CPU/CUDA wheel. `R CMD INSTALL` re-creates the package but never ships a
libtorch, so the package ends up with a CPU/ROCm mix: liblantern (ROCm) then
loads `libtorch_hip.so` from the build tree while `libtorch_cpu.so` comes from
the package -- same sonames, different builds. The symptom is an impossible
`device=-80` from `cuda_synchronize()` while every element-wise kernel still
works. `--doctor` section `[install]` now detects it.

If you staged by hardlink (`cp -al`) because of disk space: `--install` unlinks
the destination entries first. `file.copy()` truncates its destination, so
overwriting a hardlink would zero BOTH the installed and the staged files
(seen in the field: 0-byte `libtorch_cpu.so`/`libtorch_hip.so` then a segfault).

For the dev loop, the same bundles can go to `<tree>/inst` instead:
`TORCH_HOME=<tree>/inst` + `TORCH_URL`/`LANTERN_URL` + `install_torch()`.

## CI (GitHub Actions)

Phase 0+1 of `tools/ROCM_CI_PLAN.md` is implemented; the script remains the
engine and YAML only orchestrates it.

- `.github/workflows/rocm-kit.yaml`:
  - `unit` (ubuntu-latest): `bash -n tools/*.sh`, `shellcheck` on the kit
    scripts, then `./tools/test-kit.sh` (no GPU, no ROCm).
  - `build` (ubuntu-22.04): the `build-rocm-lantern` composite action.
- `.github/actions/build-rocm-lantern/action.yml`: installs `cmake rsync
  patchelf unzip ...`, installs the ROCm toolkit from the `rocm6.4` R package
  (the ROCm branch of `{cudatoolkit}`, which extracts the ROCm `.deb` files
  without root), caches and downloads the official libtorch ROCm archive
  (`download.pytorch.org`), then runs
  `./tools/make_rocm_kit.sh --from-zip <archive> --ci`. No Python venv. It does
  NOT run `--install`; CI only publishes the built kit.
- `--ci` is the CI contract: plain ASCII (no ANSI), `cp` without `-v`, and
  stale artifacts are moved to `build-lantern/quarantine/` instead of deleted
  so the workflow can upload them with `if: always()`.
- Version pinning: the `rocm6.4` package pins the versioned repo
  `https://repo.radeon.com/rocm/apt/6.4` (Ubuntu codename suite) in
  `inst/components.tsv`, and `ensure_hip_compat` fails when the toolkit
  major.minor differs from the archive's `+rocmX.Y`; override only with
  `--allow-hip-header-mismatch`.
- The generic `.github/workflows/lantern.yaml` no longer has a `rocm` row:
  it builds against a CPU/CUDA libtorch and cannot produce a ROCm artifact.
- Phase 2 (`smoke-load`, CPU dlopen) and phase 3 (`gpu-smoke`, self-hosted
  AMD runner + `tools/test_torch_rocm.R`) are documented placeholders in
  `rocm-kit.yaml`; they need runner/provisioning decisions.

## Startup prompt note

`library(torch)` only offers to download when the INSTALLED package is
missing `lib/liblantern.so` or `lib/libtorch.so`. Two rules keep it quiet:

1. `install_torch()` wants the kit layout: `LANTERN_URL=<tree>/torch-rocm-kit/lantern`
   (it needs a dir holding `lib/liblantern.so` -- pointing it at
   `build-lantern`, a cmake build tree, fails with
   `Could not install lantern from ...`).
2. once installed, put `TORCH_INSTALL=0` in `~/.Renviron`.

## Tree patches (remove with `--unpatch`)

- `src/lantern/src/Cuda.cpp` (backup: `Cuda.cpp.orig-make-rocm-kit`):
  `cuda_get_device_capability`, `cuda_memory_stats/summary/snapshot`,
  `cuda_empty_cache` and `cuda_runtime_version` now run their REAL
  implementations on the ROCm wheel (upstream guards them behind `__NVCC__`
  and throws "only supported on CUDA runtimes" otherwise); the device RNG /
  JIT bridge (`libcaffe2_nvrtc.so -> libamdhiprtc.so`) is staged as well.
- `R/cuda.R` (marked block): `cuda_synchronize(NULL)` now uses the current
  device (the -1 path crashes inside the wheel's HIP CUDAHooks) and
  `cuda_runtime_version()` decodes the HIP encoding (6.4.43482-style).
  Rerun `R CMD INSTALL` / `devtools::load_all` to pick these up.

## GPU ISA caveat

If the first kernel launch dies with `HIP error: invalid device function`,
run `./tools/make_rocm_kit.sh --doctor` and read `gpu_arch`. The wheel DOES
ship gfx900 code objects (torch 2.8.0+rocm6.4), but a Ryzen APU is usually
gfx90c (Renoir/Cezanne), which the wheel does NOT ship. The usual trap is an
`HSA_OVERRIDE_GFX_VERSION=9.0.0` exported from `~/.bashrc`: a shell-launched
R sees gfx900 and works, while R/RStudio started from a GUI/desktop launcher
does not source the shell profile, sees gfx90c, and fails. Put the override
in `~/.Renviron` (not only `~/.bashrc`) and restart R.

The override unblocks elementwise/pointwise ops; GEMM (matmul/linear/conv)
still needs a rocBLAS Tensile library for the mapped arch (gfx900/906), which
the wheel does NOT ship -> ROCm 5.7 container or a rocBLAS/Tensile rebuild.
`gpu_arch` now reads the real offload bundle ids with
`llvm-objdump --offloading` (strings fallback kept) and judges matmul by the
actual Tensile/kernel files (`rocblas/library`, `hipblaslt`), not by `.so`
strings. Never mix ROCm 6.4 and 7.2 apt packages, and never define
`__HIP_PLATFORM_NVIDIA__` together with `__HIP_PLATFORM_AMD__` (the shim
defines the AMD pair for the lantern target).
