# make_rocm_kit.sh -- usage and post-build notes

This document holds the long-form guidance that the script prints at the end
of a successful run. Paths below use `<tree>` for the `{torch}` checkout root.

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
torch-rocm-kit/lantern/     -> install.R layout       (LANTERN_URL candidate)
src/lantern/CMakeLists.txt  -> [hip-shim-rocm] block appended at the end
                               (remove with: ./tools/make_rocm_kit.sh --unpatch)
```

## Load-time note (v10)

The staged tree is HERMETIC: every wheel library carries `DT_RPATH=$ORIGIN`
and `liblantern` carries `DT_RPATH='$ORIGIN:<tree>/build-lantern/libtorch/lib'`.
`DT_RPATH` has priority over `LD_LIBRARY_PATH` and the ld.so cache
(`DT_RUNPATH` does not), so neither a shell exporting
`LD_LIBRARY_PATH=/opt/rocm/lib` nor ROCm's cache entries can swap a shared
soname (`libhsa-runtime64.so.1`, ...) onto the 7.x chain. The v6
`~/.Renviron` cure is no longer needed (harmless if already set).

If loading still fails, run `./tools/make_rocm_kit.sh --doctor`
(dlopen probe + LD_DEBUG culprit extraction -- prints WHO pulls `/opt/rocm`
in).

## Failure handling (v9)

Any failing gate QUARANTINES `inst/lib/liblantern.so` and
`torch-rocm-kit/lantern/lib/liblantern.so` before exiting. `R CMD INSTALL .`
goes through `{torch}`'s dummylantern target ("*** Skip building lantern.")
and reinstalls the PREVIOUS `inst/lib/liblantern.so`: a stale pre-alignment
binary is what resurrected the 6.4/7.2 chimera (undefined symbol ...
version ROCR_1). After a quarantine the failure is loud (`liblantern.so`
missing), never silent.

## Path A -- dev loop (recommended for #907)

```r
devtools::load_all("<tree>")        # libs loaded from inst/
# or: R CMD INSTALL "<tree>"
```

## Path B -- clean install through install.R

```r
Sys.setenv(TORCH_HOME  = "<tree>/inst")
Sys.setenv(TORCH_URL   = "<tree>/build-lantern/libtorch")
Sys.setenv(LANTERN_URL = "<tree>/torch-rocm-kit/lantern")
torch::install_torch()             # then restart R
source("test_torch_rocm.R")
```

## Startup prompt note (v15)

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

With the v12 artifacts the first kernel on gfx900 died with
`HIP error: invalid device function`. Doctor `gpu_arch` lists the
code-object targets the strings heuristic finds in the staged fatbins --
gfx900 may appear among them yet kernels can still fail at launch
(bundle-id strings can over-match; only a real launch proves coverage).
If the smoke test dies there: `HSA_OVERRIDE_GFX_VERSION=9.0.6` is the quick
experiment (elementwise usually OK, GEMM may trap); the supported path for
full BLAS on Vega10 is the ROCm 5.7 container. Never mix ROCm 6.4 and 7.2
apt packages, and never define `__HIP_PLATFORM_NVIDIA__` together with
`__HIP_PLATFORM_AMD__` (the shim defines the AMD pair for the lantern
target).
