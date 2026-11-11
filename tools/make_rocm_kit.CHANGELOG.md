# make_rocm_kit.sh -- CHANGELOG

Full design notes, one section per version (newest first).

CHANGELOG v25 -- CI migration, phase 0+1 (GitHub Actions).
  The kit stays the engine; CI only orchestrates it.
  v25:
  (a) `--ci`: plain ASCII logging (no ANSI), `cp` instead of `cp -v`,
      `KIT_KEEP_ARTIFACTS=1` + `NO_COLOR=1`. On failure the stale
      liblantern/nvrtc artifacts are MOVED to `build-lantern/quarantine/`
      instead of deleted, so the workflow can upload them with
      `if: always()`.
  (b) `--allow-hip-header-mismatch`: `ensure_hip_compat` now extracts the
      wheel HIP major.minor from `libtorch/version.py` (`+rocmX.Y`) and the
      system SDK version from `.info/version`; a mismatch is FATAL by default
      (header/ABI risk), opt-out with the new flag.
  (c) `apply_cuda_r_block` is gated on `libtorch/lib/libtorch_hip.so` like
      `patch_cuda_hip`/`ensure_hip_compat`, so CPU/CUDA trees never get the
      R-side HIP block.
  (d) the staging log reads the real wheel version from `version.py` instead
      of the hardcoded `2.8.0+rocm6.4`.
  (e) SIGPIPE-141 hardening: `find|head` pipelines use `-print -quit` or a
      trailing `|| true` under `set -o pipefail`.
  (f) `test-kit.sh` fails loudly if `kit-lib.sh` cannot be sourced, if a key
      helper is missing, or if zero tests ran (no more false green).
  (g) `.gitignore`: `torch-rocm-kit/`, `*.orig-make-rocm-kit`,
      `*.kit-applied`.
  CI files:
  - `.github/actions/install-rocm/` removed: the ROCm toolkit now comes from
    the `rocm6.4` R package (ROCm branch of `{cudatoolkit}`), which extracts
    the versioned `repo.radeon.com/rocm/apt/6.4` `.deb` files without root.
  - `.github/actions/build-rocm-lantern/action.yml` (new composite): deps ->
    `rocm6.4` R package (`ROCM_PATH`) -> cached libtorch ROCm archive from
    `download.pytorch.org` -> `make_rocm_kit.sh --from-zip --ci` -> ELF/RPATH
    audit -> artifact upload (kit + logs + quarantine). No Python venv.
  - `.github/workflows/rocm-kit.yaml` (new): `unit` (bash -n + shellcheck +
    `test-kit.sh`) and `build` (composite action); `smoke-load` (CPU) and
    `gpu-smoke` (self-hosted AMD) are documented placeholders for phases 2-3.
  - `.github/workflows/lantern.yaml`: the inert `rocm6.4` row and the invalid
    `rocm:` input are removed; ROCm now belongs to `rocm-kit.yaml`.

CHANGELOG v24 -- add the missing INSTALL half: `--install` + doctor detection.
  Field: kit build + `R CMD INSTALL .` + smoke test -> cuda_synchronize()
  fails with an impossible "device=-80" (num_gpus=1) while randn/add/reduction
  and every other cuda_* call work. Diagnosis: the INSTALLED libtorch was a
  CPU/CUDA wheel (no libtorch_hip.so, different libtorch_cpu.so md5 than the
  staged tree). The kit only STAGES libtorch; `R CMD INSTALL` re-creates the
  package without it, and a bare `torch::install_torch()` (no TORCH_URL)
  fetches a CPU/CUDA wheel. liblantern (ROCm) then resolves libtorch_hip.so
  through its RPATH to build-lantern/libtorch while libtorch_cpu.so comes
  from the package: same sonames, different builds -> CUDAHooks vtable
  mismatch, so `torch::cuda::synchronize()` dispatches to isGPUArch() with
  an uninitialised DeviceIndex byte (-80). Not an R-patch bug.
  v24:
  (a) `make_rocm_kit.sh --install` (standalone or with --from-venv/--from-zip)
      runs torch::install_torch(reinstall = TRUE) with TORCH_URL +
      LANTERN_URL, then verifies libtorch_hip.so/liblantern.so landed and
      libtorch_cpu.so matches staging;
  (b) doctor [install] warns when the installed bundle has no libtorch_hip.so
      or differs from the staged one, and prints the --install fix;
  (c) summary/docs spell out the order: kit -> R CMD INSTALL . -> kit --install.
  (d) two safeguards found while testing --install on a hardlinked staging
      (the cp -al disk fallback): before install_torch() the destination
      entries that also exist in the staged lib are unlinked (file.copy()
      truncates its destination, so a hardlink would zero BOTH copies --
      observed: libtorch_cpu.so/libtorch_hip.so went to 0 bytes and R
      segfaulted); and the installer R process runs with TORCH_LOAD=0 so it
      never dlopens the libtorch it is about to overwrite (SIGBUS on a mapped
      .so). The install is now ~1 min for 12 GB on NVMe and the smoke test
      reports 12 ok / 0 warn / 0 fail (matmul still [info] on gfx900).

CHANGELOG v23 -- fix the --doctor [gpu_arch] verdict for the "invalid device function"
  failure on a Ryzen APU. Field: build + install OK, torch_tensor(...,
  device="cuda") prints (H2D copy, no kernel), but the FIRST kernel launch
  (torch_rand, +, ^, *) dies with "HIP error: invalid device function".
  v13/v14 blamed a missing gfx900 code object ("torch 2.8.0+rocm6.4 does not
  build for gfx900"). Ground truth on the real wheel:
    llvm-objdump --offloading libtorch_hip.so
      -> ...hipv4-amdgcn-amd-amdhsa--gfx900  (240 code objects, PRESENT)
  The wheel DOES ship gfx900 kernels. The actual defect is an ARCH mismatch
  hidden by the environment: the Ryzen 5 5600H iGPU is gfx90c (Renoir Vega),
  which is NOT in the wheel. ~/.bashrc exports HSA_OVERRIDE_GFX_VERSION=9.0.0,
  so a SHELL-inherited R sees gfx900 (covered) and works, but R/RStudio
  launched from a GUI/desktop launcher does NOT source ~/.bashrc -> it sees
  gfx90c -> every kernel launch fails. (Verified matrix on the field machine:
  no override -> gfx90c -> FAIL; override 9.0.0 -> gfx900 -> OK; override
  9.0.6 -> gfx906 -> OK; HSA_XNACK=0 alone -> gfx90c -> still FAIL, because
  the wheel has no gfx90c code objects -- the v13-v22 "xnack" theory was
  wrong, gfx900:xnack+ loads the wheel's gfx900 code objects fine.)
  doctor [gpu_arch] now:
  (a) reads the device ISA as seen NOW and, with HSA_OVERRIDE_GFX_VERSION
      stripped (env -u), the REAL hardware ISA, and warns when they differ:
      the override is active in the shell but must reach the R process;
  (b) reads the REAL offload bundle ids from the staged libtorch_hip.so via
      llvm-objdump --offloading (strings fallback kept), and judges coverage
      on the EFFECTIVE arch;
  (c) for a hardware arch that is not in the wheel, recommends the
      HSA_OVERRIDE_MAP sibling and says to put it in ~/.Renviron, not only
      ~/.bashrc. Adds [gfx90c]="9.0.0" (Renoir/Cezanne APU -> gfx900) to the
      map;
  (d) judges matmul by the ACTUAL Tensile/kernel files (rocblas/library,
      hipblaslt), not by .so strings. librocblas.so contains "gfx900" as a
      supported-arch LIST string with NO gfx900 kernel, so v13-v22 wrongly
      reported "matmul covers it"; the wheel ships only gfx908/gfx90a/gfx942/
      gfx1030/gfx1100-1102/gfx1200-1201 Tensile kernels, hence GEMM still
      fails on gfx900/906/90c and needs ROCm 5.7 or a rocBLAS/Tensile rebuild.
  The end-of-run reminder (v14) is reworded accordingly: it no longer claims
  the wheel lacks gfx900.

CHANGELOG v22 -- make the nvrtc bridge part of the INSTALLED artifacts.
  Field failure: torch::cuda_get_rng_state() / JIT kernel compilation dies
  with "Error in dlopen: libcaffe2_nvrtc.so" even though the wheel ships
  the file. Ground truth: torch 2.8.0+rocm6.4 DOES ship
  lib/libcaffe2_nvrtc.so (a real ELF exporting load_nvrtc and linking
  libhiprtc -- not the symlink ensure_nvrtc_stub() would create), so
  ensure_nvrtc_stub() correctly no-ops. But the kit only copied
  liblantern.so into inst/lib and torch-rocm-kit/lantern/lib; the nvrtc
  file stayed in build-lantern/libtorch/lib. At load time the INSTALLED
  libtorch_cpu.so (DT_RUNPATH=$ORIGIN = the package lib dir) dlopens
  "libcaffe2_nvrtc.so" by bare name and does not find it.
  Fix: install_nvrtc_bridge() copies the staged libcaffe2_nvrtc.so next to
  liblantern in BOTH layouts. The dev copy (inst/lib) gets
  DT_RPATH='$ORIGIN:<staged>/lib' because its own NEEDED chain
  (libhiprtc/libamdhip64/libamd_comgr/...) lives in the staged wheel, not
  in inst/lib; the kit copy keeps DT_RPATH='$ORIGIN' (install_torch()
  drops it next to the full wheel). The EXIT trap now also quarantines
  both copies, and doctor [nvrtc] reports whether the installed copies exist.

CHANGELOG v21 -- fix the R CMD INSTALL / library(torch) failure
    "undefined symbol: _ZN3c104cuda20CUDACachingAllocator9allocatorE"
  (demangled: c10::cuda::CUDACachingAllocator::allocator) on a freshly built
  liblantern. The wheel is a HIP build: torch's hipify pass compiled
  c10/cuda/CUDACachingAllocator.cpp as c10/hip/HIPCachingAllocator and the
  library exports c10::hip::HIPCachingAllocator::allocator (nm -D
  libc10_hip.so). The wheel still ships the VANILLA CUDA-named header, so
  lantern's AllocatorCuda.cpp:4 / Cuda.cpp:9 compile the inline get()
  wrappers that reference c10::cuda::CUDACachingAllocator::allocator -- a
  symbol no wheel library defines (ldd -r reports exactly this one).
  Fix: ensure_hip_compat() stages c10/cuda/CUDACachingAllocator.h, which
  redirects the namespace to c10::hip::HIPCachingAllocator (a
  using-directive: qualified lookup of
  c10::cuda::CUDACachingAllocator::getDeviceStats/emptyCache then calls the
  HIP inline wrappers, whose allocator is the wheel's real object). The
  shim also keeps the transitive surface the original header used to
  provide (c10/cuda/CUDAException.h for C10_CUDA_CHECK, cuda_runtime_api.h
  for cudaGetDevice/cudaMemGetInfo).
  No source patch needed; the shim is first on lantern's include path and is
  a no-op for CUDA/CPU trees (hip-compat is only added for a ROCm wheel).

CHANGELOG v20 -- fix the v19 field failure at 56% in make (src/Cuda.cpp):
    ATen/cuda/CUDAContextLight.h:95: error: 'cusolverDnHandle_t' does not name
      a type; did you mean 'hipsolverDnHandle_t'?
  Ground truth (headers pulled straight out of the torch 2.8.0+rocm6.4 wheel
  by HTTP range requests): the wheel installs the VANILLA CUDA-named ATen/cuda
  tree, whose cusolver surface is
    L15-17: "#ifdef CUDART_VERSION #include <cusolverDn.h> #endif" -- DEAD on
            HIP (the v16 shims define no CUDART_VERSION), so the v16
            cusolverDn.h shim carrying the cusolverDnHandle_t alias is never
            reached;
    L23-25: "#if defined(USE_ROCM) #include <hipsolver/hipsolver.h> #endif"
            -- ACTIVE (the kit pins -DUSE_ROCM): declares hipsolverDnHandle_t
            but no cusolver alias (hence the gcc suggestion);
    L94-96: "#if defined(CUDART_VERSION) || defined(USE_ROCM)
            cusolverDnHandle_t getCurrentCUDASolverDnHandle();" -- ACTIVE
            under the pinned USE_ROCM, needs cusolverDnHandle_t from nowhere
            the wheel provides. (v19 believed include AND declaration were
            both dead under CUDART_VERSION; 2.8.0 guards the declaration with
            "CUDART_VERSION || USE_ROCM".) Same story in
            ATen/cuda/Exceptions.h:112 (cusolverGetErrorMessage takes a
            cusolverStatus_t; its cusolver_common.h include is the dead
            !USE_ROCM branch).
  Fix: the cusparse.h shim -- included BEFORE <hipsolver/hipsolver.h> by BOTH
  torch headers (CUDAContextLight.h:8, Exceptions.h:4) -- now declares the
  two cusolver types self-contained (struct cusolverDnContext* / int: plain
  pointer + int, ABI-identical to hipsolverDnHandle_t/hipsolverStatus_t, no
  hipsolver-tag coupling, no ordering hazard). The cusolverDn.h shim aliases
  onto the hipsolver types only when reached standalone (macro guard).
  Also:
  (a) cuda_runtime_api.h shim: + cudaMemGetInfo -> hipMemGetInfo. Ground
      truth: lantern src/AllocatorCuda.cpp:26 calls it inside C10_CUDA_CHECK;
      that TU was never reached by an earlier round, so the 26-token map
      (built from {CUDAContext.h, CUDAHooks.h, CUDACachingAllocator.h})
      missed it.
  (b) NEW shims cuda_runtime.h (forwards to the api shim) and library_types.h
      (cudaDataType + CUDA_R_*/CUDA_C_*): both included by wheel headers
      adjacent to the closure (ATenCUDAGeneral.h/CUDADevice.h, CUDADataType.h)
      -- insurance, not needed by today's lantern closure.
  (c) flat-SDK bridge: torch 2.8.0 includes <hipsolver/hipsolver.h>
      PER-COMPONENT (CUDAContextLight.h:24, Exceptions.h:10); on ROCm <=6.x
      flat SDKs (no hipsolver/ dir) the snapshot mirrors the flat
      hipsolver.h into hip-compat/hipsolver/hipsolver.h so the include
      resolves on every SDK generation. hipsolver.h is now a HARD
      requirement like hipsparse/hipblas/hipblaslt.
  (d) the compat probe is now FATAL, not informational (opt-out:
      LANTERN_ROCM_KIT_PROBE_WARN=1). This failure class compiles fine for
      every lantern TU except Cuda.cpp and dies ~10 min into make; the probe
      produces the same g++ error in seconds. The probe also covers
      ATen/cuda/Exceptions.h (the cusolverStatus_t site) and
      torch/csrc/cuda/memory_snapshot.h (AllocatorCuda's closure).
  Validation: full syntax sweep of ALL lantern TUs (torch main src/lantern,
  40 files) against the REAL wheel include tree (torch 2.8.0+rocm6.4, 9437
  headers fetched into the sandbox by range requests): 40/40 compile clean
  with the v20 shims; Cuda.cpp reproduces the v19 failure with the v19 set.

CHANGELOG v19 -- fix the v18 field failure at the flags gate (host: ROCm 7.2.4
  SDK complete, ROCM_PATH set, every header resolves): "the torch cuda-header
  closure does NOT compile with the real lantern flags" -- flags-probe.log:
    ATen/cuda/Exceptions.h:59: error: 'cublasStatus_t' was not declared in this
      scope; did you mean 'hipblasStatus_t'?
    ATen/cuda/Exceptions.h:71: error: 'cusparseStatus_t' was not declared in
      this scope; did you mean 'hipsparseStatus_t'?
  Root cause: torch 2.8.0's ATen/cuda/Exceptions.h (pulled by
  ATen/cuda/CUDAContext.h:9) DECLARES _cublasGetErrorEnum(cublasStatus_t) and
  cusparseGetErrorString(cusparseStatus_t) -- cublas*/cusparse* STATUS types
  the v16 shims never aliased (they mapped the four HANDLE types only).
  torch's own ROCm build hipifies those declarations away; a downstream g++
  build hits them raw. Fix: the cublas_v2.h shim now also aliases
  cublasStatus_t -> hipblasStatus_t and the cusparse.h shim
  cusparseStatus_t -> hipsparseStatus_t, plus the paired
  CUBLAS_STATUS_SUCCESS / CUSPARSE_STATUS_SUCCESS defines (insurance for the
  AT_CUBLAS_CHECK / AT_CUSPARSE_CHECK macros of the same header, which expand
  the type and the success constant together in downstream TUs).
  Also:
  (a) the self-heal budget rises 15 -> 50 iterations. Field evidence: the
      ROCm 7.2.4 hip/ runtime closure revealed 15 headers ONE BY ONE (g++
      stops at the first missing include, so the closure fills one header per
      probe run) and EXHAUSTED the old budget: the informational probe never
      reached Exceptions.h and always printed "compat probe still failing"
      ending on hip/library_types.h -- the 15th fix, applied AFTER the last
      probe run of the loop. A budget-exhausted state now prints its own
      explicit warning instead of the generic one.
  (b) the fatal flags-gate message now prints the FIRST g++ errors INLINE --
      a residual failure is self-diagnosing instead of requiring another
      `cat flags-probe.log` round-trip.
  (c) the unit harness (scripts/test_hip_compat_v19.sh) now mirrors the REAL
      2.8.0 closure: its fake wheel ships an ATen/cuda/Exceptions.h with the
      cublasStatus_t/cusparseStatus_t declarations. The v18 harness lacked
      that file, which is exactly why the sandbox run passed while the field
      gate failed. The aliases are regression-tested against the REAL ROCm
      7.2.4 headers (hipblas 3.2.0 / hipsparse 4.2.0), and a negative test
      proves the harness would catch the bug.
  Note: the manual workaround of copying /opt/rocm*/include into hip-compat
  is neither needed nor persistent -- ensure_hip_compat() wipes and rebuilds
  the directory on every run; the self-heal now fills everything the probe
  reveals, and the real lantern flags also carry the SDK include dirs via
  torch's imported HIP targets.

CHANGELOG v18 -- fix the v17 "no HIP SDK headers found" false failure on
  ROCm 7.x hosts WITH a complete SDK (field: rocm-hip-sdk 7.2.4 installed,
  /opt/rocm-7.2.4/include present, hipsparse.h still "not found"). ROCm 7.x
  moved the math headers into per-component subdirs: hipblas-dev 3.2.0
  ships include/hipblas/hipblas.h, hipsparse-dev 4.2.0
  include/hipsparse/hipsparse.h, hipblaslt-dev 1.2.2
  include/hipblaslt/hipblaslt.h, hipsolver-dev 3.2.0
  include/hipsolver/hipsolver.h -- there is NO flat <name>.h at the include
  root anymore, and the v17 sentinel (-f $cand/hipsparse.h) matched
  nothing. ensure_hip_compat() now:
  (a) locates hipsparse.h RECURSIVELY (find -H -- /opt/rocm/include is
      usually a symlink) under each candidate include root; flat ROCm 6.x
      layouts still win through the direct file test;
  (b) snapshots each forwarded header via sdk_snap(): flat layout -> direct
      copy (as v17); per-component layout -> a 1-line flat WRAPPER
      ("#include <hipsparse/hipsparse.h>") + a recursive copy of the whole
      component dir into hip-compat, so every relative internal include
      ("internal/precond/...", "hipsparse-types.h") and every
      cross-component include (hipblaslt.h 1.2.2 pulls <hipblas/hipblas.h>
      and <hipblas-common/hipblas-common.h>) resolves INSIDE hip-compat
      with zero self-heal iterations;
  (c) hipblas-common.h joins hipsolver.h as best-effort snapshot (new
      hipblas 3.2.0 / hipblaslt 1.2.2 dependency);
  (d) the die messages name the exact apt packages (hipblas-dev,
      hipsparse-dev, hipblaslt-dev -- all pulled by rocm-hip-sdk) instead
      of the misleading generic rocm-dev suggestion.
  Verified against the REAL ROCm 7.2.4 header tree (packages pulled from
  repo.radeon.com/rocm/apt/7.2.4): the wrappers compile the full
  compat-probe with -fsyntax-only; the ROCm 6.x flat path stays covered by
  the unit harness (scripts/test_hip_compat_v18.sh).

CHANGELOG v17 -- fix the v16 compile failure at 51% (src/Cuda.cpp.o): the six
  shims resolved, but hip-compat/cusparse.h:5 died on "#include <hipsparse.h>:
  No such file or directory". Split include surface: the wheel BUNDLES the
  hip/ RUNTIME headers under libtorch/include/hip/ (that is why the v16
  cuda_runtime_api.h shim compiled -- its <hip/hip_runtime_api.h> resolved
  from the torch include dirs, one line BEFORE the failing cusparse.h), but
  NO math-library headers: hipsparse.h, hipblas.h, hipblaslt.h, hipsolver.h
  exist only in a system HIP SDK, which the kit deliberately keeps off
  lantern's include path (hermetic link policy; adding /opt/rocm/include to
  the target would also let its hip/ headers shadow the wheel's bundled 6.4
  ones). The v14/v15 runs never reached the math headers (platform macro
  first, then the NVIDIA typedef storm), so the hole only opened once both
  were fixed. ensure_hip_compat() now closes it in four steps:
  (a) locate the host HIP SDK headers: ROCM_PATH/HIP_PATH, /opt/rocm,
      /opt/rocm-* (newest first), /usr/include -- the first dir holding
      hipsparse.h wins; none -> die with the install one-liner;
  (b) SNAPSHOT the forwarded math headers into build-lantern/hip-compat/
      (already FIRST on lantern's include path): ordering-independent, never
      shadows a wheel header, and compile-time ONLY (opaque handles -- the
      wheel's runtime chain stays authoritative, nothing new gets linked);
      hipsparse/hipblas/hipblaslt are hard requirements (torch 2.8.0 includes
      <cusparse.h>/<cublas_v2.h>/<cublasLt.h> unguarded), hipsolver.h is
      best-effort (its shim only fires under CUDART_VERSION, dead on HIP);
  (c) SELF-HEAL: g++ -fsyntax-only probe of the exact torch 2.8.0 cuda-header
      closure (six shims + ATen/cuda/CUDAContext.h + detail/CUDAHooks.h +
      c10/cuda/{CUDAFunctions,CUDAStream,CUDACachingAllocator}.h) with the
      hard-coded shim defines; every residual "fatal error: X.h" is cured by
      copying X.h from the SDK into hip-compat (path preserved), retrying --
      the math headers pull more SDK headers transitively (roc* family,
      hip/hip_version.h); unknown headers or non-missing-header errors are
      left to the second gate;
  (d) post-configure, the flags gate now ALSO syntax-compiles the probe with
      the REAL flags.make -D/-I/-std set: any residual breakage dies BEFORE
      make with the full g++ output instead of at 51%. Artifacts for manual
      reruns: hip-compat/compat-probe.cpp, compat-probe.log, flags-probe.log.

CHANGELOG v16 -- fix the v15 compile failure at 51%: "conflicting declaration
  'using uchar1'" (x28) + "redefinition of 'struct dim3'" + "cannot convert
  'cudaMemcpyKind' to 'hipMemcpyKind'" in src/Cuda.cpp.o -> gmake Error 2.
  Root cause: v15 enables the REAL cuda_* implementations (LANTERN_HIP), so
  Cuda.cpp now compiles <ATen/cuda/CUDAContext.h> +
  <c10/cuda/CUDACachingAllocator.h> etc. The wheel installs torch's VANILLA
  headers: ATen/cuda/CUDAContextLight.h includes <cuda_runtime_api.h>,
  <cusparse.h>, <cublas_v2.h>, <cublasLt.h> UNGUARDED (lines 7-13),
  c10/cuda/CUDAStream.h:3 and CUDAFunctions.h:14 include
  <cuda_runtime_api.h>, CUDAException.h:9 includes <cuda.h>. torch's own
  ROCm build never compiles those as-is (tools/amd_build hipifies its
  sources into build-dir copies first), but a downstream g++ build hits
  them raw -- while the USE_ROCM branches of the same TU pull the HIP
  headers from the system SDK (hipsolver -> hip_complex ->
  amd_hip_vector_types). Two outcomes, both fatal:
  (a) host with NVIDIA CUDA headers in /usr/include (this machine): NVIDIA
      typedefs (uchar1/uchar2/.../dim3/cudaMemcpyKind) collide with the HIP
      ones -> the conflict storm; CUDAFunctions.h:94 (the TORCH_HIP_VERSION
      >= 301 branch) then feeds NVIDIA's cudaMemcpyKind to
      hipMemcpyWithStream;
  (b) host without NVIDIA headers: "cuda_runtime_api.h: No such file or
      directory".
  Fix -- ensure_hip_compat() stages build-lantern/hip-compat/, six
  cuda->HIP shim headers shadowing the unguarded includes (cuda.h,
  cuda_runtime_api.h, cusparse.h, cublas_v2.h, cublasLt.h, cusolverDn.h):
  cuda_runtime_api.h includes <hip/hip_runtime_api.h> and maps the EXACT
  26-token cuda* closure of the torch 2.8.0 headers lantern compiles
  (cudaError_t/cudaStream_t/cudaEvent_t/cudaMemcpyKind/cudaDeviceProp/
  cudaStreamCaptureStatus... -> hipError_t/hipStream_t/hipEvent_t/
  hipMemcpyKind/hipDeviceProp_t/hipStreamCaptureStatus... -- the capture
  enum members hipStreamCaptureStatusNone/Active/Invalidated match torch's
  static_asserts, verified against ROCm 6.4.4 sources), and the math shims
  alias the handles the ROCm headers do NOT provide (hipsparse.h ships no
  cusparseHandle_t, hipblas.h no cublasHandle_t, hipblaslt.h no
  cublasLtHandle_t, hipSOLVER's dense handle is hipsolverDnHandle_t). The
  hip-shim block puts that dir FIRST on lantern's include path
  (target_include_directories BEFORE) and pins USE_ROCM explicitly (the
  wheel's torch config exports it, but the target no longer depends on
  that). A post-configure check greps flags.make for -DUSE_ROCM + the
  hip-compat dir, so a silent injection miss dies immediately instead of
  resurfacing as the conflict storm at 51%. patch_cuda_hip also (void)-casts
  the nodiscard hipRuntimeGetVersion call (warning noise only).

CHANGELOG v15 -- the kit BUILDS and the GPU chain works end-to-end (field:
  tensor on cuda:0 OK). This release closes the remaining functional gaps
  reported from RStudio, all upstream {torch} properties that a ROCm wheel
  build must compensate for:
  (1) src/lantern/src/Cuda.cpp guards every real cuda_* implementation with
      "#ifdef __NVCC__", which torch's CMakeLists sets ONLY for $ENV{CUDA}
      nvcc builds. On a ROCm wheel build the functions exist but compile to
      their #else stubs that THROW at runtime: cuda_get_device_capability,
      cuda_memory_stats/summary/snapshot, cuda_runtime_version,
      cuda_empty_cache ("`cuda_get_device` is only supported on CUDA
      runtimes." etc). All of them are plain host-side calls the wheel's
      torch_hip/c10_hip libraries implement (at::cuda::getDeviceProperties,
      c10::cuda::CUDACachingAllocator, torch::cuda::_memory_snapshot_pickled).
      patch_cuda_hip() widens the 7 guards to "#if defined(__NVCC__) ||
      defined(LANTERN_HIP)", switches the only CUDA-only symbol
      (cudaRuntimeGetVersion) to hipRuntimeGetVersion, fixes the
      copy-pasted error message of _lantern_cuda_get_runtime_version (it
      claimed "cuda_device_stats"), and rewords the messages to mention
      HIP (ROCm). LANTERN_HIP is defined by the hip-shim, gated on a ROCm
      wheel. Marked + backed up (Cuda.cpp.orig-make-rocm-kit); --unpatch
      restores. cuda_get_rng_state/set_rng_state were never guarded --
      they needed (2) instead.
  (2) torch::cuda_get_rng_state() dies with "Error in dlopen:
      libcaffe2_nvrtc.so" -- the hipified libtorch_hip dlopens that stub
      lazily (device RNG state / JIT kernel compilation). CUDA wheels ship
      it; ROCm 6.x pip wheels do NOT (newer ROCm wheels ship it again --
      tile-ai/tilelang#3711 documents the wheel-level fix landing later).
      The equivalent C API family (hiprtc*) lives in libamdhiprtc, so
      ensure_nvrtc_stub() symlinks libcaffe2_nvrtc.so -> libamdhiprtc.so in
      the staged tree when the wheel lacks its own copy. Doctor [9] probes
      the bridge; [10] checks the installed {torch} package + the install
      env vars (the startup download prompt at library(torch) is
      diagnosed there -- LANTERN_URL must be the kit layout, never
      build-lantern; TORCH_INSTALL=0 silences it).
  R-side (marked block APPENDED to R/cuda.R -- no surgery, tree-version
  proof): cuda_synchronize(device = NULL) passed -1 ("sync all devices")
  which dies inside the wheel's HIP CUDAHooks ("device=-80" assert) while
  its own roxygen doc promises cuda_current_device(); and
  cuda_runtime_version() decoded with the CUDA /1000 layout although
  hipRuntimeGetVersion returns major*1e7 + minor*1e5 + patch (ROCm 6.4 ->
  60443482 = 6.4.43482, the CUDA decode turns it into nonsense).
  Also: $CUDA is unset before cmake (lantern's CMakeLists takes a nvcc
  branch whenever it is set: enable_language(CUDA) dies without nvcc and
  would re-enable the __NVCC__ paths whose cuda* calls do not exist on
  HIP), and python3 joined the required-tools list.

CHANGELOG v14 -- first --from-venv compile: at 51% (the first TUs that
  include ATen/cuda headers) the build died TWICE, and BOTH are wheel
  properties the lantern build must compensate for:
  (1) c10/cuda/CUDAMacros.h:8: "fatal error: c10/cuda/impl/
      cuda_cmake_macros.h: No such file or directory". Torch's AMD HIP
      build does not generate that file (upstream says so in the comment
      around that include) and ROCm pip wheels ship none. Upstream provides
      the escape hatch: define C10_CUDA_NO_CMAKE_CONFIGURE_FILE. (Some
      libtorch zip layouts DO carry the file -- the define is harmless
      there.)
  (2) hip_complex.h:35 / hip_runtime_api.h:10250: "#error Must define
      exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__".
      Chain: Cuda.cpp -> ATen/cuda/CUDAContextLight.h (USE_ROCM, defined
      via the wheel's exported targets) -> <hipsolver.h> -> the SYSTEM
      ROCm 7.2 SDK headers (the wheel ships no SDK headers) -> hip/*.h
      platform guard. lantern compiles plain g++ (no hipcc), so nothing
      defines the platform macro. torch's own LoadHIP passes
      -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__ to its try_compile
      probes -- the shim now passes the same pair to the lantern target.
  v14 shim addition: target_compile_definitions(lantern PRIVATE
  __HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__ C10_CUDA_NO_CMAKE_CONFIGURE_FILE),
  gated on TARGET lantern AND the staged tree being a ROCm wheel
  (libtorch_hip.so present) so CUDA/CPU trees are untouched.
  Doctor fixes from the same field report:
  (a) the [8] matmul scan skipped EVERY rocblas/hipblaslt copy: the pip
      wheel ships REAL UNVERSIONED files (librocblas.so) and the kit adds
      the versioned names (librocblas.so.4) as SYMLINKS, so the "! -L"
      filter dropped them all; v14 resolves symlinks (readlink -f) and
      dedupes by real path.
  (b) the [8] torch-kernel arch list was full of prefix junk (gfx0, gfx1,
      gfx9, gfx11, gfx94, gfx9f) -- the bundle-id strings heuristic matched
      ids shorter than any real ISA; v14 filters to gfx + 2 digits + at
      least 1 more char (gfx900, gfx90a, gfx1030 all pass; gfx94/gfx9f die).
  (c) the staged version.py's HIP line was silently dropped (single-quoted
      values); v14 parses both quote styles and any spacing.

CHANGELOG v13 -- the loader chapter is CLOSED. With the v12 artifacts the
  first GPU smoke test allocates on cuda:0 (torch_tensor(1:12,
  device="cuda:0") prints a CUDALongType): the hermetic 6.4 chain loads,
  HSA enumerates the device, memory works. The failure moved ONE layer up:
  "* 5" now dies with "HIP error: invalid device function" -- the fat
  binaries inside the wheel's libtorch_hip.so contain NO code object for
  the device's gfx target (torch wheels are built for a FIXED
  TORCH_ROCM_ARCH list; Vega10/gfx900 is not in torch 2.8.0+rocm6.4's
  list). That is a wheel property -- no kit change can cure it. v13 makes
  --doctor say so instead of leaving the user to guess:
  [8] GPU ISA vs wheel code objects: parses the rocminfo agents, extracts
      the code-object bundle ids embedded in the staged libtorch_hip.so
      and the GEMM libraries (librocblas/libhipblaslt), prints a verdict
      per device: COVERED / HSA_OVERRIDE_GFX_VERSION=x.y.z suggestion
      (only when a same-generation sibling IS in the wheel's list;
      EXPERIMENTAL for gfx900->9.0.6: Vega20-only instructions may trap at
      runtime) / "rebuild with TORCH_ROCM_ARCH=<target>" (or the ROCm 5.7
      container for full BLAS on Vega10).
  Also: the kit copy's DLOPEN-FAIL in staging ("libtorch_cpu.so: cannot
  open" -- expected, because install_torch() drops it next to the wheel
  libs and audit_kit_layout() already proved that layout) is now ANNOTATED
  instead of triggering a pointless LD_DEBUG capture whose culprit grep
  came back empty (the confusing "who requires what:" with nothing after
  it in the first v12 field report).

CHANGELOG v12 -- the first FULL v11 field run: build OK, five NEEDED repairs,
  DT_RPATH set, closure audit OK in all three regimes -- and then the script
  DIED in the summary: "line 910: ORIGIN: unbound variable". The summary is
  an UNQUOTED heredoc and one literal '$ORIGIN' inside it got parameter-
  expanded under `set -u` -> exit 1 -> the EXIT trap QUARANTINED the freshly
  built, fully audited artifacts (inst/kit copies deleted although 3/4 and
  4/4 had already installed them). Meanwhile the doctor's hostile probe
  finally CAUGHT the chimera red-handed on the OLD installed artifact:
  LD_LIBRARY_PATH=/opt/rocm/lib -> libMIOpen.so.1 resolves to
  /opt/rocm/lib/libMIOpen.so.1 (ROCm 7.2.4 ships the SAME soname as the
  wheel!) -> it needs libamdhip64.so.7 -> which binds the ALREADY-LOADED 6.4
  libhsa -> "undefined symbol ... version ROCR_1". The v11 build artifact
  SURVIVED the very same hostile probe (DLOPEN-OK): the hermetization
  works. v12:
  1) the summary heredoc escapes \$ORIGIN (the bug that nuked a good run);
  2) FRESH_OK flag: once the gates passed and both copies are in place, a
     late failure no longer quarantines audited artifacts;
  3) the doctor's ~/.Renviron check ignores COMMENTED lines (false alarm on
     an already-fixed Renviron);
  4) a full end-to-end kit test (fake wheel zip + real cmake) that would
     have caught the heredoc bug before shipping.

CHANGELOG v11 -- doctor feedback from the field. The first real --doctor
  output proved three things: (1) R runs PERMANENTLY with
  LD_LIBRARY_PATH=/opt/rocm/lib (shell profile AND ~/.Renviron) -- the
  chimera fuel; the doctor now flags it loudly; (2) the v9 artifact's
  RUNPATH itself carried /opt/rocm-7.2.4/lib and /opt/rocm/lib (CMake link
  dirs baked in) -- v10's force-rpath already replaces it wholesale;
  (3) the last kit run had been INTERRUPTED mid-staging (partial rsync:
  dev symlinks present, big real files like libtorch_cpu.so missing), which
  made the probes fail with a misleading "libtorch_cpu.so: cannot open" and
  left the quarantine to remove the inst/kit copies. v11 doctor additions:
  staging-inventory sanity (names the interrupted-staging state instead of
  letting the probes mislead), /opt/rocm symlink resolution + hsa/MIOpen
  inventory, libtorch_hip.so in the key-rpath list (ROCm wheels ship
  torch_hip, not torch_cuda), fixed the silent "(no rpath)" branch, and a
  patience note on the 2-3 GB staging step.

CHANGELOG v10 -- the v9 gates all passed ("closure audit OK") yet R CMD
  INSTALL still died on "/opt/rocm/lib/libamdhip64.so.7: undefined symbol:
  hsa_amd_memory_get_preferred_copy_engine, version ROCR_1" with a FRESH
  liblantern: the chimera is assembled AT LOAD TIME by the ambient
  environment, not at link time. The v9 audit was structurally blind: it
  forced LD_LIBRARY_PATH=<staged> FIRST, so any soname shared between the
  wheel (6.4) and /opt/rocm (7.x) -- libhsa-runtime64.so.1, possibly
  libMIOpen / rccl / ... -- resolved to the staged copy during the audit,
  while the real R process (no override, or an LD_LIBRARY_PATH=/opt/rocm
  inherited from the shell) lets the loader take the SYSTEM 7.x copy; that
  copy needs libamdhip64.so.7, which binds against the ALREADY-LOADED 6.4
  libhsa and dies on the missing ROCR_1 symbol. Python survives the same
  environment because nothing in its chain RPATH-protects the 6.4 hsa
  first. Three cures, layered:
  1) hermetize_libs(): EVERY library of the staged tree gets DT_RPATH=$ORIGIN
     (patchelf --force-rpath). DT_RPATH has PRIORITY over LD_LIBRARY_PATH
     and over the ld.so cache, unlike DT_RUNPATH -- a hostile environment
     can no longer swap a shared soname onto the 7.x chain. The built
     liblantern gets DT_RPATH='$ORIGIN:<staged>/lib'; the kit artifact
     '$ORIGIN' alone (install_torch drops it next to the wheel libs).
  2) audit_closure() now runs in THREE regimes and fails on the first bad
     one: pristine (LD_LIBRARY_PATH unset -- what the R process actually
     does), forced (staged first -- wheel self-sufficiency), hostile
     (LD_LIBRARY_PATH=/opt/rocm/lib -- stress test of the hermetization).
     The kit artifact is audited inside a SIMULATED install layout
     (hardlinked next to the wheel libs, like install_torch() will do).
  3) --doctor: standalone, read-only diagnosis of the CURRENT install:
     environment (LD_LIBRARY_PATH, ~/.Renviron, ld.so cache), /opt/rocm
     soname-overlap inventory, RPATH-vs-RUNPATH types of the key libs,
     NEEDED/RPATH dumps of every liblantern copy, a compiled dlopen probe
     (RTLD_NOW, like dyn.load) run pristine AND hostile, plus LD_DEBUG=libs
     culprit extraction ("find library=... required by ...") for the probe
     and for a real Rscript library(torch).

CHANGELOG v9 -- fix the chimera RELOAD: "undefined symbol:
  hsa_amd_memory_get_preferred_copy_engine, version ROCR_1" came back at
  R CMD INSTALL time, even though the kit gates now catch bad links.
  Mechanism (the missing piece was on the R side): {torch}'s src/configure
  sets LANTERN_TARGET=dummylantern unless BUILD_LANTERN is exported, so a
  plain `R CMD INSTALL .` NEVER builds lantern -- it prints "*** Skip
  building lantern." and silently reinstalls the PREVIOUS
  inst/lib/liblantern.so (the kit's step 3/4 copy). When a kit run stops
  BEFORE step 3/4 (build error, gate failure), that stale artifact stays in
  place: a pre-alignment liblantern whose DIRECT NEEDED list still contains
  the system ROCm 7 math libs (libhipblas.so.3, libhipsparse.so.4,
  libhipsolver.so.1, libhipblaslt.so.1). At load time those drag
  /opt/rocm/lib/libamdhip64.so.7 into the process; it binds to the
  ALREADY-LOADED libhsa-runtime64.so.1 of the wheel's 6.4 chain (bundled in
  torch/lib), which lacks the recent ROCR_1 symbol -> crash. Python works
  because nothing there ever loads the .so.7. v9 makes the failure loud and
  the artifact flow airtight:
  1) patchelf is now a REQUIRED tool. Without it repair_needed_entries
     silently no-op'ed, the gate then stopped the kit, and the stale
     inst/lib/liblantern.so kept poisoning every R CMD INSTALL.
  2) On ANY kit failure an EXIT trap QUARANTINES inst/lib/liblantern.so and
     torch-rocm-kit/lantern/lib/liblantern.so: R CMD INSTALL then fails
     with a loud "liblantern.so missing" instead of installing a 6.4/7.2
     chimera.
  3) audit_closure(): after the build, ldd walks the FULL transitive
     closure of liblantern in the staged layout -- any dependency resolving
     into the system ROCm (or "not found") fails the kit. verify_runtime_deps
     only sees DIRECT NEEDED entries; this catches everything else.
  4) The summary prints liblantern's final direct NEEDED list (eyeball
     check without readelf).

CHANGELOG v8 -- fix the next gate failure: "unresolved direct dependencies of
  liblantern: libhipblas.so.3 libhipsparse.so.4 libhipsolver.so.1
  libhipblaslt.so.1". The v4 shim re-aligned the HIP RUNTIME (libamdhip64)
  but torch's exported HIP interfaces still dragged the SYSTEM roc* math
  libraries (ROCm 7.x sonames: hipblas .so.3, hipsparse .so.4, hipsolver
  .so.1, hipblaslt .so.1) into lantern's direct DT_NEEDED, while the wheel
  ships its own sonames (hipblas .so.2, hipsparse .so.1, hipsolver .so.0,
  hipblaslt .so.0). /opt/rocm being in lantern's RUNPATH, the loader would
  have silently MIXED 7.x math libs with the wheel's 6.4 chain (the v4
  crash class, reborn). The shim now aligns ALL of them with three
  complementary CMake mechanisms (whichever one the interface uses, one
  hits) plus a bash-side last resort:
  a) target_link_directories(lantern BEFORE "${TORCH_PATH}/lib") -- plain
     -l<name> items resolve to the staged wheel copies first;
  b) bare-name imported targets (hipblas, hipsparse, ...) created or
     redirected to the wheel files -- CMake substitutes them into plain-name
     interface items at generate time;
  c) a sweep of every IMPORTED target still pointing outside the wheel
     (roc::X, hip::X, full-path locations): re-aligned to the same-family
     library staged from the wheel;
  d) repair_needed_entries() -- after the build, any direct NEEDED still
     missing from the staging dir is rewritten onto the wheel's SONAME with
     'patchelf --replace-needed' (no-op without patchelf). The
     verify_runtime_deps gate then has the final word.

CHANGELOG v7 -- fix "libamdhip64.so.6: cannot open shared object file" that
  SURVIVED v6 even though the RUNPATH was correct. Two distinct defects:
  1) Link time vs load time: the linker only reads the SONAME stored INSIDE
     the ELF it links (DT_NEEDED = libamdhip64.so.6), but dlopen() must find
     a FILE with that exact name in the RUNPATH dirs. The staged wheel dir
     ships the HIP runtime under another name (e.g. libamdhip64.so.6.4.43482
     and/or libamdhip64.so) while every OTHER wheel lib happens to carry its
     exact SONAME as file name -- hence the single, very confusing failure.
     New ensure_soname_links() scans the staged lib dir, reads each real
     library's SONAME (objdump -p) and creates/repairs the matching symlink.
     New verify_runtime_deps() then re-checks, after the build, that EVERY
     direct NEEDED of liblantern exists BY EXACT NAME in the staged dir:
     the failure is now caught at build time with an explicit message
     instead of a cryptic R CMD INSTALL error.
  2) On localized systems (fr_FR) the readelf guards were silent no-ops:
     French readelf prints "Bibliotheque partagee : [...]" so the sed
     patterns matching 'Shared library:' / 'Library runpath:' never matched.
     The script now exports LC_ALL=C (all binutils output is parsed in the
     C locale).
  Also: the 4/4 kit artifact gets RUNPATH='$ORIGIN' (patchelf, optional) so
  install_torch() can drop it next to the libtorch bundle, anywhere.

CHANGELOG v6 -- fix at load time from an INSTALLED package: "libamdhip64.so.6:
  cannot open shared object file". The v4 re-pointing worked (DT_NEEDED is
  now the wheel's libamdhip64.so.6) but the loader failed to FIND it when
  liblantern.so is dlopen'ed from the installed library directory: the
  lookup goes through the RUNPATH of liblantern, which must explicitly point
  at the staged libtorch dir. v6 passes explicit CMAKE_BUILD_RPATH /
  CMAKE_INSTALL_RPATH so the RUNPATH is guaranteed in the built binary,
  verifies the RUNPATH with readelf after the build, and documents two cures
  in the summary (LD_LIBRARY_PATH through ~/.Renviron, or an ld.so.conf.d
  entry).
CHANGELOG v5 -- English translation of all comments and messages
  (upstream-ready, ASCII-only). No logic change vs v4.

CHANGELOG v4 -- fix at load time: "/opt/rocm/lib/libamdhip64.so.7: undefined
  symbol: hsa_amd_memory_get_preferred_copy_engine, version ROCR_1".
  Diagnosis: the build succeeded, but liblantern got linked against the
  SYSTEM HIP runtime (hip::amdhip64 defined by hip-config.cmake 7.2 ->
  /opt/rocm) while libc10_hip/libtorch_cuda from the wheel embed the 6.4
  chain. At dlopen time, the libhsa-runtime64 of the 6.4 chain is already
  loaded when the .so.7 arrives and demands a recent ROCR_1 symbol that this
  older libhsa does not export -> failure. Python works because nothing
  there ever loads the .so.7. v4 REWRITES the IMPORTED_LOCATION of the
  hip::{amdhip64,host,device} targets to the libamdhip64 EMBEDDED in the
  wheel (${TORCH_PATH}/lib) -- exactly the runtime already validated on the
  Python side -- and adds a post-build readelf guard (die if libamdhip64.so.7
  remains linked).

CHANGELOG v3 -- fix "Some (but not all) targets in this export set were
  already defined (hip::amdhip64)":
  v2 defined hip::amdhip64 BEFORE find_package(Torch) (through
  CMAKE_PROJECT_INCLUDE / CMAKE_PROJECT_TOP_LEVEL_INCLUDES, executed right
  after project()). But on ROCm 7.x the modern hip-config.cmake -- found
  without any trouble by torch's LoadHIP.cmake (HIP 7.2.53211 on the target
  machine) -- defines the WHOLE export set hip::amdhip64 + hip::host +
  hip::device by itself; the pre-existing target made the export set
  partially defined => configure aborted. The v2 premise ("legacy FindHIP,
  version 0.0.0") does not apply to libtorch 2.8's LoadHIP, which goes
  through CONFIG mode.
  v3 stopped injecting anything up-front. The shim became a conditional
  fallback ADDED AFTER find_package(Torch) in <lantern-src>/CMakeLists.txt
  (marked block, idempotent, removed by --unpatch): a no-op when torch
  already defined the targets (system ROCm present), a life-saver on
  "wheel-only" machines without /opt/rocm -- the common case for torch R
  users, relevant for the upstream follow-up of #907.
