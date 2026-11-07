# make_rocm_kit.sh -- CHANGELOG

Full design notes, one section per version (newest first).

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
