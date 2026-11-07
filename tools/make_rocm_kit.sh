#!/usr/bin/env bash
# =============================================================================
# make_rocm_kit.sh v16 -- build liblantern with ROCm, aligned with the {torch}
# dev tree (issue mlverse/torch#907)
#
# Put this script in <torch-tree>/tools/ (or pass --tree <root>).
# It feeds the directories ALREADY present in the tree:
#
#   build-lantern/libtorch/      <- staged ROCm libtorch (default location
#                                   expected by lantern's CMake: TORCH_PATH
#                                   computed from CMAKE_CURRENT_BINARY_DIR)
#   build-lantern/               <- lantern cmake build dir (cache purged)
#   inst/lib/liblantern.so       <- result copied for dev loading
#                                   (devtools::load_all() reads inst/)
#   torch-rocm-kit/lantern/      <- staged in install.R format (lib/liblantern.so)
#                                   for a clean torch::install_torch()
#
# Usage:
#   ./tools/make_rocm_kit.sh --from-venv ~/python/vision/.venv
#   ./tools/make_rocm_kit.sh --from-zip  /path/to/libtorch-...+rocm6.4.zip
#   ./tools/make_rocm_kit.sh --unpatch    (removes the hip-shim block from CMakeLists)
#   ./tools/make_rocm_kit.sh --doctor     (loader culprit + GPU-ISA vs wheel-arch verdict)
#   options: --tree <torch-root>     (default: auto-detected from tools/)
#            --lantern-src <dir>     (default: <tree>/src/lantern; the legacy
#                                     top-level lantern/ of platform/amd_gpu is
#                                     obsolete on the upstream side)
#
# CHANGELOG v16 -- fix the v15 compile failure at 51%: "conflicting declaration
#   'using uchar1'" (x28) + "redefinition of 'struct dim3'" + "cannot convert
#   'cudaMemcpyKind' to 'hipMemcpyKind'" in src/Cuda.cpp.o -> gmake Error 2.
#   Root cause: v15 enables the REAL cuda_* implementations (LANTERN_HIP), so
#   Cuda.cpp now compiles <ATen/cuda/CUDAContext.h> +
#   <c10/cuda/CUDACachingAllocator.h> etc. The wheel installs torch's VANILLA
#   headers: ATen/cuda/CUDAContextLight.h includes <cuda_runtime_api.h>,
#   <cusparse.h>, <cublas_v2.h>, <cublasLt.h> UNGUARDED (lines 7-13),
#   c10/cuda/CUDAStream.h:3 and CUDAFunctions.h:14 include
#   <cuda_runtime_api.h>, CUDAException.h:9 includes <cuda.h>. torch's own
#   ROCm build never compiles those as-is (tools/amd_build hipifies its
#   sources into build-dir copies first), but a downstream g++ build hits
#   them raw -- while the USE_ROCM branches of the same TU pull the HIP
#   headers from the system SDK (hipsolver -> hip_complex ->
#   amd_hip_vector_types). Two outcomes, both fatal:
#   (a) host with NVIDIA CUDA headers in /usr/include (this machine): NVIDIA
#       typedefs (uchar1/uchar2/.../dim3/cudaMemcpyKind) collide with the HIP
#       ones -> the conflict storm; CUDAFunctions.h:94 (the TORCH_HIP_VERSION
#       >= 301 branch) then feeds NVIDIA's cudaMemcpyKind to
#       hipMemcpyWithStream;
#   (b) host without NVIDIA headers: "cuda_runtime_api.h: No such file or
#       directory".
#   Fix -- ensure_hip_compat() stages build-lantern/hip-compat/, six
#   cuda->HIP shim headers shadowing the unguarded includes (cuda.h,
#   cuda_runtime_api.h, cusparse.h, cublas_v2.h, cublasLt.h, cusolverDn.h):
#   cuda_runtime_api.h includes <hip/hip_runtime_api.h> and maps the EXACT
#   26-token cuda* closure of the torch 2.8.0 headers lantern compiles
#   (cudaError_t/cudaStream_t/cudaEvent_t/cudaMemcpyKind/cudaDeviceProp/
#   cudaStreamCaptureStatus... -> hipError_t/hipStream_t/hipEvent_t/
#   hipMemcpyKind/hipDeviceProp_t/hipStreamCaptureStatus... -- the capture
#   enum members hipStreamCaptureStatusNone/Active/Invalidated match torch's
#   static_asserts, verified against ROCm 6.4.4 sources), and the math shims
#   alias the handles the ROCm headers do NOT provide (hipsparse.h ships no
#   cusparseHandle_t, hipblas.h no cublasHandle_t, hipblaslt.h no
#   cublasLtHandle_t, hipSOLVER's dense handle is hipsolverDnHandle_t). The
#   hip-shim block puts that dir FIRST on lantern's include path
#   (target_include_directories BEFORE) and pins USE_ROCM explicitly (the
#   wheel's torch config exports it, but the target no longer depends on
#   that). A post-configure check greps flags.make for -DUSE_ROCM + the
#   hip-compat dir, so a silent injection miss dies immediately instead of
#   resurfacing as the conflict storm at 51%. patch_cuda_hip also (void)-casts
#   the nodiscard hipRuntimeGetVersion call (warning noise only).
#
# CHANGELOG v15 -- the kit BUILDS and the GPU chain works end-to-end (field:
#   tensor on cuda:0 OK). This release closes the remaining functional gaps
#   reported from RStudio, all upstream {torch} properties that a ROCm wheel
#   build must compensate for:
#   (1) src/lantern/src/Cuda.cpp guards every real cuda_* implementation with
#       "#ifdef __NVCC__", which torch's CMakeLists sets ONLY for $ENV{CUDA}
#       nvcc builds. On a ROCm wheel build the functions exist but compile to
#       their #else stubs that THROW at runtime: cuda_get_device_capability,
#       cuda_memory_stats/summary/snapshot, cuda_runtime_version,
#       cuda_empty_cache ("`cuda_get_device` is only supported on CUDA
#       runtimes." etc). All of them are plain host-side calls the wheel's
#       torch_hip/c10_hip libraries implement (at::cuda::getDeviceProperties,
#       c10::cuda::CUDACachingAllocator, torch::cuda::_memory_snapshot_pickled).
#       patch_cuda_hip() widens the 7 guards to "#if defined(__NVCC__) ||
#       defined(LANTERN_HIP)", switches the only CUDA-only symbol
#       (cudaRuntimeGetVersion) to hipRuntimeGetVersion, fixes the
#       copy-pasted error message of _lantern_cuda_get_runtime_version (it
#       claimed "cuda_device_stats"), and rewords the messages to mention
#       HIP (ROCm). LANTERN_HIP is defined by the hip-shim, gated on a ROCm
#       wheel. Marked + backed up (Cuda.cpp.orig-make-rocm-kit); --unpatch
#       restores. cuda_get_rng_state/set_rng_state were never guarded --
#       they needed (2) instead.
#   (2) torch::cuda_get_rng_state() dies with "Error in dlopen:
#       libcaffe2_nvrtc.so" -- the hipified libtorch_hip dlopens that stub
#       lazily (device RNG state / JIT kernel compilation). CUDA wheels ship
#       it; ROCm 6.x pip wheels do NOT (newer ROCm wheels ship it again --
#       tile-ai/tilelang#3711 documents the wheel-level fix landing later).
#       The equivalent C API family (hiprtc*) lives in libamdhiprtc, so
#       ensure_nvrtc_stub() symlinks libcaffe2_nvrtc.so -> libamdhiprtc.so in
#       the staged tree when the wheel lacks its own copy. Doctor [9] probes
#       the bridge; [10] checks the installed {torch} package + the install
#       env vars (the startup download prompt at library(torch) is
#       diagnosed there -- LANTERN_URL must be the kit layout, never
#       build-lantern; TORCH_INSTALL=0 silences it).
#   R-side (marked block APPENDED to R/cuda.R -- no surgery, tree-version
#   proof): cuda_synchronize(device = NULL) passed -1 ("sync all devices")
#   which dies inside the wheel's HIP CUDAHooks ("device=-80" assert) while
#   its own roxygen doc promises cuda_current_device(); and
#   cuda_runtime_version() decoded with the CUDA /1000 layout although
#   hipRuntimeGetVersion returns major*1e7 + minor*1e5 + patch (ROCm 6.4 ->
#   60443482 = 6.4.43482, the CUDA decode turns it into nonsense).
#   Also: $CUDA is unset before cmake (lantern's CMakeLists takes a nvcc
#   branch whenever it is set: enable_language(CUDA) dies without nvcc and
#   would re-enable the __NVCC__ paths whose cuda* calls do not exist on
#   HIP), and python3 joined the required-tools list.
#
# CHANGELOG v14 -- first --from-venv compile: at 51% (the first TUs that
#   include ATen/cuda headers) the build died TWICE, and BOTH are wheel
#   properties the lantern build must compensate for:
#   (1) c10/cuda/CUDAMacros.h:8: "fatal error: c10/cuda/impl/
#       cuda_cmake_macros.h: No such file or directory". Torch's AMD HIP
#       build does not generate that file (upstream says so in the comment
#       around that include) and ROCm pip wheels ship none. Upstream provides
#       the escape hatch: define C10_CUDA_NO_CMAKE_CONFIGURE_FILE. (Some
#       libtorch zip layouts DO carry the file -- the define is harmless
#       there.)
#   (2) hip_complex.h:35 / hip_runtime_api.h:10250: "#error Must define
#       exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__".
#       Chain: Cuda.cpp -> ATen/cuda/CUDAContextLight.h (USE_ROCM, defined
#       via the wheel's exported targets) -> <hipsolver.h> -> the SYSTEM
#       ROCm 7.2 SDK headers (the wheel ships no SDK headers) -> hip/*.h
#       platform guard. lantern compiles plain g++ (no hipcc), so nothing
#       defines the platform macro. torch's own LoadHIP passes
#       -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__ to its try_compile
#       probes -- the shim now passes the same pair to the lantern target.
#   v14 shim addition: target_compile_definitions(lantern PRIVATE
#   __HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__ C10_CUDA_NO_CMAKE_CONFIGURE_FILE),
#   gated on TARGET lantern AND the staged tree being a ROCm wheel
#   (libtorch_hip.so present) so CUDA/CPU trees are untouched.
#   Doctor fixes from the same field report:
#   (a) the [8] matmul scan skipped EVERY rocblas/hipblaslt copy: the pip
#       wheel ships REAL UNVERSIONED files (librocblas.so) and the kit adds
#       the versioned names (librocblas.so.4) as SYMLINKS, so the "! -L"
#       filter dropped them all; v14 resolves symlinks (readlink -f) and
#       dedupes by real path.
#   (b) the [8] torch-kernel arch list was full of prefix junk (gfx0, gfx1,
#       gfx9, gfx11, gfx94, gfx9f) -- the bundle-id strings heuristic matched
#       ids shorter than any real ISA; v14 filters to gfx + 2 digits + at
#       least 1 more char (gfx900, gfx90a, gfx1030 all pass; gfx94/gfx9f die).
#   (c) the staged version.py's HIP line was silently dropped (single-quoted
#       values); v14 parses both quote styles and any spacing.
#
# CHANGELOG v13 -- the loader chapter is CLOSED. With the v12 artifacts the
#   first GPU smoke test allocates on cuda:0 (torch_tensor(1:12,
#   device="cuda:0") prints a CUDALongType): the hermetic 6.4 chain loads,
#   HSA enumerates the device, memory works. The failure moved ONE layer up:
#   "* 5" now dies with "HIP error: invalid device function" -- the fat
#   binaries inside the wheel's libtorch_hip.so contain NO code object for
#   the device's gfx target (torch wheels are built for a FIXED
#   TORCH_ROCM_ARCH list; Vega10/gfx900 is not in torch 2.8.0+rocm6.4's
#   list). That is a wheel property -- no kit change can cure it. v13 makes
#   --doctor say so instead of leaving the user to guess:
#   [8] GPU ISA vs wheel code objects: parses the rocminfo agents, extracts
#       the code-object bundle ids embedded in the staged libtorch_hip.so
#       and the GEMM libraries (librocblas/libhipblaslt), prints a verdict
#       per device: COVERED / HSA_OVERRIDE_GFX_VERSION=x.y.z suggestion
#       (only when a same-generation sibling IS in the wheel's list;
#       EXPERIMENTAL for gfx900->9.0.6: Vega20-only instructions may trap at
#       runtime) / "rebuild with TORCH_ROCM_ARCH=<target>" (or the ROCm 5.7
#       container for full BLAS on Vega10).
#   Also: the kit copy's DLOPEN-FAIL in staging ("libtorch_cpu.so: cannot
#   open" -- expected, because install_torch() drops it next to the wheel
#   libs and audit_kit_layout() already proved that layout) is now ANNOTATED
#   instead of triggering a pointless LD_DEBUG capture whose culprit grep
#   came back empty (the confusing "who requires what:" with nothing after
#   it in the first v12 field report).
#
# CHANGELOG v12 -- the first FULL v11 field run: build OK, five NEEDED repairs,
#   DT_RPATH set, closure audit OK in all three regimes -- and then the script
#   DIED in the summary: "line 910: ORIGIN: unbound variable". The summary is
#   an UNQUOTED heredoc and one literal '$ORIGIN' inside it got parameter-
#   expanded under `set -u` -> exit 1 -> the EXIT trap QUARANTINED the freshly
#   built, fully audited artifacts (inst/kit copies deleted although 3/4 and
#   4/4 had already installed them). Meanwhile the doctor's hostile probe
#   finally CAUGHT the chimera red-handed on the OLD installed artifact:
#   LD_LIBRARY_PATH=/opt/rocm/lib -> libMIOpen.so.1 resolves to
#   /opt/rocm/lib/libMIOpen.so.1 (ROCm 7.2.4 ships the SAME soname as the
#   wheel!) -> it needs libamdhip64.so.7 -> which binds the ALREADY-LOADED 6.4
#   libhsa -> "undefined symbol ... version ROCR_1". The v11 build artifact
#   SURVIVED the very same hostile probe (DLOPEN-OK): the hermetization
#   works. v12:
#   1) the summary heredoc escapes \$ORIGIN (the bug that nuked a good run);
#   2) FRESH_OK flag: once the gates passed and both copies are in place, a
#      late failure no longer quarantines audited artifacts;
#   3) the doctor's ~/.Renviron check ignores COMMENTED lines (false alarm on
#      an already-fixed Renviron);
#   4) a full end-to-end kit test (fake wheel zip + real cmake) that would
#      have caught the heredoc bug before shipping.
#
# CHANGELOG v11 -- doctor feedback from the field. The first real --doctor
#   output proved three things: (1) R runs PERMANENTLY with
#   LD_LIBRARY_PATH=/opt/rocm/lib (shell profile AND ~/.Renviron) -- the
#   chimera fuel; the doctor now flags it loudly; (2) the v9 artifact's
#   RUNPATH itself carried /opt/rocm-7.2.4/lib and /opt/rocm/lib (CMake link
#   dirs baked in) -- v10's force-rpath already replaces it wholesale;
#   (3) the last kit run had been INTERRUPTED mid-staging (partial rsync:
#   dev symlinks present, big real files like libtorch_cpu.so missing), which
#   made the probes fail with a misleading "libtorch_cpu.so: cannot open" and
#   left the quarantine to remove the inst/kit copies. v11 doctor additions:
#   staging-inventory sanity (names the interrupted-staging state instead of
#   letting the probes mislead), /opt/rocm symlink resolution + hsa/MIOpen
#   inventory, libtorch_hip.so in the key-rpath list (ROCm wheels ship
#   torch_hip, not torch_cuda), fixed the silent "(no rpath)" branch, and a
#   patience note on the 2-3 GB staging step.
#
# CHANGELOG v10 -- the v9 gates all passed ("closure audit OK") yet R CMD
#   INSTALL still died on "/opt/rocm/lib/libamdhip64.so.7: undefined symbol:
#   hsa_amd_memory_get_preferred_copy_engine, version ROCR_1" with a FRESH
#   liblantern: the chimera is assembled AT LOAD TIME by the ambient
#   environment, not at link time. The v9 audit was structurally blind: it
#   forced LD_LIBRARY_PATH=<staged> FIRST, so any soname shared between the
#   wheel (6.4) and /opt/rocm (7.x) -- libhsa-runtime64.so.1, possibly
#   libMIOpen / rccl / ... -- resolved to the staged copy during the audit,
#   while the real R process (no override, or an LD_LIBRARY_PATH=/opt/rocm
#   inherited from the shell) lets the loader take the SYSTEM 7.x copy; that
#   copy needs libamdhip64.so.7, which binds against the ALREADY-LOADED 6.4
#   libhsa and dies on the missing ROCR_1 symbol. Python survives the same
#   environment because nothing in its chain RPATH-protects the 6.4 hsa
#   first. Three cures, layered:
#   1) hermetize_libs(): EVERY library of the staged tree gets DT_RPATH=$ORIGIN
#      (patchelf --force-rpath). DT_RPATH has PRIORITY over LD_LIBRARY_PATH
#      and over the ld.so cache, unlike DT_RUNPATH -- a hostile environment
#      can no longer swap a shared soname onto the 7.x chain. The built
#      liblantern gets DT_RPATH='$ORIGIN:<staged>/lib'; the kit artifact
#      '$ORIGIN' alone (install_torch drops it next to the wheel libs).
#   2) audit_closure() now runs in THREE regimes and fails on the first bad
#      one: pristine (LD_LIBRARY_PATH unset -- what the R process actually
#      does), forced (staged first -- wheel self-sufficiency), hostile
#      (LD_LIBRARY_PATH=/opt/rocm/lib -- stress test of the hermetization).
#      The kit artifact is audited inside a SIMULATED install layout
#      (hardlinked next to the wheel libs, like install_torch() will do).
#   3) --doctor: standalone, read-only diagnosis of the CURRENT install:
#      environment (LD_LIBRARY_PATH, ~/.Renviron, ld.so cache), /opt/rocm
#      soname-overlap inventory, RPATH-vs-RUNPATH types of the key libs,
#      NEEDED/RPATH dumps of every liblantern copy, a compiled dlopen probe
#      (RTLD_NOW, like dyn.load) run pristine AND hostile, plus LD_DEBUG=libs
#      culprit extraction ("find library=... required by ...") for the probe
#      and for a real Rscript library(torch).
#
# CHANGELOG v9 -- fix the chimera RELOAD: "undefined symbol:
#   hsa_amd_memory_get_preferred_copy_engine, version ROCR_1" came back at
#   R CMD INSTALL time, even though the kit gates now catch bad links.
#   Mechanism (the missing piece was on the R side): {torch}'s src/configure
#   sets LANTERN_TARGET=dummylantern unless BUILD_LANTERN is exported, so a
#   plain `R CMD INSTALL .` NEVER builds lantern -- it prints "*** Skip
#   building lantern." and silently reinstalls the PREVIOUS
#   inst/lib/liblantern.so (the kit's step 3/4 copy). When a kit run stops
#   BEFORE step 3/4 (build error, gate failure), that stale artifact stays in
#   place: a pre-alignment liblantern whose DIRECT NEEDED list still contains
#   the system ROCm 7 math libs (libhipblas.so.3, libhipsparse.so.4,
#   libhipsolver.so.1, libhipblaslt.so.1). At load time those drag
#   /opt/rocm/lib/libamdhip64.so.7 into the process; it binds to the
#   ALREADY-LOADED libhsa-runtime64.so.1 of the wheel's 6.4 chain (bundled in
#   torch/lib), which lacks the recent ROCR_1 symbol -> crash. Python works
#   because nothing there ever loads the .so.7. v9 makes the failure loud and
#   the artifact flow airtight:
#   1) patchelf is now a REQUIRED tool. Without it repair_needed_entries
#      silently no-op'ed, the gate then stopped the kit, and the stale
#      inst/lib/liblantern.so kept poisoning every R CMD INSTALL.
#   2) On ANY kit failure an EXIT trap QUARANTINES inst/lib/liblantern.so and
#      torch-rocm-kit/lantern/lib/liblantern.so: R CMD INSTALL then fails
#      with a loud "liblantern.so missing" instead of installing a 6.4/7.2
#      chimera.
#   3) audit_closure(): after the build, ldd walks the FULL transitive
#      closure of liblantern in the staged layout -- any dependency resolving
#      into the system ROCm (or "not found") fails the kit. verify_runtime_deps
#      only sees DIRECT NEEDED entries; this catches everything else.
#   4) The summary prints liblantern's final direct NEEDED list (eyeball
#      check without readelf).
#
# CHANGELOG v8 -- fix the next gate failure: "unresolved direct dependencies of
#   liblantern: libhipblas.so.3 libhipsparse.so.4 libhipsolver.so.1
#   libhipblaslt.so.1". The v4 shim re-aligned the HIP RUNTIME (libamdhip64)
#   but torch's exported HIP interfaces still dragged the SYSTEM roc* math
#   libraries (ROCm 7.x sonames: hipblas .so.3, hipsparse .so.4, hipsolver
#   .so.1, hipblaslt .so.1) into lantern's direct DT_NEEDED, while the wheel
#   ships its own sonames (hipblas .so.2, hipsparse .so.1, hipsolver .so.0,
#   hipblaslt .so.0). /opt/rocm being in lantern's RUNPATH, the loader would
#   have silently MIXED 7.x math libs with the wheel's 6.4 chain (the v4
#   crash class, reborn). The shim now aligns ALL of them with three
#   complementary CMake mechanisms (whichever one the interface uses, one
#   hits) plus a bash-side last resort:
#   a) target_link_directories(lantern BEFORE "${TORCH_PATH}/lib") -- plain
#      -l<name> items resolve to the staged wheel copies first;
#   b) bare-name imported targets (hipblas, hipsparse, ...) created or
#      redirected to the wheel files -- CMake substitutes them into plain-name
#      interface items at generate time;
#   c) a sweep of every IMPORTED target still pointing outside the wheel
#      (roc::X, hip::X, full-path locations): re-aligned to the same-family
#      library staged from the wheel;
#   d) repair_needed_entries() -- after the build, any direct NEEDED still
#      missing from the staging dir is rewritten onto the wheel's SONAME with
#      'patchelf --replace-needed' (no-op without patchelf). The
#      verify_runtime_deps gate then has the final word.
#
# CHANGELOG v7 -- fix "libamdhip64.so.6: cannot open shared object file" that
#   SURVIVED v6 even though the RUNPATH was correct. Two distinct defects:
#   1) Link time vs load time: the linker only reads the SONAME stored INSIDE
#      the ELF it links (DT_NEEDED = libamdhip64.so.6), but dlopen() must find
#      a FILE with that exact name in the RUNPATH dirs. The staged wheel dir
#      ships the HIP runtime under another name (e.g. libamdhip64.so.6.4.43482
#      and/or libamdhip64.so) while every OTHER wheel lib happens to carry its
#      exact SONAME as file name -- hence the single, very confusing failure.
#      New ensure_soname_links() scans the staged lib dir, reads each real
#      library's SONAME (objdump -p) and creates/repairs the matching symlink.
#      New verify_runtime_deps() then re-checks, after the build, that EVERY
#      direct NEEDED of liblantern exists BY EXACT NAME in the staged dir:
#      the failure is now caught at build time with an explicit message
#      instead of a cryptic R CMD INSTALL error.
#   2) On localized systems (fr_FR) the readelf guards were silent no-ops:
#      French readelf prints "Bibliotheque partagee : [...]" so the sed
#      patterns matching 'Shared library:' / 'Library runpath:' never matched.
#      The script now exports LC_ALL=C (all binutils output is parsed in the
#      C locale).
#   Also: the 4/4 kit artifact gets RUNPATH='$ORIGIN' (patchelf, optional) so
#   install_torch() can drop it next to the libtorch bundle, anywhere.
#
# CHANGELOG v6 -- fix at load time from an INSTALLED package: "libamdhip64.so.6:
#   cannot open shared object file". The v4 re-pointing worked (DT_NEEDED is
#   now the wheel's libamdhip64.so.6) but the loader failed to FIND it when
#   liblantern.so is dlopen'ed from the installed library directory: the
#   lookup goes through the RUNPATH of liblantern, which must explicitly point
#   at the staged libtorch dir. v6 passes explicit CMAKE_BUILD_RPATH /
#   CMAKE_INSTALL_RPATH so the RUNPATH is guaranteed in the built binary,
#   verifies the RUNPATH with readelf after the build, and documents two cures
#   in the summary (LD_LIBRARY_PATH through ~/.Renviron, or an ld.so.conf.d
#   entry).
# CHANGELOG v5 -- English translation of all comments and messages
#   (upstream-ready, ASCII-only). No logic change vs v4.
#
# CHANGELOG v4 -- fix at load time: "/opt/rocm/lib/libamdhip64.so.7: undefined
#   symbol: hsa_amd_memory_get_preferred_copy_engine, version ROCR_1".
#   Diagnosis: the build succeeded, but liblantern got linked against the
#   SYSTEM HIP runtime (hip::amdhip64 defined by hip-config.cmake 7.2 ->
#   /opt/rocm) while libc10_hip/libtorch_cuda from the wheel embed the 6.4
#   chain. At dlopen time, the libhsa-runtime64 of the 6.4 chain is already
#   loaded when the .so.7 arrives and demands a recent ROCR_1 symbol that this
#   older libhsa does not export -> failure. Python works because nothing
#   there ever loads the .so.7. v4 REWRITES the IMPORTED_LOCATION of the
#   hip::{amdhip64,host,device} targets to the libamdhip64 EMBEDDED in the
#   wheel (${TORCH_PATH}/lib) -- exactly the runtime already validated on the
#   Python side -- and adds a post-build readelf guard (die if libamdhip64.so.7
#   remains linked).
#
# CHANGELOG v3 -- fix "Some (but not all) targets in this export set were
#   already defined (hip::amdhip64)":
#   v2 defined hip::amdhip64 BEFORE find_package(Torch) (through
#   CMAKE_PROJECT_INCLUDE / CMAKE_PROJECT_TOP_LEVEL_INCLUDES, executed right
#   after project()). But on ROCm 7.x the modern hip-config.cmake -- found
#   without any trouble by torch's LoadHIP.cmake (HIP 7.2.53211 on the target
#   machine) -- defines the WHOLE export set hip::amdhip64 + hip::host +
#   hip::device by itself; the pre-existing target made the export set
#   partially defined => configure aborted. The v2 premise ("legacy FindHIP,
#   version 0.0.0") does not apply to libtorch 2.8's LoadHIP, which goes
#   through CONFIG mode.
#   v3 stopped injecting anything up-front. The shim became a conditional
#   fallback ADDED AFTER find_package(Torch) in <lantern-src>/CMakeLists.txt
#   (marked block, idempotent, removed by --unpatch): a no-op when torch
#   already defined the targets (system ROCm present), a life-saver on
#   "wheel-only" machines without /opt/rocm -- the common case for torch R
#   users, relevant for the upstream follow-up of #907.
# =============================================================================
set -euo pipefail

# v7: parse binutils output (readelf/objdump) in the C locale. On a localized
# system (e.g. LANG=fr_FR.UTF-8) readelf prints "Bibliotheque partagee : [...]"
# instead of "Shared library: [...]" and every sed-based guard below silently
# matched nothing. Script messages stay ASCII English (upstream-ready).
export LC_ALL=C

MODE=""; SRC=""; TREE_ARG=""; LANTERN_ARG=""; UNPATCH=""; DOCTOR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-venv)    MODE="venv"; SRC="${2:?}"; shift 2 ;;
    --from-zip)     MODE="zip";  SRC="${2:?}"; shift 2 ;;
    --tree)         TREE_ARG="${2:?}"; shift 2 ;;
    --lantern-src)  LANTERN_ARG="${2:?}"; shift 2 ;;
    --unpatch)      UNPATCH=1; shift ;;
    --doctor)       DOCTOR=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
[[ -z "$MODE" && -z "$UNPATCH" && -z "$DOCTOR" ]] && { echo "Usage: $0 --from-venv <venv> | --from-zip <libtorch.zip> | --unpatch | --doctor"; exit 1; }

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*"; exit 1; }

# >>> kit-functions (test-extractable block: pure functions only) --------------
# --- hip-shim block: markers + idempotent handling ----------------------------
MARK_A='# >>> hip-shim-rocm (make_rocm_kit.sh)'
MARK_B='# <<< hip-shim-rocm'

# remove the marked block (and trailing blank lines)
strip_hip_shim() {
  local f="$1" tmp
  grep -qF "$MARK_A" "$f" || return 0
  tmp=$(mktemp)
  awk -v A="$MARK_A" -v B="$MARK_B" '
    BEGIN { sk = 0 }
    index($0, A) == 1 { sk = 1; next }
    index($0, B) == 1 { sk = 0; next }
    sk == 0 { buf[NR] = $0 }
    END {
      n = NR
      while (n > 0 && buf[n] == "") { delete buf[n]; n-- }
      for (i = 1; i <= NR; i++) if (i in buf) print buf[i]
    }' "$f" > "$tmp" && mv "$tmp" "$f"
}

# append (or rewrite) the block AFTER find_package(Torch) -- at the end of the
# file: target names are resolved at generation time, not while parsing
apply_hip_shim() {
  local f="$1"
  [[ -f "$f" ]] || die "CMakeLists.txt not found: $f"
  strip_hip_shim "$f"
  [[ -f "$f.orig-make-rocm-kit" ]] || cp "$f" "$f.orig-make-rocm-kit"
  cat >> "$f" <<'EOF'

# >>> hip-shim-rocm (make_rocm_kit.sh) -- self-managed block (remove with: tools/make_rocm_kit.sh --unpatch)
# v4 -- ALIGN the HIP runtime with the staged wheel (see CHANGELOG v4 in the script):
# torch's CMake config (LoadHIP) points hip::* at the SYSTEM ROCm, while
# libc10_hip/libtorch_cuda from the wheel embed their own ROCm chain (6.4).
# Linking the system libamdhip64.so.7 on top mixes two runtimes: at load time,
# the already-loaded libhsa (6.4 chain) does not export the recent ROCR_1
# symbol required by the .so.7 -> "undefined symbol:
# hsa_amd_memory_get_preferred_copy_engine".
# We therefore REWRITE IMPORTED_LOCATION to the libamdhip64 bundled in the
# wheel (exactly the runtime already validated on the Python side). If the
# targets do not exist (wheel-only machine), create them: wheel first, system
# ROCm as a last resort.
# v8 -- ALIGN the roc* math libraries as well (hipblas, hipsparse, hipsolver,
# hipblaslt, ...). torch's exported HIP interfaces drag the SYSTEM ROCm math
# libs (7.x sonames) into lantern's direct DT_NEEDED while the wheel ships
# its own sonames -> silent 6.4/7.x mixing or load failure. Three
# complementary mechanisms; whichever the interface uses, one hits:
#   a) the staged lib dir is searched FIRST for plain -l<name> items;
#   b) bare-name imported targets are created/redirected to the wheel files
#      (CMake substitutes them into plain-name interface items);
#   c) any imported target still pointing outside the wheel (roc::X, hip::X,
#      full-path locations) is re-aligned to the same-family wheel library.
set(_KIT_HIP "")
file(GLOB _KIT_HIP_GLOB "${TORCH_PATH}/lib/libamdhip64.so*")
if(_KIT_HIP_GLOB)
  list(SORT _KIT_HIP_GLOB)
  list(GET _KIT_HIP_GLOB 0 _KIT_HIP)
endif()
foreach(_hiptgt amdhip64 host device)
  if(TARGET hip::${_hiptgt})
    if(_KIT_HIP)
      set_target_properties(hip::${_hiptgt} PROPERTIES IMPORTED_LOCATION "${_KIT_HIP}")
      message(STATUS "[hip-shim] hip::${_hiptgt} re-aligned to the wheel: ${_KIT_HIP}")
    else()
      get_target_property(_HIPLOC hip::${_hiptgt} IMPORTED_LOCATION)
      message(STATUS "[hip-shim] hip::${_hiptgt} left on system ROCm: ${_HIPLOC}")
    endif()
  elseif(_KIT_HIP)
    add_library(hip::${_hiptgt} UNKNOWN IMPORTED)
    set_target_properties(hip::${_hiptgt} PROPERTIES IMPORTED_LOCATION "${_KIT_HIP}")
    message(STATUS "[hip-shim] wheel fallback: hip::${_hiptgt} -> ${_KIT_HIP}")
  else()
    find_library(_HIPCAND NAMES amdhip64
                 HINTS "$ENV{ROCM_PATH}" "$ENV{HIP_PATH}" /opt/rocm
                 PATH_SUFFIXES lib)
    if(NOT _HIPCAND)
      message(FATAL_ERROR "[hip-shim] libamdhip64 not found (neither ${TORCH_PATH}/lib, nor ROCM_PATH, nor /opt/rocm)")
    endif()
    add_library(hip::${_hiptgt} UNKNOWN IMPORTED)
    set_target_properties(hip::${_hiptgt} PROPERTIES IMPORTED_LOCATION "${_HIPCAND}")
    message(STATUS "[hip-shim] system fallback: hip::${_hiptgt} -> ${_HIPCAND}")
  endif()
endforeach()
unset(_KIT_HIP_GLOB)
unset(_KIT_HIP)
# v8 (a) -- make sure plain -l<name> items resolve to the staged wheel first.
if(TARGET lantern)
  target_link_directories(lantern BEFORE PRIVATE "${TORCH_PATH}/lib")
endif()
# v8 (b) -- bare-name imported targets for the known ROCm math libraries.
set(_KIT_ROCM_LIBS amdhip64 hipblas hipblaslt hipsparse hipsolver hipfft hiprand
                   rocblas rocsolver rocsparse rocrand hiprtc roctracer64 MIOpen)
foreach(_kitlib ${_KIT_ROCM_LIBS})
  set(_KIT_CAND "")
  file(GLOB _KIT_GLOB "${TORCH_PATH}/lib/lib${_kitlib}.so*")
  if(_KIT_GLOB)
    list(SORT _KIT_GLOB)
    list(GET _KIT_GLOB 0 _KIT_CAND)
  endif()
  if(_KIT_CAND)
    if(TARGET ${_kitlib})
      set_target_properties(${_kitlib} PROPERTIES IMPORTED_LOCATION "${_KIT_CAND}"
                                                  IMPORTED_LOCATION_RELEASE "${_KIT_CAND}")
      message(STATUS "[hip-shim] ${_kitlib} re-aligned to the wheel: ${_KIT_CAND}")
    else()
      add_library(${_kitlib} UNKNOWN IMPORTED)
      set_target_properties(${_kitlib} PROPERTIES IMPORTED_LOCATION "${_KIT_CAND}"
                                                  IMPORTED_LOCATION_RELEASE "${_KIT_CAND}")
      message(STATUS "[hip-shim] wheel fallback target: ${_kitlib} -> ${_KIT_CAND}")
    endif()
  endif()
  unset(_KIT_GLOB)
  unset(_KIT_CAND)
endforeach()
# v8 (c) -- sweep every imported target still pointing outside the staged
# wheel and re-align it to the same-family library (covers roc::X / hip::X
# style targets and full-path locations from the system ROCm).
get_property(_KIT_TGTS DIRECTORY PROPERTY IMPORTED_TARGETS)
foreach(_tgt IN LISTS _KIT_TGTS)
  get_target_property(_KIT_LOC "${_tgt}" IMPORTED_LOCATION)
  if(NOT _KIT_LOC)
    continue()
  endif()
  string(FIND "${_KIT_LOC}" "${TORCH_PATH}/" _KIT_POS)
  if(_KIT_POS EQUAL 0)
    continue()                       # already points at the staged wheel
  endif()
  get_filename_component(_KIT_BASE "${_KIT_LOC}" NAME)
  string(REGEX REPLACE "^lib([A-Za-z0-9_]+)\\.so.*$" "\\1" _KIT_STEM "${_KIT_BASE}")
  if(_KIT_STEM STREQUAL _KIT_BASE)
    continue()                       # not a lib<name>.so* path
  endif()
  set(_KIT_CAND "")
  file(GLOB _KIT_GLOB "${TORCH_PATH}/lib/lib${_KIT_STEM}.so*")
  if(_KIT_GLOB)
    list(SORT _KIT_GLOB)
    list(GET _KIT_GLOB 0 _KIT_CAND)
  endif()
  if(NOT _KIT_CAND)
    message(STATUS "[hip-shim] ${_tgt} left on system ROCm (no wheel counterpart: ${_KIT_STEM})")
    continue()
  endif()
  set_target_properties("${_tgt}" PROPERTIES IMPORTED_LOCATION "${_KIT_CAND}"
                                             IMPORTED_LOCATION_RELEASE "${_KIT_CAND}")
  message(STATUS "[hip-shim] ${_tgt} re-aligned: ${_KIT_LOC} -> ${_KIT_CAND}")
  unset(_KIT_GLOB)
  unset(_KIT_CAND)
endforeach()
unset(_KIT_TGTS)
# v14 -- COMPILE definitions for a ROCm wheel. Two gaps the wheel cannot fill,
# both hit at 51% (the first TU including ATen/cuda headers):
# (1) torch's AMD HIP build does NOT generate c10/cuda/impl/cuda_cmake_macros.h
#     (see the comment around the include in c10/cuda/CUDAMacros.h upstream),
#     so ROCm pip wheels ship none. Upstream provides the escape macro
#     C10_CUDA_NO_CMAKE_CONFIGURE_FILE. Without it:
#     "fatal error: c10/cuda/impl/cuda_cmake_macros.h: No such file or
#     directory" from c10/cuda/CUDAMacros.h:8. (Some libtorch zip layouts
#     carry the file -- defining is harmless there.)
# (2) lantern compiles plain g++ (no hipcc). ATen/cuda/CUDAContextLight.h
#     (USE_ROCM) pulls <hipsolver.h> from the SYSTEM ROCm SDK headers (the
#     wheel ships no SDK headers), and those headers #error out without a HIP
#     platform macro: "Must define exactly one of __HIP_PLATFORM_AMD__ or
#     __HIP_PLATFORM_NVIDIA__". torch's own LoadHIP passes
#     -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__ to its try_compile probes
#     (cmake/public/LoadHIP.cmake); hipcc would define it implicitly. The
#     shim passes the same pair to the lantern target.
# Gated on the staged tree being a ROCm wheel so CUDA/CPU trees are untouched.
if(TARGET lantern AND EXISTS "${TORCH_PATH}/lib/libtorch_hip.so")
  target_compile_definitions(lantern PRIVATE
    __HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__ C10_CUDA_NO_CMAKE_CONFIGURE_FILE)
  message(STATUS "[hip-shim] compile definitions for the ROCm wheel: __HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__ C10_CUDA_NO_CMAKE_CONFIGURE_FILE")
endif()
# v15 -- enable lantern's CUDA-flavored implementations on the ROCm wheel.
# src/lantern/src/Cuda.cpp guards the real cuda_* code with "#ifdef __NVCC__"
# (torch's CMakeLists sets it ONLY for $ENV{CUDA} nvcc builds); on a ROCm
# wheel build every one of those functions compiled to its #else stub that
# THROWS at runtime ("`cuda_get_device` is only supported on CUDA runtimes."):
# cuda_get_device_capability, cuda_memory_stats/summary/snapshot,
# cuda_runtime_version, cuda_empty_cache. The implementations are plain
# host-side calls the wheel's torch_hip/c10_hip libraries provide
# (at::cuda::getDeviceProperties, c10::cuda::CUDACachingAllocator,
# torch::cuda::_memory_snapshot_pickled). The kit patches Cuda.cpp to widen
# those guards to LANTERN_HIP (patch_cuda_hip) and switches the only
# CUDA-only call (cudaRuntimeGetVersion) to hipRuntimeGetVersion. The define
# lives here, gated on a ROCm wheel, so CUDA/CPU trees are untouched.
if(TARGET lantern AND EXISTS "${TORCH_PATH}/lib/libtorch_hip.so")
  target_compile_definitions(lantern PRIVATE LANTERN_HIP=1)
  message(STATUS "[hip-shim] LANTERN_HIP=1 -- cuda_* API paths enabled for the ROCm wheel (patch_cuda_hip)")
endif()
# v16 -- cuda->HIP compat headers FIRST on lantern's include path + USE_ROCM.
# torch 2.8.0's ATen/cuda + c10/cuda headers include <cuda_runtime_api.h>,
# <cusparse.h>, <cublas_v2.h>, <cublasLt.h>, <cuda.h> UNGUARDED even under
# USE_ROCM (torch's own ROCm build hipifies its sources instead of compiling
# these vanilla headers). Downstream g++ resolves them to the NVIDIA toolkit
# headers in /usr/include when present (conflicting uchar1/dim3/cudaMemcpyKind
# declarations vs the HIP headers the USE_ROCM branches pull) or fails
# outright when absent. The kit stages HIP-backed shims in
# build-lantern/hip-compat (ensure_hip_compat) and puts that dir before every
# other include dir of the target. USE_ROCM is exported by the wheel's torch
# config today; spelling it out keeps the target self-sufficient.
if(TARGET lantern AND EXISTS "${TORCH_PATH}/lib/libtorch_hip.so" AND EXISTS "${TORCH_PATH}/../hip-compat/cuda_runtime_api.h")
  target_include_directories(lantern BEFORE PRIVATE "${TORCH_PATH}/../hip-compat")
  target_compile_definitions(lantern PRIVATE USE_ROCM)
  message(STATUS "[hip-shim] cuda->HIP compat headers first on the include path: ${TORCH_PATH}/../hip-compat (+ USE_ROCM)")
endif()
# <<< hip-shim-rocm
EOF
}

# --- v16: cuda -> HIP compat headers (torch's unguarded CUDA includes) --------
# The ROCm wheel installs torch's VANILLA headers and six of them include
# CUDA-named headers unguarded even under USE_ROCM. torch's own ROCm build
# hipifies its sources before compiling (tools/amd_build), so upstream never
# hits this; a downstream g++ build does. The compat dir shadows the six
# includes with HIP-backed equivalents. The token map is the EXACT cuda*
# closure of {ATen/cuda/CUDAContext.h, ATen/cuda/detail/CUDAHooks.h,
# c10/cuda/CUDACachingAllocator.h} in torch 2.8.0 (26 cuda* tokens; verified
# programmatically) + the four math-handle aliases the ROCm headers do NOT
# provide. Gated on a ROCm wheel staging so CUDA/CPU trees are untouched.
ensure_hip_compat() {
  [[ -f "$LIBTORCH_DIR/lib/libtorch_hip.so" ]] || return 0
  local dir="$BUILD_DIR/hip-compat"
  rm -rf "$dir"                       # clean regeneration on every kit run
  mkdir -p "$dir"

  cat > "$dir/cuda_runtime_api.h" <<'EOF'
#pragma once
/* cuda -> HIP compat shim (make_rocm_kit.sh v16).
 * torch 2.8.0 includes <cuda_runtime_api.h> UNGUARDED even under USE_ROCM
 * (ATen/cuda/CUDAContextLight.h:7, c10/cuda/CUDAStream.h:3,
 * c10/cuda/CUDAFunctions.h:14). torch's own ROCm build hipifies its sources
 * before compiling; downstream g++ builds must map the CUDA names onto HIP
 * themselves. Token set = the exact cuda* closure of the torch 2.8.0 headers
 * lantern compiles (ATen/cuda/CUDAContext.h, ATen/cuda/detail/CUDAHooks.h,
 * c10/cuda/CUDACachingAllocator.h + their c10/cuda closure). */
#include <hip/hip_runtime_api.h>

using cudaError_t    = hipError_t;
using cudaStream_t   = hipStream_t;
using cudaEvent_t    = hipEvent_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaDeviceProp = hipDeviceProp_t;
using cudaStreamCaptureMode   = hipStreamCaptureMode;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;

#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyAsyncPeer hipMemcpyPeerAsync
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaEventCreateWithFlags hipEventCreateWithFlags
EOF

  cat > "$dir/cuda.h" <<'EOF'
#pragma once
/* cuda -> HIP compat shim (v16): c10/cuda/CUDAException.h:9 includes <cuda.h>
 * unguarded. torch's ROCm build hipifies it to <hip/hip_runtime.h>; the
 * driver-API surface is not referenced by the closure lantern compiles, so
 * the runtime API + the token map below are enough. */
#include <hip/hip_runtime_api.h>
#include <cuda_runtime_api.h>
EOF

  cat > "$dir/cusparse.h" <<'EOF'
#pragma once
/* cuda -> HIP compat shim (v16): ATen/cuda/CUDAContextLight.h:8 includes
 * <cusparse.h> unguarded. hipSPARSE ships no cusparse* compat typedefs
 * (hipsparseHandle_t is void*, verified against ROCm 6.4.4). */
#include <hipsparse.h>
using cusparseHandle_t = hipsparseHandle_t;
EOF

  cat > "$dir/cublas_v2.h" <<'EOF'
#pragma once
/* cuda -> HIP compat shim (v16): ATen/cuda/CUDAContextLight.h:9 includes
 * <cublas_v2.h> unguarded (hipBLAS itself includes cublas_v2.h only under
 * __HIP_PLATFORM_NVCC__). hipBLAS ships no cublasHandle_t alias. */
#include <hipblas.h>
using cublasHandle_t = hipblasHandle_t;
EOF

  cat > "$dir/cublasLt.h" <<'EOF'
#pragma once
/* cuda -> HIP compat shim (v16): ATen/cuda/CUDAContextLight.h:13 includes
 * <cublasLt.h> unguarded. hipBLASLt defines hipblasLtHandle_t (void*), no
 * cublasLtHandle_t alias. */
#include <hipblaslt.h>
using cublasLtHandle_t = hipblasLtHandle_t;
EOF

  cat > "$dir/cusolverDn.h" <<'EOF'
#pragma once
/* cuda -> HIP compat shim (v16): ATen/cuda/CUDAContextLight.h:16 includes
 * <cusolverDn.h> under "#ifdef CUDART_VERSION" (dead here -- the HIP shim
 * defines no CUDART_VERSION, exactly like torch's own ROCm build); kept for
 * robustness. hipSOLVER's dense handle is hipsolverDnHandle_t; no
 * cusolverDnHandle_t alias is shipped. */
#include <hipsolver.h>
using cusolverDnHandle_t = hipsolverDnHandle_t;
EOF

  echo "    [compat] wrote 6 cuda->HIP shim headers in $dir (torch 2.8.0 closure: 26 cuda* tokens + 4 handle aliases)"
}

# --- v7: SONAME symlink repair -------------------------------------------------
# The linker records in DT_NEEDED the SONAME stored INSIDE the ELF it links,
# but dlopen() looks for a FILE named exactly like that SONAME in the RUNPATH
# dirs. Wheels/staging trees sometimes ship a library only under a versioned
# or unversioned name (e.g. libamdhip64.so.6.4.43482, libamdhip64.so) -> link
# OK, load KO ("libamdhip64.so.6: cannot open shared object file"). This scans
# the staged lib dir and creates the missing "<SONAME> -> <file>" symlinks.
ensure_soname_links() {
  local dir="$1" f base soname created=0
  [[ -d "$dir" ]] || return 0
  for f in "$dir"/*.so*; do
    [[ -e "$f" ]] || continue                # unmatched glob or dangling link
    [[ -L "$f" ]] && continue                # only real files carry a SONAME
    base="$(basename "$f")"
    soname="$(objdump -p "$f" 2>/dev/null | awk '$1=="SONAME"{print $2; exit}' || true)"
    [[ -n "$soname" ]] || continue
    [[ "$soname" == "$base" ]] && continue
    if [[ -L "$dir/$soname" && ! -e "$dir/$soname" ]]; then
      rm -f "$dir/$soname"                   # repair a dangling symlink
    fi
    if [[ ! -e "$dir/$soname" ]]; then
      ln -s "$base" "$dir/$soname"
      echo "    [soname] linked $soname -> $base (dlopen needs the exact SONAME as file name)"
      created=$((created + 1))
    fi
  done
  [[ "$created" -gt 0 ]] && log "created $created SONAME symlink(s) in $dir"
  return 0
}

# v7: post-build gate -- every direct dependency of liblantern must exist BY
# EXACT NAME in the staged lib dir. This is precisely the lookup dlopen()
# performs at load time; catching it here turns a cryptic R CMD INSTALL error
# into an actionable build-time message.
verify_runtime_deps() {
  local lib="$1" libdir="$2" needed missing=""
  ensure_soname_links "$libdir"
  while IFS= read -r needed; do
    [[ -n "$needed" ]] || continue
    case "$needed" in
      libc.so*|libm.so*|libdl.so*|librt.so*|libpthread.so*|libstdc++.so*|libgcc_s.so*|ld-linux*) continue ;;
    esac
    if [[ ! -e "$libdir/$needed" ]]; then
      missing+=" $needed"
    fi
  done < <(readelf -d "$lib" 2>/dev/null | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p')
  if [[ -n "$missing" ]]; then
    die "unresolved direct dependencies of liblantern:$missing
     none of them exists in $libdir (dlopen would load the SYSTEM copy from
     /opt/rocm and mix two ROCm versions, or fail outright on other machines).
     If patchelf is installed, rerun this script: it rewrites stale NEEDED
     entries onto the wheel's SONAMEs automatically (repair_needed_entries).
     Otherwise re-stage the libtorch side: tools/make_rocm_kit.sh --from-venv <venv>"
  fi
  log "runtime deps OK: every direct NEEDED of liblantern resolves in $libdir"
}

# v8: last-resort alignment of lantern's DIRECT dependencies. For every
# non-system NEEDED that does not exist in the staged lib dir, rewrite it to
# the SONAME of the same-family library staged from the wheel (patchelf).
# Normally a no-op: the CMake shim (mechanisms a/b/c) already links the
# wheel's copies. Keeps the gate meaningful when an exported interface bakes
# full system paths into the link line.
repair_needed_entries() {
  local lib="$1" libdir="$2" needed stem cand f soname
  command -v patchelf >/dev/null 2>&1 || return 0
  while IFS= read -r needed; do
    [[ -n "$needed" ]] || continue
    case "$needed" in
      libc.so*|libm.so*|libdl.so*|librt.so*|libpthread.so*|libstdc++.so*|libgcc_s.so*|ld-linux*) continue ;;
    esac
    [[ -e "$libdir/$needed" ]] && continue
    stem="$(printf '%s' "$needed" | sed -E 's/^lib([A-Za-z0-9_]+)\.so.*/\1/')"
    [[ "$stem" != "$needed" ]] || continue
    cand=""
    for f in "$libdir/lib$stem".so*; do
      [[ -f "$f" && ! -L "$f" ]] && { cand="$f"; break; }
    done
    if [[ -z "$cand" ]]; then
      warn "no wheel counterpart for $needed -- left as is (the gate below will judge)"
      continue
    fi
    soname="$(objdump -p "$cand" 2>/dev/null | awk '$1=="SONAME"{print $2; exit}' || true)"
    [[ -n "$soname" ]] || continue
    echo "    [needed] $needed -> $soname (patchelf --replace-needed, wheel alignment)"
    patchelf --replace-needed "$needed" "$soname" "$lib"
  done < <(readelf -d "$lib" 2>/dev/null | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p')
  return 0
}

# --- v15: enable the REAL cuda_* implementations on the ROCm wheel ------------
# src/lantern/src/Cuda.cpp guards them with "#ifdef __NVCC__" (7 sites: the
# include block + 6 function bodies), which is only set for $ENV{CUDA} nvcc
# builds. patch_cuda_hip() widens the guards to LANTERN_HIP (defined by the
# hip-shim, gated on a ROCm wheel), switches cudaRuntimeGetVersion to
# hipRuntimeGetVersion, fixes the copy-pasted message of the runtime-version
# function and rewords the messages to mention HIP (ROCm).
# Marked + backed up like the hip-shim block; --unpatch restores the file.
patch_cuda_hip() {
  local f="$LANTERN_SRC/src/Cuda.cpp" bak="$LANTERN_SRC/src/Cuda.cpp.orig-make-rocm-kit"
  [[ -f "$f" ]] || { echo "    [patch] $f not found -- skipped (older lantern layout)"; return 0; }
  [[ -f "$LIBTORCH_DIR/lib/libtorch_hip.so" ]] || { echo "    [patch] not a ROCm wheel staging -- Cuda.cpp untouched"; return 0; }
  if grep -q 'LANTERN_HIP' "$f"; then
    if [[ -f "$bak" ]]; then
      cp "$bak" "$f"                  # reapply from the pristine copy (idempotent)
    else
      warn "Cuda.cpp already carries the LANTERN_HIP patch and no pristine backup exists -- leaving it as is (if the upstream file changed, restore your VCS copy first, then rerun)"
      return 0
    fi
  else
    [[ -f "$bak" ]] || cp "$f" "$bak"
  fi
  python3 - "$f" <<'PYEOF' || die "patch_cuda_hip: Cuda.cpp patch failed -- unexpected tree layout. Restore with: cp '$bak' '$f'"
import re, sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

def fail(msg):
    sys.stderr.write("patch_cuda_hip: %s\n" % msg)
    sys.exit(1)

# 1) widen every __NVCC__ guard (include block + 6 function bodies)
src, n_guards = re.subn(r'^#ifdef __NVCC__',
                        '#if defined(__NVCC__) || defined(LANTERN_HIP)',
                        src, flags=re.M)
if n_guards < 6:
    fail("expected >= 6 '#ifdef __NVCC__' guards, found %d "
         "(tree layout changed? restore the backup and report)" % n_guards)

# 2) explicit HIP runtime include (hipRuntimeGetVersion); include-guarded
#    upstream, so a double include is harmless
anchor = "#include <torch/csrc/cuda/memory_snapshot.h>\n#endif\n"
if anchor in src:
    src = src.replace(anchor, anchor +
                      "#ifdef LANTERN_HIP\n"
                      "#include <hip/hip_runtime_api.h>   /* hipRuntimeGetVersion; the wheel's libamdhip64 exports it */\n"
                      "#endif\n", 1)
else:
    # different include layout (e.g. unguarded includes in older trees):
    # hip/hip_runtime_api.h is already in the transitive closure there
    sys.stderr.write("patch_cuda_hip: include anchor not found -- relying on the transitive hip include\n")

# 3) runtime version: hipRuntimeGetVersion under LANTERN_HIP (the only
#    CUDA-only symbol in the enabled paths) + fix the copy-pasted message
m = re.search(r'int _lantern_cuda_get_runtime_version\(\) \{\n.*?\n\}\n', src, re.S)
if not m:
    fail("_lantern_cuda_get_runtime_version not found")
fn = m.group(0)
new_fn = fn.replace(
    "#if defined(__NVCC__) || defined(LANTERN_HIP)\n"
    "  int runtimeVersion;\n"
    "  cudaRuntimeGetVersion(&runtimeVersion);\n"
    "  return runtimeVersion;\n",
    "#if defined(__NVCC__) || defined(LANTERN_HIP)\n"
    "  int runtimeVersion;\n"
    "#if defined(LANTERN_HIP)\n"
    "  (void)hipRuntimeGetVersion(&runtimeVersion);   /* HIP/ROCm: same C ABI family */\n"
    "#else\n"
    "  cudaRuntimeGetVersion(&runtimeVersion);\n"
    "#endif\n"
    "  return runtimeVersion;\n")
if new_fn == fn:
    fail("runtime-version body not matched")
new_fn = new_fn.replace('"`cuda_device_stats` is only supported on CUDA runtimes."',
                        '"`cuda_runtime_version` is only supported on CUDA runtimes."')
src = src.replace(fn, new_fn, 1)

# 4) all remaining messages mention HIP too (the renamed one included)
src, n_msgs = re.subn(r'is only supported on CUDA runtimes\.',
                      'is only supported on CUDA or HIP (ROCm) runtimes.', src)
if n_msgs != 6:
    fail("expected exactly 6 error messages to reword, got %d" % n_msgs)

# sanity: guards present, hip call present, braces still balanced
if src.count("#if defined(__NVCC__) || defined(LANTERN_HIP)") != n_guards:
    fail("guard count mismatch after patching")
if "hipRuntimeGetVersion" not in src:
    fail("hipRuntimeGetVersion missing after patching")
if src.count("{") != src.count("}"):
    fail("brace imbalance after patching")

open(path, "w", encoding="utf-8").write(src)
print("    [patch] Cuda.cpp: %d guards widened (LANTERN_HIP), hipRuntimeGetVersion branch, 6 messages fixed" % n_guards)
PYEOF
}

# --- v15: R-side cuda_* fixes (marked block APPENDED to R/cuda.R) -------------
# Append + strip (no source surgery) = tree-version proof: R resolves the
# LAST definition of a function, so the block below overrides the upstream
# ones. Idempotent via the markers; --unpatch strips it.
MARK_R_A='# >>> cuda-hip-rocm (make_rocm_kit.sh)'
MARK_R_B='# <<< cuda-hip-rocm'

strip_cuda_r_block() {
  local f="$1" tmp
  [[ -f "$f" ]] || return 0
  grep -qF "$MARK_R_A" "$f" || return 0
  tmp=$(mktemp)
  awk -v A="$MARK_R_A" -v B="$MARK_R_B" '
    BEGIN { sk = 0 }
    index($0, A) == 1 { sk = 1; next }
    index($0, B) == 1 { sk = 0; next }
    sk == 0 { buf[NR] = $0 }
    END {
      n = NR
      while (n > 0 && buf[n] == "") { delete buf[n]; n-- }
      for (i = 1; i <= NR; i++) if (i in buf) print buf[i]
    }' "$f" > "$tmp" && mv "$tmp" "$f"
}

apply_cuda_r_block() {
  local f="$TREE/R/cuda.R"
  [[ -f "$f" ]] || { warn "R/cuda.R not found -- R-side cuda_* fixes skipped"; return 0; }
  strip_cuda_r_block "$f"
  cat >> "$f" <<'EOF'

# >>> cuda-hip-rocm (make_rocm_kit.sh) -- self-managed block (remove with: tools/make_rocm_kit.sh --unpatch)
# v15 -- ROCm/HIP fixes for the cuda_* API; these definitions override the
# ones above (R resolves the LAST definition of a function):
# - cuda_synchronize(device = NULL) passed -1 ("synchronize all devices"),
#   which dies inside the wheel's HIP CUDAHooks ("device >= 0 && device <
#   num_gpus ... device=-80" assert). The roxygen doc above already promises
#   cuda_current_device() as the default.
# - cuda_runtime_version() decoded the CUDA layout (major*1000 + minor*10 +
#   patch); on HIP, hipRuntimeGetVersion returns major*1e7 + minor*1e5 +
#   patch (ROCm 6.4 -> 60443482), which the CUDA decode turns into nonsense.
cuda_synchronize <- function(device = NULL) {
  if (is.null(device)) {
    device <- cuda_current_device()
  }
  cpp_cuda_synchronize(device)
}
cuda_runtime_version <- function() {
  v <- cpp_cuda_get_runtime_version()
  if (v >= 10000000) {
    major <- trunc(v / 10000000)
    minor <- trunc((v - major * 10000000) / 100000)
    patch <- v - major * 10000000 - minor * 100000
  } else {
    major <- trunc(v / 1000)
    minor <- trunc((v - major * 1000) / 10)
    patch <- v - major * 1000 - minor * 10
  }
  numeric_version(paste(major, minor, patch, sep = "."))
}
# <<< cuda-hip-rocm
EOF
  echo "    [patch] R/cuda.R: cuda-hip-rocm block appended (cuda_synchronize default device + HIP runtime-version decode)"
  if command -v Rscript >/dev/null 2>&1; then
    Rscript --vanilla -e "invisible(parse('$f'))" 2>/dev/null \
      || warn "R/cuda.R does not PARSE after the patch -- run: Rscript -e \"parse('$f')\""
  fi
}

# --- v15: bridge for the hipified nvrtc loader --------------------------------
# libtorch_hip.so dlopens "libcaffe2_nvrtc.so" lazily (device RNG state --
# observed in the field with torch::cuda_get_rng_state() -- and JIT kernel
# compilation). CUDA wheels ship the stub; the ROCm 6.x pip wheels do NOT
# (newer ROCm wheels ship it again; see tile-ai/tilelang#3711 for the
# wheel-level fix). On ROCm the equivalent C API family (hiprtc*) lives in
# libamdhiprtc, so a symlink named libcaffe2_nvrtc.so pointing at the staged
# hiprtc gives the loader the file it wants (the real work happens inside
# hiprtc). No-op when the wheel already ships its own copy; warning when
# there is no hiprtc at all (RNG state + JIT stay broken; doctor [9]).
ensure_nvrtc_stub() {
  local lib="$LIBTORCH_DIR/lib" cand
  [[ -d "$lib" ]] || return 0
  [[ -e "$lib/libcaffe2_nvrtc.so" ]] && return 0
  for cand in "$lib"/libamdhiprtc.so* "$lib"/libhiprtc.so*; do
    [[ -e "$cand" ]] || continue
    ln -sfn "$(basename "$cand")" "$lib/libcaffe2_nvrtc.so"
    echo "    [nvrtc] linked libcaffe2_nvrtc.so -> $(basename "$cand") (hiprtc exposes the C API the hipified loader expects)"
    return 0
  done
  warn "nvrtc bridge: the wheel ships neither libcaffe2_nvrtc.so nor libhiprtc* -- torch::cuda_get_rng_state() and JIT kernel compilation will fail until a wheel that bundles one is staged"
  return 0
}

# v9/v10: marker directory for the loader audit; overridable in tests (fake
# ROCm root). In production this is the system ROCm install the wheel chain
# must NEVER be mixed with.
ROCM_MARK="/opt/rocm"
# v10: hostile-library dir for the stress audit + doctor probes (overridable
# in tests). Defaults to the system ROCm lib dir.
ROCM_HOSTILE_LIB="/opt/rocm/lib"

# v10: make the staged tree HERMETIC. Every real ELF in the staged lib dir
# gets DT_RPATH='$ORIGIN' (patchelf --force-rpath). Why DT_RPATH and not the
# usual DT_RUNPATH: in the ELF search order, DT_RPATH of the object being
# resolved is consulted BEFORE LD_LIBRARY_PATH and the ld.so cache, while
# DT_RUNPATH is consulted AFTER LD_LIBRARY_PATH. A shell exporting
# LD_LIBRARY_PATH=/opt/rocm/lib (or ROCm's ld.so.conf cache entry) can
# therefore silently swap any soname shared between the wheel 6.4 chain and
# the system 7.x chain -- the exact mechanism that rebuilt the chimera at
# load time even though every link-time gate passed. With DT_RPATH=$ORIGIN
# on every wheel library, no ambient environment can override the staged
# copies anymore. (The three-regime audit_closure below proves it.)
hermetize_libs() {
  local dir="$1" f n=0 tot=0
  for f in "$dir"/*.so*; do
    [[ -f "$f" && ! -L "$f" ]] || continue
    readelf -h "$f" >/dev/null 2>&1 || continue
    tot=$((tot+1))
  done
  for f in "$dir"/*.so*; do
    [[ -f "$f" && ! -L "$f" ]] || continue
    readelf -h "$f" >/dev/null 2>&1 || continue
    if ! patchelf --force-rpath --set-rpath '$ORIGIN' "$f" 2>/dev/null; then
      warn "hermetize: patchelf failed on $(basename "$f") -- the audits below will judge"
      continue
    fi
    n=$((n+1))
    [[ $((n % 20)) -eq 0 ]] && echo "    ... $n/$tot hermetized"
  done
  log "hermetized $n/$tot libraries in $dir (DT_RPATH=\$ORIGIN beats LD_LIBRARY_PATH + ld.so.cache)"
}

# v10: WHOLE-closure audit in THREE regimes. verify_runtime_deps only inspects
# liblantern's DIRECT NEEDED entries; this walks the FULL transitive closure
# exactly like the dynamic loader will at dlopen time (ldd), under three
# different environments:
#   pristine : LD_LIBRARY_PATH unset  -- what the R process actually does;
#              catches cache-based swaps and missing sonames;
#   forced   : LD_LIBRARY_PATH=<staged> first -- wheel self-sufficiency;
#   hostile  : LD_LIBRARY_PATH=/opt/rocm/lib -- stress test: with the v10
#              DT_RPATH hermetization nothing may move to /opt/rocm. This is
#              the regime the v9 audit could not see (it prepended <staged> to
#              the INHERITED LD_LIBRARY_PATH, masking the swap).
# Any resolution into the system ROCm means the wheel 6.4 chain and the 7.x
# chain would end up in ONE process (the "undefined symbol ... version ROCR_1"
# crash class); a "not found" line is a guaranteed load failure. Both die,
# with the offending lines shown.
audit_closure() {
  local lib="$1" libdir="$2" regime trace nf bad
  for regime in pristine forced hostile; do
    case "$regime" in
      pristine) trace=$(env -u LD_LIBRARY_PATH ldd "$lib" 2>&1 || true) ;;
      forced)   trace=$(env "LD_LIBRARY_PATH=$libdir" ldd "$lib" 2>&1 || true) ;;
      hostile)
        if [[ -z "${ROCM_HOSTILE_LIB:-}" || ! -d "${ROCM_HOSTILE_LIB:-}" ]]; then
          warn "hostile audit skipped (ROCM_HOSTILE_LIB=${ROCM_HOSTILE_LIB:-<unset>} not present)"
          continue
        fi
        trace=$(env "LD_LIBRARY_PATH=$ROCM_HOSTILE_LIB" ldd "$lib" 2>&1 || true) ;;
    esac
    if [[ "$regime" == "pristine" ]]; then
      echo "$trace" | grep -E 'lib(amdhip64|hsa-runtime64)' | sed 's/^/    [rocm-chain] /' || true
    fi
    nf=$(echo "$trace" | grep 'not found' || true)
    if [[ -n "$nf" ]]; then
      echo "$nf" | sed 's/^/    /'
      die "closure audit [$regime]: unresolved dependency for $(basename "$lib") -- see the 'not found' lines above"
    fi
    bad=$(echo "$trace" | grep -F "$ROCM_MARK" || true)
    if [[ -n "$bad" ]]; then
      echo "$bad" | sed 's/^/    /'
      die "closure audit [$regime]: $(basename "$lib") resolves dependencies from $ROCM_MARK -- the wheel 6.4 chain and the system ROCm 7.x would be mixed at load time (undefined-symbol class)"
    fi
  done
  log "closure audit OK [pristine+forced+hostile]: $(basename "$lib") is hermetic"
}

# v10: audit the KIT artifact in a simulated install layout. The kit copy
# carries DT_RPATH='$ORIGIN' only, so auditing it in place (kit/lantern/lib,
# alone) would trivially fail: at install_torch() time it lives NEXT TO the
# wheel libs. We rebuild that layout: hardlinks of the staged libs + a real
# copy of the artifact in a scratch dir, then run the three-regime audit on
# it. (Hardlink for real files -- $ORIGIN must be the joint dir; plain
# symlinks for the soname links, preserving their targets.)
audit_kit_layout() {
  local artifact="$1" libdir="$2" aud="$3" f base tgt
  rm -rf "$aud"; mkdir -p "$aud"
  for f in "$libdir"/*; do
    [[ -e "$f" ]] || continue
    base="$(basename "$f")"
    [[ "$base" == "liblantern.so" ]] && continue
    if [[ -L "$f" ]]; then
      tgt="$(readlink "$f")"
      ln -s "$tgt" "$aud/$base" 2>/dev/null || true
    else
      ln "$f" "$aud/$base" 2>/dev/null || cp "$f" "$aud/$base" 2>/dev/null || true
    fi
  done
  cp "$artifact" "$aud/liblantern.so"
  audit_closure "$aud/liblantern.so" "$libdir"
  rm -rf "$aud"
}

# v9: remove stale liblantern artifacts so a HALF-FAILED kit run can no longer
# poison the next `R CMD INSTALL .`. Mechanism: {torch}'s src/configure sets
# LANTERN_TARGET=dummylantern unless BUILD_LANTERN is exported, so
# `R CMD INSTALL .` NEVER builds lantern ("*** Skip building lantern.") and
# silently reinstalls the PREVIOUS inst/lib/liblantern.so. If that file is a
# pre-alignment binary, its system-ROCm NEEDEDs drag /opt/rocm 7.x libs into
# the process at load time (see CHANGELOG v9). Quarantining turns that silent
# chimera into a loud "liblantern.so not found".
quarantine_stale_libs() {
  local f
  for f in "$TREE/inst/lib/liblantern.so" "$KIT_DIR/lantern/lib/liblantern.so"; do
    [[ -e "$f" ]] || continue
    rm -f "$f"
    echo "    [quarantine] removed stale artifact: $f"
  done
}

# v9: EXIT trap -- on any non-zero exit, quarantine before dying.
on_fail_quarantine() {
  local rc=$?
  [[ "$rc" -eq 0 ]] && return 0
  # v12: once the gates passed and both copies are installed (FRESH_OK=1), a
  # late failure -- e.g. while printing the summary -- must NOT quarantine the
  # audited fresh artifacts (that deleted a GOOD run in the field).
  if [[ "${FRESH_OK:-0}" == "1" ]]; then
    warn "kit run FAILED (rc=$rc) AFTER the gates passed and the artifacts were copied:"
    warn "inst/lib/liblantern.so and torch-rocm-kit/lantern/lib/liblantern.so are the AUDITED fresh copies -- kept (no quarantine)."
    return 0
  fi
  warn "kit run FAILED (rc=$rc) -- quarantining stale liblantern artifacts:"
  quarantine_stale_libs
  warn "'R CMD INSTALL .' reuses the PREVIOUS inst/lib/liblantern.so ('*** Skip building lantern.'),"
  warn "so a stale copy would resurrect the 6.4/7.2 chimera (undefined symbol ... ROCR_1)."
  warn "Fix the failure above, then rerun this script and R CMD INSTALL."
}

# v13: GPU ISA vs the code objects EMBEDDED in the wheel. "HIP error: invalid
# device function" at the FIRST kernel launch (device enumeration and
# allocation fine -- the loader layer is proven good) means the wheel's fat
# binaries contain no code object for the device's gfx target. Torch wheels
# are built for a fixed TORCH_ROCM_ARCH list baked in at build time; the kit
# cannot change it. This section reads the device targets (rocminfo), the
# code-object bundle ids actually embedded in the staged libraries (strings
# heuristic -- reliable for HIP fatbins and Tensile kernel names) and prints
# a verdict with a fix. Override targets are only suggested when the
# same-generation sibling IS in the wheel's list (identical base ISA; kernels
# using the sibling's ISA extensions may still trap at runtime).
declare -A HSA_OVERRIDE_MAP=(
  [gfx900]="9.0.6:experimental"
  [gfx1031]="10.3.0"  [gfx1032]="10.3.0"  [gfx1034]="10.3.0"
  [gfx1035]="10.3.0"  [gfx1036]="10.3.0"
  [gfx1102]="11.0.0"  [gfx1103]="11.0.0"
  [gfx1150]="11.0.0:experimental"  [gfx1151]="11.0.0:experimental"
)
doctor_gpu_arch() {
  local rinfo gpus="" arch line wa="" files=() labels=() arcs=() lst i f ov tag sib wv="" rf seen_rf=""
  echo "  [8] GPU ISA vs wheel code objects ('invalid device function' layer)"
  [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]] && echo "      HSA_OVERRIDE_GFX_VERSION currently set: $HSA_OVERRIDE_GFX_VERSION"
  if [[ -f "$LIBTORCH_DIR/version.py" ]]; then
    # v14: quote- and spacing-tolerant (single-quoted values were silently dropped)
    wv=$(sed -n -E \
         -e "s|^[[:space:]]*__version__[[:space:]]*=[[:space:]]*[\"']?([^\"']+)[\"']?.*|wheel: \1|p" \
         -e "s|^[[:space:]]*hip[[:space:]]*=[[:space:]]*[\"']?([^\"']+)[\"']?.*|/ HIP \1|p" \
         "$LIBTORCH_DIR/version.py" 2>/dev/null | tr '\n' ' ')
    [[ -n "$wv" ]] && echo "      staged $wv"
  fi
  for rinfo in "${ROCM_MARK}/bin/rocminfo" "$(command -v rocminfo 2>/dev/null || true)"; do
    [[ -n "$rinfo" && -x "$rinfo" ]] || continue
    gpus=$(LC_ALL=C "$rinfo" 2>/dev/null | awk -F': ' \
             '/^[[:space:]]*Name:/{n=$2} /^[[:space:]]*Marketing Name:/{if (n ~ /^gfx/) print n" | "$2}' || true)
    break
  done
  if [[ -z "$gpus" ]]; then
    for rinfo in "${ROCM_MARK}/bin/rocm_agent_enumerator" "$(command -v rocm_agent_enumerator 2>/dev/null || true)"; do
      [[ -n "$rinfo" && -x "$rinfo" ]] || continue
      gpus=$(LC_ALL=C "$rinfo" 2>/dev/null | grep -E '^gfx[0-9a-f]+$' | sort -u || true)
      break
    done
  fi
  if [[ -z "$gpus" ]]; then
    warn "      (no rocminfo / rocm_agent_enumerator found -- install 'rocminfo' so the doctor can read the GPU ISA)"
    return 0
  fi
  echo "$gpus" | sed 's/^/      device: /'

  # the arch lists baked into the staged fatbins
  f="$LIBTORCH_DIR/lib/libtorch_hip.so"
  if [[ ! -f "$f" || ! -r "$f" ]]; then
    warn "      (staged libtorch_hip.so not found -- run the kit first: ./tools/make_rocm_kit.sh --from-venv <venv>)"
    return 0
  fi
  files=("$f"); labels=("torch kernels")
  # v14: the pip wheel ships REAL UNVERSIONED files (librocblas.so) and the
  # kit adds the versioned names (librocblas.so.4) as SYMLINKS -- a "! -L"
  # filter would drop them all. Resolve symlinks and dedupe by real path.
  seen_rf=" $f "
  for f in "$LIBTORCH_DIR"/lib/librocblas.so* "$LIBTORCH_DIR"/lib/libhipblaslt.so*; do
    [[ -e "$f" ]] || continue
    rf="$(readlink -f "$f" 2>/dev/null || true)"
    [[ -n "$rf" && -f "$rf" ]] || rf="$f"
    [[ -f "$rf" ]] || continue
    case " $seen_rf " in *" $rf "*) continue ;; esac
    seen_rf="$seen_rf $rf"
    files+=("$rf"); labels+=("matmul kernels ($(basename "$rf"))")
  done
  for i in "${!files[@]}"; do
    # v14: require gfx + 2 digits + at least 1 more char -- ids like gfx0/gfx1/gfx9/
    # gfx11/gfx94/gfx9f are prefix junk from unrelated strings, never real ISA
    # names (real ones: gfx900, gfx90a, gfx1030, gfx1100, ...).
    lst=$(strings "${files[$i]}" | grep -oE 'gfx[0-9]{2}[0-9a-z]{1,2}' | sort -u | tr '\n' ' ' || true)
    echo "      ${labels[$i]}: $lst"
    arcs+=("$lst")
  done
  wa="${arcs[0]}"

  # verdict per device, then the GEMM layer judged independently
  while IFS= read -r line; do
    arch=$(printf '%s' "$line" | awk '{print $1}')
    [[ "$arch" == gfx* ]] || continue
    if [[ " $wa " == *" $arch "* ]]; then
      echo "      verdict [$arch]: COVERED by the torch kernels (code objects present)."
    elif [[ "$arch" == "gfx803" ]]; then
      echo "      verdict [$arch]: Polaris -- dropped by ROCm >= 5: no override, no modern rebuild path (last resort: ROCm 4.x container)."
    elif [[ -n "${HSA_OVERRIDE_MAP[$arch]:-}" ]]; then
      ov="${HSA_OVERRIDE_MAP[$arch]}"; tag=""
      if [[ "$ov" == *":experimental" ]]; then tag=" (EXPERIMENTAL)"; ov="${ov%%:*}"; fi
      sib="gfx$(printf '%s' "$ov" | tr -d '.')"
      if [[ " $wa " == *" $sib "* ]]; then
        echo "      verdict [$arch]: NOT in the torch kernels, but same-generation sibling $sib IS -- try the 30-second test:$tag"
        echo "        HSA_OVERRIDE_GFX_VERSION=$ov Rscript -e 'library(torch); print(torch_tensor(1:12, device=\"cuda:0\")*5)'"
        echo "        (for every R session: add 'HSA_OVERRIDE_GFX_VERSION=$ov' to ~/.Renviron)"
        echo "        caveat: $sib code objects may use ISA instructions $arch lacks -> some kernels can trap ('illegal instruction'), esp. GEMM."
      else
        echo "      verdict [$arch]: NOT covered and sibling $sib is not in this wheel either --"
        echo "        rebuild libtorch from source with TORCH_ROCM_ARCH=$arch (a wheel property; the kit cannot change it)."
      fi
    else
      echo "      verdict [$arch]: NOT covered and no same-generation sibling known --"
      echo "        rebuild libtorch from source with TORCH_ROCM_ARCH=$arch."
    fi
    for i in "${!files[@]}"; do
      [[ "${labels[$i]}" == "torch kernels" ]] && continue
      lst="${arcs[$i]}"
      if [[ -n "$lst" && " $lst " == *" $arch "* ]]; then
        echo "      matmul [$arch]: $(basename "${files[$i]}") covers it."
      else
        echo "      matmul [$arch]: $(basename "${files[$i]}") ships NO kernels for it -- matmul/conv fail even if the override unblocks elementwise ops."
      fi
    done
  done <<< "$gpus"
  echo "      note: arch lists = code-object bundle ids found in the staged fatbins (strings heuristic, reliable for HIP fatbins / Tensile kernels; ids shorter than 3 chars after gfx are filtered as prefix junk)."
}

# v10: --doctor. Standalone, READ-ONLY diagnosis of a load failure (never
# quarantines anything: it runs before the EXIT trap is armed). Answers ONE
# question: WHO drags the system ROCm into the torch load? Prints the
# environment, the sonames shared between the wheel and /opt/rocm, the rpath
# TYPES (RPATH beats LD_LIBRARY_PATH, RUNPATH loses), the NEEDED/RPATH of
# every liblantern copy, a compiled dlopen probe (RTLD_NOW|RTLD_LOCAL, like
# R's dyn.load) run pristine and hostile, and -- on failure -- the LD_DEBUG
# culprit lines ("find library=... required by ..."), for the probe and for a
# real Rscript library(torch). v13 adds [8]: GPU ISA vs the code objects
# embedded in the wheel (the "invalid device function" layer, above loading).
doctor() {
  local f lib p ip n=0 cands=() probe="$BUILD_DIR/doctor_probe" srct="$BUILD_DIR/doctor_probe.c" dbg="$BUILD_DIR/doctor-lddebug.log" renv="" rp=""
  mkdir -p "$BUILD_DIR"
  for t in readelf ldd gcc strings; do command -v "$t" >/dev/null || die "doctor needs $t"; done

  echo "  [1] environment"
  echo "      LD_LIBRARY_PATH = ${LD_LIBRARY_PATH:-<unset>}"
  if [[ "${LD_LIBRARY_PATH:-}" == *"$ROCM_MARK/lib"* ]]; then
    warn "      [!!] the shell exports the system ROCm in LD_LIBRARY_PATH -- chimera fuel. Remove it from your shell profile (~/.bashrc, /etc/profile.d/rocm*); the v10+ artifacts are immune (DT_RPATH), your system is not."
  fi
  if [[ -f "$HOME/.Renviron" ]]; then
    renv=$(grep -n 'LD_LIBRARY_PATH' "$HOME/.Renviron" 2>/dev/null | grep -vE '^[0-9]+:[[:space:]]*#' || true)
    if [[ -n "$renv" ]]; then
      echo "$renv" | sed 's/^/      Renviron:/'
      if [[ "$renv" == *"$ROCM_MARK"* ]]; then
        warn "      [!!] ~/.Renviron re-exports the system ROCm into LD_LIBRARY_PATH for EVERY R session -- comment that line out (R needs no ROCm in LD_LIBRARY_PATH; the staged wheel carries its own chain)."
      fi
    else
      echo "      Renviron: (no ACTIVE LD_LIBRARY_PATH line)"
    fi
  else
    echo "      Renviron: (no ~/.Renviron)"
  fi
  echo "      ld.so.conf.d rocm entries:"
  grep -l rocm /etc/ld.so.conf.d/*.conf 2>/dev/null | sed 's/^/        /' || echo "        (none)"

  echo "  [2] system ROCm sonames"
  echo "      $ROCM_MARK resolves to: $(readlink -f "$ROCM_MARK" 2>/dev/null || echo '(not a symlink / absent)')"
  ls -la "$ROCM_MARK"/lib/libamdhip64.so* "$ROCM_MARK"/lib/libhsa-runtime64.so* "$ROCM_MARK"/lib/libMIOpen.so* 2>/dev/null | sed 's/^/      /' || echo "      (absent: $ROCM_MARK/lib)"
  ldconfig -p 2>/dev/null | grep -E 'lib(amdhip64|hsa-runtime64)\.so' | sed 's/^/      cache: /' || true

  if [[ -f "$LIBTORCH_DIR/lib/libtorch_cpu.so" ]]; then
    echo "  [3] staging inventory OK. Sonames ALSO present in $ROCM_MARK/lib (mixing candidates):"
    n=0
    for f in "$LIBTORCH_DIR"/lib/*.so*; do
      [[ -e "$f" ]] || continue
      [[ -e "$ROCM_MARK/lib/$(basename "$f")" ]] || continue
      echo "      $(basename "$f")"; n=$((n+1))
    done
    [[ "$n" -eq 0 ]] && echo "      (none)"
    echo "  [4] rpath TYPE of key staged libs (RPATH wins over LD_LIBRARY_PATH; RUNPATH loses)"
    for f in libamdhip64.so.6 libtorch_hip.so libtorch_cpu.so libhsa-runtime64.so.1; do
      p=""
      for c in "$LIBTORCH_DIR/lib/"$f*; do [[ -e "$c" ]] && { p="$c"; break; }; done
      if [[ -n "$p" ]]; then
        rp=$(readelf -d "$p" 2>/dev/null | grep -E 'Library (rpath|runpath)' | head -n1)
        if [[ -n "$rp" ]]; then echo "      $(basename "$p"): $rp"
        else echo "      $(basename "$p"): (NO rpath at all -- hostile env can override it)"; fi
      fi
    done
  elif [[ -d "$LIBTORCH_DIR/lib" ]]; then
    warn "  [3] staging INCOMPLETE (libtorch_cpu.so missing) -- the last kit run was interrupted mid-staging (rm -rf + rsync)."
    warn "      Rerun the kit END-TO-END and let the copy finish; the probes below fail misleadingly until then."
  fi

  echo "  [5] liblantern copies"
  if command -v Rscript >/dev/null 2>&1; then
    ip=$(Rscript --vanilla -e 'cat(system.file(package="torch"))' 2>/dev/null || true)
    if [[ -n "$ip" && -f "$ip/lib/liblantern.so" ]]; then cands+=("$ip/lib/liblantern.so"); echo "      installed: $ip/lib/liblantern.so"; fi
  fi
  [[ -f "$TREE/inst/lib/liblantern.so" ]]       && { cands+=("$TREE/inst/lib/liblantern.so"); echo "      inst:     $TREE/inst/lib/liblantern.so"; }
  [[ -f "$KIT_DIR/lantern/lib/liblantern.so" ]] && { cands+=("$KIT_DIR/lantern/lib/liblantern.so"); echo "      kit:      $KIT_DIR/lantern/lib/liblantern.so"; }
  lib=$(find "$BUILD_DIR" -maxdepth 2 -name 'liblantern.so' 2>/dev/null | head -n1 || true)
  [[ -n "$lib" ]] && { cands+=("$lib"); echo "      build:    $lib"; }
  [[ "${#cands[@]}" -eq 0 ]] && echo "      (none -- run the kit first)"

  for lib in "${cands[@]}"; do
    echo "  [readelf] $lib"
    readelf -d "$lib" 2>/dev/null | sed -n -e 's/.*Shared library: \[\([^]]*\)\].*/      NEEDED  \1/p' \
                                             -e 's/.*Library runpath: \[\([^]]*\)\].*/      RUNPATH \1/p' \
                                             -e 's/.*Library rpath: \[\([^]]*\)\].*/      RPATH   \1/p'
  done

  cat > "$srct" <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
  if (argc < 2) return 2;
  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "DLOPEN-FAIL: %s\n", dlerror()); return 1; }
  printf("DLOPEN-OK: %s\n", argv[1]);
  dlclose(h);
  return 0;
}
EOF
  gcc -O0 -o "$probe" "$srct" -ldl || die "doctor: cannot compile the dlopen probe"

  # v13: the kit copy ships ALONE in kit/lantern/lib -- install_torch() drops
  # it NEXT TO the wheel libs, and audit_kit_layout() already proved that
  # layout at build time. A "libtorch_cpu.so: cannot open" failure THERE is
  # expected in staging: annotate it instead of burning an LD_DEBUG capture
  # on it (the first field report showed the confusing empty culprit grep).
  expected_staging_fail() {
    local lib="$1"
    [[ "$lib" == "$KIT_DIR/lantern/lib/liblantern.so" && ! -e "$KIT_DIR/lantern/lib/libtorch_cpu.so" && ! -e "$KIT_DIR/lantern/lib/libtorch.so" ]]
  }
  echo "  [6] dlopen probe (RTLD_NOW), pristine environment"
  for lib in "${cands[@]}"; do
    if env -u LD_LIBRARY_PATH "$probe" "$lib"; then :; else
      if expected_staging_fail "$lib"; then
        echo "      (expected in staging: the kit copy ships ALONE -- install_torch() drops it next to the wheel libs, layout proven by audit_kit_layout)"
      else
        echo "      (real failure -- culprit extraction in [7])"
      fi
    fi
  done
  echo "  [7] dlopen probe (RTLD_NOW), hostile LD_LIBRARY_PATH=${ROCM_HOSTILE_LIB:-/opt/rocm/lib}"
  if [[ -n "${ROCM_HOSTILE_LIB:-}" && -d "${ROCM_HOSTILE_LIB:-}" ]]; then
    for lib in "${cands[@]}"; do
      if env "LD_LIBRARY_PATH=$ROCM_HOSTILE_LIB" "$probe" "$lib"; then :; else
        if expected_staging_fail "$lib"; then
          echo "      (expected in staging: the kit copy ships ALONE -- install_torch() drops it next to the wheel libs)"
          continue
        fi
        echo "      capturing LD_DEBUG=libs,files to name the culprit..."
        env "LD_LIBRARY_PATH=$ROCM_HOSTILE_LIB" LD_DEBUG=libs,files "$probe" "$lib" >/dev/null 2>"$dbg" || true
        echo "      -- who requires what (culprit lines):"
        { grep -E 'needed by|required by' "$dbg" 2>/dev/null | grep -E 'amdhip64|hsa-runtime64' ;
          grep -E 'needed by|required by' "$dbg" 2>/dev/null | grep -F "$ROCM_MARK" ; } | sort -u | sed 's/^/      /' | head -20
        echo "      -- resolutions into $ROCM_MARK:"
        grep 'trying file=' "$dbg" 2>/dev/null | grep -F "$ROCM_MARK" | sed 's/^/      /' | head -15
        echo "      -- full log: $dbg"
      fi
    done
  else
    echo "      (skipped: hostile dir not present)"
  fi
  doctor_gpu_arch
  echo "  [9] nvrtc/hiprtc bridge (device RNG state + JIT kernel compilation)"
  if [[ ! -d "$LIBTORCH_DIR/lib" ]]; then
    echo "      (staged tree absent -- run the kit first)"
  else
    if [[ -e "$LIBTORCH_DIR/lib/libcaffe2_nvrtc.so" ]]; then
      if [[ -L "$LIBTORCH_DIR/lib/libcaffe2_nvrtc.so" ]]; then
        echo "      libcaffe2_nvrtc.so -> $(readlink "$LIBTORCH_DIR/lib/libcaffe2_nvrtc.so")  (kit bridge: the ROCm 6.x wheel ships no nvrtc stub)"
      else
        echo "      libcaffe2_nvrtc.so: shipped by the wheel"
      fi
      if [[ -x "$probe" ]] && env -u LD_LIBRARY_PATH "$probe" "$LIBTORCH_DIR/lib/libcaffe2_nvrtc.so" >/dev/null 2>&1; then
        echo "      dlopen probe: OK"
      else
        warn "      dlopen probe FAILED -- inspect with: $probe $LIBTORCH_DIR/lib/libcaffe2_nvrtc.so"
      fi
    else
      warn "      libcaffe2_nvrtc.so MISSING from staging -- torch::cuda_get_rng_state() dies with 'Error in dlopen: libcaffe2_nvrtc.so' (rerun the kit: v15 bridges it to libamdhiprtc when the wheel bundles one)"
    fi
    local h=""
    for h in "$LIBTORCH_DIR"/lib/lib*hiprtc*; do [[ -e "$h" ]] && break; h=""; done
    if [[ -n "$h" ]]; then
      echo "      hiprtc in staging: $(basename "$h")"
    else
      echo "      hiprtc in staging: (none -- this wheel bundles no hiprtc: device RNG state + JIT stay broken on it)"
    fi
  fi
  echo "  [10] installed {torch} package + install env vars (shell view; vars set in ~/.Renviron are R-only)"
  if command -v Rscript >/dev/null 2>&1; then
    local ipx
    ipx=$(Rscript --vanilla -e 'cat(system.file(package="torch"))' 2>/dev/null || true)
    if [[ -n "$ipx" ]]; then
      local ln
      for ln in liblantern.so libtorch.so; do
        if [[ -e "$ipx/lib/$ln" ]]; then
          echo "      installed $ipx/lib/$ln: present"
        else
          warn "      installed $ipx/lib/$ln: MISSING -- library(torch) will offer to install at every startup"
        fi
      done
    else
      echo "      (torch package not installed)"
    fi
  else
    echo "      (Rscript not available)"
  fi
  local dv
  for dv in TORCH_HOME TORCH_URL LANTERN_URL TORCH_INSTALL; do
    echo "      $dv = ${!dv:-<unset>}"
  done
  if [[ -n "${LANTERN_URL:-}" && "${LANTERN_URL:-}" == *"build-lantern"* ]]; then
    warn "      LANTERN_URL points at the cmake build tree -- install_lib() needs <dir>/lib/liblantern.so (the kit layout). Use $KIT_DIR/lantern, or install_torch() fails with 'Could not install lantern from ...'"
  fi
  if [[ "${TORCH_INSTALL:-}" != "0" ]]; then
    echo "      tip: add TORCH_INSTALL=0 to ~/.Renviron to silence the startup download prompt once the libraries are installed"
  fi
  echo "  [11] R-faithful probe (Rscript + LD_DEBUG)"
  if command -v Rscript >/dev/null 2>&1 && [[ -n "${cands[0]:-}" && -f "${cands[0]}" ]]; then
    env LD_DEBUG=libs,files Rscript --vanilla -e 'library(torch)' >/dev/null 2>"$dbg" || true
    { grep -E 'needed by|required by' "$dbg" 2>/dev/null | grep -E 'amdhip64|hsa-runtime64' ;
      grep -E 'needed by|required by' "$dbg" 2>/dev/null | grep -F "$ROCM_MARK" ; } | sort -u | sed 's/^/      /' | head -20 || echo "      (no rocm chain in the R trace)"
    echo "      -- full log: $dbg"
  else
    echo "      (Rscript or installed torch not available)"
  fi
  log "DOCTOR done. If [7]/[11] name a culprit, rerun the kit (v10 hermetizes the tree). If they stay clean but [8] says NOT covered for your gfx target, that is the layer ABOVE loading: the wheel has no code object for your GPU ISA (HSA_OVERRIDE_GFX_VERSION with a same-generation sibling, or rebuild libtorch with TORCH_ROCM_ARCH=<target> -- nothing the kit can fix). [9] judges the nvrtc/hiprtc bridge, [10] the installed package + install env vars. Paste this whole output if unsure."
}

# <<< kit-functions -------------------------------------------------------------

# --- Resolve the {torch} tree root --------------------------------------------
if [[ -n "$TREE_ARG" ]]; then
  TREE="$(cd "$TREE_ARG" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  TREE="$(dirname "$SCRIPT_DIR")"          # standard case: tools/../
fi
[[ -f "$TREE/DESCRIPTION" ]]          || die "DESCRIPTION not found: $TREE is not a {torch} tree root (use --tree)"
[[ -d "$TREE/R" ]]                    || die "R/ not found under $TREE"
LANTERN_SRC="${LANTERN_ARG:-$TREE/src/lantern}"
[[ -f "$LANTERN_SRC/CMakeLists.txt" ]] || die "CMakeLists.txt not found in $LANTERN_SRC (the legacy top-level lantern/ is no longer the upstream source; pass --lantern-src if needed)"

# --- Unpatch mode --------------------------------------------------------------
if [[ -n "$UNPATCH" ]]; then
  strip_hip_shim "$LANTERN_SRC/CMakeLists.txt"
  echo "hip-shim block removed from $LANTERN_SRC/CMakeLists.txt (backup: CMakeLists.txt.orig-make-rocm-kit)"
  # v15: restore the patched Cuda.cpp + strip the R/cuda.R block
  if [[ -f "$LANTERN_SRC/src/Cuda.cpp.orig-make-rocm-kit" ]]; then
    mv -f "$LANTERN_SRC/src/Cuda.cpp.orig-make-rocm-kit" "$LANTERN_SRC/src/Cuda.cpp"
    echo "restored $LANTERN_SRC/src/Cuda.cpp from the kit backup"
  fi
  if grep -qF "$MARK_R_A" "$TREE/R/cuda.R" 2>/dev/null; then
    strip_cuda_r_block "$TREE/R/cuda.R"
    echo "cuda-hip-rocm block removed from $TREE/R/cuda.R"
  fi
  exit 0
fi

BUILD_DIR="$TREE/build-lantern"
LIBTORCH_DIR="$BUILD_DIR/libtorch"
KIT_DIR="$TREE/torch-rocm-kit"

# v10: --doctor runs BEFORE any tool requirement and BEFORE the quarantine
# trap is armed: it is read-only and must never delete the artifacts it is
# diagnosing.
if [[ -n "$DOCTOR" ]]; then
  doctor
  exit 0
fi

for tool in cmake rsync readelf objdump python3; do
  command -v "$tool" >/dev/null || die "Missing required tool: $tool"
done
# v9: patchelf is REQUIRED, not optional. Without it repair_needed_entries
# silently no-op'ed, the gate then stopped the kit, and the STALE
# inst/lib/liblantern.so kept poisoning every R CMD INSTALL (chimera reload).
command -v patchelf >/dev/null || die "Missing required tool: patchelf (v9 needs it to rewrite stale NEEDED entries). Install with: sudo apt install patchelf"

# v9: from here on, ANY failure quarantines the stale liblantern artifacts on
# the way out, so the next `R CMD INSTALL .` cannot silently reinstall them.
trap on_fail_quarantine EXIT

# --- 1/4 Stage the ROCm libtorch into build-lantern/libtorch ------------------
log "1/4 Staging ROCm libtorch -> $LIBTORCH_DIR ($MODE)   [copies 2-3 GB -- let it FINISH, do not Ctrl-C]"
if [[ "${LD_LIBRARY_PATH:-}" == *"$ROCM_MARK"* ]]; then
  warn "shell LD_LIBRARY_PATH contains the system ROCm -- the build gates are env-proof (pristine/forced/hostile), but consider cleaning your shell profile"
fi
rm -rf "$LIBTORCH_DIR"
mkdir -p "$LIBTORCH_DIR"                       # fix: rsync does not create parents

if [[ "$MODE" == "venv" ]]; then
  VENV="$(cd "$SRC" && pwd -P)"
  TORCH_PKG=$(compgen -G "$VENV/lib/python3*/site-packages/torch" | head -n1)
  [[ -n "$TORCH_PKG" ]] || die "torch package not found in $VENV"
  PY="$VENV/bin/python"
  [[ -x "$PY" ]] || die "python interpreter missing: $PY"
  echo "torch $( "$PY" -c 'import torch; print(torch.__version__)' ) / HIP $( "$PY" -c 'import torch; print(getattr(torch.version, "hip", None))' )"
  rsync -a --exclude "__pycache__" "$TORCH_PKG/" "$LIBTORCH_DIR/"
  # install.R expects lib/libtorch.so; pip wheels only ship libtorch_cpu.so.
  ln -sf libtorch_cpu.so "$LIBTORCH_DIR/lib/libtorch.so"
else
  [[ -f "$SRC" ]] || die "Zip not found: $SRC"
  TMP=$(mktemp -d)
  unzip -q "$SRC" -d "$TMP"
  SUB=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -n1)
  # empty the staging dir then move the zip content into it
  rm -rf "$LIBTORCH_DIR" && mkdir -p "$LIBTORCH_DIR"
  # v12: dotglob+nullglob inside a subshell -- the old `mv "$SUB"/* "$SUB"/.[!.]*`
  # died on wheels without dotfiles (the literal glob fails the whole mv), and
  # an empty archive must fail LOUDLY instead of tripping set -e mid-glob.
  if [[ -n "$SUB" ]]; then
    ( shopt -s dotglob nullglob; entries=("$SUB"/*); [[ ${#entries[@]} -gt 0 ]] || exit 3; mv "${entries[@]}" "$LIBTORCH_DIR/" ) \
      || die "zip staging failed (empty or unreadable archive?)"
  else
    ( shopt -s nullglob; entries=("$TMP"/*); [[ ${#entries[@]} -gt 0 ]] || exit 3; mv "${entries[@]}" "$LIBTORCH_DIR/" ) \
      || die "zip staging failed (empty or unreadable archive?)"
  fi
  rm -rf "$TMP"
fi

[[ -f "$LIBTORCH_DIR/lib/libtorch_cpu.so" ]] || die "libtorch_cpu.so missing after staging"
[[ -d "$LIBTORCH_DIR/share/cmake" ]]         || die "share/cmake missing (find_package(Torch) will fail)"
# v7: the wheel may ship a library without the exact file name of its SONAME
# (dlopen looks DT_NEEDED names up verbatim). Create the missing symlinks now,
# BEFORE the link step records those SONAMEs into liblantern.
ensure_soname_links "$LIBTORCH_DIR/lib"
# v15: bridge the hipified nvrtc loader (device RNG state / JIT kernels).
ensure_nvrtc_stub
# v10: hermetize the staged tree BEFORE the build: every wheel library gets
# DT_RPATH=$ORIGIN, which has PRIORITY over LD_LIBRARY_PATH and the ld.so
# cache (DT_RUNPATH does not). This kills the load-time soname swap that
# rebuilt the 6.4/7.2 chimera even though every link-time gate passed.
hermetize_libs "$LIBTORCH_DIR/lib"
log "expected libtorch 2.8.0+rocm6.4 -- staged + hermetized: $LIBTORCH_DIR"

# --- 2/4 Build liblantern in build-lantern/ -----------------------------------
log "2/4 Building lantern (purged cache) -> $BUILD_DIR"
# purge a potentially stale cmake cache (but do NOT re-stage libtorch/!)
rm -rf "$BUILD_DIR/CMakeFiles" "$BUILD_DIR/CMakeCache.txt" "$BUILD_DIR/cmake-install" 2>/dev/null || true
# and remove liblantern.so from older builds (the final find must pick the fresh one)
rm -f "$BUILD_DIR/lib/liblantern.so" "$BUILD_DIR/liblantern.so"

# v15: $CUDA must not leak into the build -- lantern's CMakeLists takes a
# nvcc branch whenever it is set (enable_language(CUDA) + sort_vert_kernel.cu
# + __NVCC__ on Cuda.cpp/Amp.cpp). A ROCm wheel build has no nvcc: the branch
# either dies on enable_language(CUDA) or silently re-enables paths whose
# cudaRuntimeGetVersion/cudaMalloc families do not exist on HIP. The kit
# drives the HIP alignment itself (hip-shim + patch_cuda_hip).
unset CUDA || true

# v15: enable the real cuda_* implementations for the ROCm wheel BEFORE cmake
# compiles them (Cuda.cpp source patch; the LANTERN_HIP define comes from the
# shim below), and the R-side cuda_* fixes (marked block appended to R/cuda.R).
patch_cuda_hip
apply_cuda_r_block
# v16: stage the cuda->HIP compat headers for the torch include closure (the
# unguarded <cuda_runtime_api.h>/<cusparse.h>/... includes, see CHANGELOG v16).
ensure_hip_compat

# v4 hip shim: appended AFTER find_package(Torch) -- re-aligns hip::* to the
# HIP runtime EMBEDDED in the wheel (prevents mixing system 7.x with wheel
# 6.4), or creates the targets as a fallback (wheel-only machines).
# See CHANGELOG v4.
apply_hip_shim "$LANTERN_SRC/CMakeLists.txt"

cmake -S "$LANTERN_SRC" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_PATH="$LIBTORCH_DIR" \
      -DCMAKE_BUILD_RPATH="$LIBTORCH_DIR/lib" \
      -DCMAKE_INSTALL_RPATH="$LIBTORCH_DIR/lib"
# v16: fail fast when the compat injection did not land (a silent miss would
# resurface as the uchar1/dim3 conflict storm at 51%). ROCm-wheel trees only.
if [[ -f "$LIBTORCH_DIR/lib/libtorch_hip.so" ]]; then
  FLAGS_MAKE="$BUILD_DIR/CMakeFiles/lantern.dir/flags.make"
  [[ -f "$FLAGS_MAKE" ]] || die "flags.make missing after configure -- cmake configure failed above"
  grep -q -- '-DUSE_ROCM' "$FLAGS_MAKE" \
    || die "USE_ROCM missing from lantern's compile flags -- hip-shim block did not apply to $LANTERN_SRC/CMakeLists.txt"
  grep -q 'hip-compat' "$FLAGS_MAKE" \
    || die "hip-compat include dir missing from lantern's compile flags -- hip-shim block did not apply"
  echo "    flags check: USE_ROCM + hip-compat include dir present (ok)"
fi
cmake --build "$BUILD_DIR" --target lantern --parallel "$(nproc)"

LIBLANTERN=$(find "$BUILD_DIR" -maxdepth 2 -name 'liblantern.so' | head -n1)
[[ -n "$LIBLANTERN" ]] || die "liblantern.so not found after build"
log "Build OK: $LIBLANTERN"

# v8: align any remaining system-soname NEEDED onto the wheel's copies
# (last resort -- the CMake shim above normally already does the job).
repair_needed_entries "$LIBLANTERN" "$LIBTORCH_DIR/lib"

# v10: convert the artifact's rpath into DT_RPATH (priority over a hostile
# LD_LIBRARY_PATH) covering both consumers of this file:
#   $ORIGIN            -> the kit layout (liblantern next to the wheel libs)
#   $LIBTORCH_DIR/lib  -> the build-tree/inst copies on this machine
patchelf --force-rpath --set-rpath "\$ORIGIN:$LIBTORCH_DIR/lib" "$LIBLANTERN"
echo "    liblantern DT_RPATH -> \$ORIGIN:$LIBTORCH_DIR/lib (beats LD_LIBRARY_PATH)"

# --- v7 guards: catch load-time problems at BUILD time -------------------------
# (1) never mix the system 7.x HIP runtime with the wheel chain ("undefined
#     symbol: ... version ROCR_1", see CHANGELOG v4);
# (2) the RUNPATH must carry the staged libtorch dir so an INSTALLED copy of
#     liblantern still finds the wheel libs (CHANGELOG v6);
# (3) every direct NEEDED of liblantern must exist BY EXACT NAME in the staged
#     lib dir -- exactly the lookup dlopen() performs (CHANGELOG v7).
NEEDED_HIP=$(readelf -d "$LIBLANTERN" 2>/dev/null \
             | sed -n 's/.*Shared library: \[\(libamdhip64[^]]*\)\].*/\1/p' \
             | head -n1 || true)
case "$NEEDED_HIP" in
  libamdhip64.so.7*) die "liblantern links $NEEDED_HIP (system ROCm) -- conflicts with the wheel chain. Check the '[hip-shim]' lines from the configure step above." ;;
  "")  echo "    HIP runtime: no direct NEEDED (ok)" ;;
  *)   echo "    HIP runtime linked: $NEEDED_HIP (expected: the wheel's one)" ;;
esac
RPATH_VAL=$(readelf -d "$LIBLANTERN" 2>/dev/null \
            | sed -n -e 's/.*Library runpath: \[\([^]]*\)\].*/\1/p' -e 's/.*Library rpath: \[\([^]]*\)\].*/\1/p' \
            | head -n1 || true)
case ":${RPATH_VAL}:" in
  *":$LIBTORCH_DIR/lib:"*) echo "    RPATH/RUNPATH includes the staged libtorch dir (ok)" ;;
  *) warn "liblantern RPATH does not include $LIBTORCH_DIR/lib -- installed copies would only load with LD_LIBRARY_PATH set" ;;
esac
verify_runtime_deps "$LIBLANTERN" "$LIBTORCH_DIR/lib"
audit_closure "$LIBLANTERN" "$LIBTORCH_DIR/lib"

log "liblantern direct NEEDED entries (final):"
readelf -d "$LIBLANTERN" 2>/dev/null | sed -n 's/.*(Shared library: \[\([^]]*\)\]).*/    \1/p'

# --- 3/4 Copy into inst/lib (dev loading) --------------------------------------
log "3/4 Installing into the dev tree (inst/lib)"
mkdir -p "$TREE/inst/lib"
cp -v "$LIBLANTERN" "$TREE/inst/lib/liblantern.so"

# --- 4/4 Stage in install.R format for torch::install_torch() ------------------
log "4/4 Staging lantern/ for install_torch() -> $KIT_DIR/lantern"
rm -rf "$KIT_DIR/lantern"
mkdir -p "$KIT_DIR/lantern/lib"
cp -v "$LIBLANTERN" "$KIT_DIR/lantern/lib/liblantern.so"
# v7: make the kit artifact relocatable -- '$ORIGIN' resolves against the dir
# that will hold liblantern at install time (next to the libtorch bundle).
# Without patchelf the artifact keeps the absolute build-tree RUNPATH, which
# only works on this machine.
patchelf --force-rpath --set-rpath '$ORIGIN' "$KIT_DIR/lantern/lib/liblantern.so"
echo "    kit lantern DT_RPATH -> \$ORIGIN (relocatable + beats LD_LIBRARY_PATH)"
# v10: audit the kit artifact in a SIMULATED install layout (hardlinked next
# to the wheel libs, exactly where install_torch() will drop it).
audit_kit_layout "$KIT_DIR/lantern/lib/liblantern.so" "$LIBTORCH_DIR/lib" "$KIT_DIR/.audit"

# v12: everything above passed AND both copies are in place. A failure from
# here on (e.g. while printing the summary) must NOT quarantine the audited
# fresh artifacts (see CHANGELOG v12).
FRESH_OK=1

# --- Summary -------------------------------------------------------------------
cat <<EOF

Aligned tree:
  build-lantern/libtorch/          -> $LIBTORCH_DIR   (TORCH_URL candidate)
  build-lantern/hip-compat/        -> cuda->HIP shim headers (torch's unguarded
                                      <cuda_runtime_api.h>/<cusparse.h>/...
                                      includes; first on lantern's include path)
  build-lantern/liblantern.so      -> $LIBLANTERN
  inst/lib/liblantern.so           -> dev loading (load_all)
  torch-rocm-kit/lantern/          -> $KIT_DIR/lantern   (LANTERN_URL candidate)
  src/lantern/CMakeLists.txt       -> [hip-shim-rocm] block appended at the
                                      end of the file (remove with: ./tools/make_rocm_kit.sh --unpatch)

Load-time note (v10):
  the staged tree is HERMETIC: every wheel library carries DT_RPATH=\$ORIGIN
  and liblantern DT_RPATH='\$ORIGIN:$LIBTORCH_DIR/lib'. DT_RPATH has priority
  over LD_LIBRARY_PATH and the ld.so cache (DT_RUNPATH does not), so neither
  a shell exporting LD_LIBRARY_PATH=/opt/rocm/lib nor ROCm's cache entries
  can swap a shared soname (libhsa-runtime64.so.1, ...) onto the 7.x chain.
  The v6 ~/.Renviron cure is no longer needed (harmless if already set).
  If loading still fails, run: ./tools/make_rocm_kit.sh --doctor
  (dlopen probe + LD_DEBUG culprit extraction -- prints WHO pulls /opt/rocm in).

Failure handling (v9):
  any failing gate now QUARANTINES inst/lib/liblantern.so and
  torch-rocm-kit/lantern/lib/liblantern.so BEFORE exiting. `R CMD INSTALL .`
  goes through {torch}'s dummylantern target ("*** Skip building lantern.")
  and reinstalls the PREVIOUS inst/lib/liblantern.so: a stale pre-alignment
  binary is what resurrected the 6.4/7.2 chimera (undefined symbol ...
  version ROCR_1). After a quarantine the failure is loud (liblantern.so
  missing), never silent.

Path A -- dev loop (recommended for #907):
  devtools::load_all("$TREE")        # libs loaded from inst/
  # or: R CMD INSTALL "$TREE"

Path B -- clean install through install.R:
  Sys.setenv(TORCH_HOME  = "$TREE/inst")
  Sys.setenv(TORCH_URL   = "$LIBTORCH_DIR")
  Sys.setenv(LANTERN_URL = "$KIT_DIR/lantern")
  torch::install_torch()             # then restart R
  source("test_torch_rocm.R")

Startup prompt note (v15):
  library(torch) only offers to download when the INSTALLED package is
  missing lib/liblantern.so or lib/libtorch.so. Two rules keep it quiet:
    1. install_torch() wants the kit layout: LANTERN_URL=$KIT_DIR/lantern
       (it needs a dir holding lib/liblantern.so -- pointing it at
       build-lantern, a cmake build tree, fails with 'Could not install
       lantern from ...').
    2. once installed, put TORCH_INSTALL=0 in ~/.Renviron.
  v15 also patched this tree (remove with --unpatch):
    - src/lantern/src/Cuda.cpp (backup: Cuda.cpp.orig-make-rocm-kit):
      cuda_get_device_capability, cuda_memory_stats/summary/snapshot,
      cuda_empty_cache and cuda_runtime_version now run their REAL
      implementations on the ROCm wheel (upstream guards them behind
      __NVCC__ and throws 'only supported on CUDA runtimes' otherwise);
      the device RNG / JIT bridge (libcaffe2_nvrtc.so -> libamdhiprtc.so)
      is staged as well.
    - R/cuda.R (marked block): cuda_synchronize(NULL) now uses the current
      device (the -1 path crashes inside the wheel's HIP CUDAHooks) and
      cuda_runtime_version() decodes the HIP encoding (6.4.43482-style).
      Rerun R CMD INSTALL / devtools::load_all to pick these up.
EOF

warn "Reminder (v14): with the v12 artifacts the first kernel on gfx900 died with
'HIP error: invalid device function'. Doctor [8] lists the code-object targets the
strings heuristic finds in the staged fatbins -- gfx900 may appear among them yet
kernels can still fail at launch (bundle-id strings can over-match; only a real
launch proves coverage). If the smoke test dies there: HSA_OVERRIDE_GFX_VERSION=9.0.6
is the quick experiment (elementwise usually OK, GEMM may trap); the supported path
for full BLAS on Vega10 is the ROCm 5.7 container. Never mix ROCm 6.4 and 7.2 apt
packages, and never define __HIP_PLATFORM_NVIDIA__ together with __HIP_PLATFORM_AMD__
(the shim defines the AMD pair for the lantern target)."
