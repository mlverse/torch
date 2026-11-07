#!/usr/bin/env bash
# =============================================================================
# make_rocm_kit.sh v17 -- build liblantern with ROCm, aligned with the {torch}
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
#   ./tools/make_rocm_kit.sh --force      (reapply source patches over local edits)
#   options: --tree <torch-root>     (default: auto-detected from tools/)
#            --lantern-src <dir>     (default: <tree>/src/lantern; the legacy
#                                     top-level lantern/ of platform/amd_gpu is
#                                     obsolete on the upstream side)
#

# CHANGELOG (full design notes: tools/make_rocm_kit.CHANGELOG.md)
#   v17 -- snapshot HIP SDK math headers + g++ -fsyntax-only self-heal probe
#   v16 -- cuda->HIP compat shims (tools/hip-compat/) + USE_ROCM
#   v15 -- LANTERN_HIP: real cuda_* implementations + nvrtc bridge + R fixes
#   v14 -- __HIP_PLATFORM_AMD__/__HIP_PLATFORM_HCC__ + C10_CUDA_NO_CMAKE_CONFIGURE_FILE
#   v13 -- doctor: GPU ISA vs wheel code objects ("invalid device function")
#   v12 -- escape $ORIGIN in the summary; FRESH_OK; end-to-end kit test
#   v11 -- doctor: staging inventory, /opt/rocm inventory, key-rpath list
#   v10 -- hermetize (DT_RPATH=$ORIGIN) + three-regime closure audit + doctor
#   v9  -- quarantine stale artifacts on failure; full closure audit
#   v8  -- align the roc* math libraries with the wheel copies
#   v7  -- SONAME symlink repair + exact-NAME dependency gate + LC_ALL=C
#   v6  -- build/install RPATH for installed liblantern copies
#   v5  -- English translation (ASCII-only); no logic change
#   v4  -- link the HIP runtime EMBEDDED in the wheel, not the system ROCm
#   v3  -- hip-shim as a post-find_package(Torch) conditional fallback
#
# Platform: Linux only (rocminfo, patchelf, readlink -f, GNU binutils).
# =============================================================================
set -euo pipefail

# v7: parse binutils output (readelf/objdump) in the C locale. On a localized
# system (e.g. LANG=fr_FR.UTF-8) readelf prints "Bibliotheque partagee : [...]"
# instead of "Shared library: [...]" and every sed-based guard below silently
# matched nothing. LANG is exported too: some readelf/objdump builds consult
# LANG rather than LC_ALL. Script messages stay ASCII English (upstream-ready).
export LC_ALL=C
export LANG=C

MODE=""; SRC=""; TREE_ARG=""; LANTERN_ARG=""; UNPATCH=""; DOCTOR=""; FORCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-venv)    MODE="venv"; SRC="${2:?}"; shift 2 ;;
    --from-zip)     MODE="zip";  SRC="${2:?}"; shift 2 ;;
    --tree)         TREE_ARG="${2:?}"; shift 2 ;;
    --lantern-src)  LANTERN_ARG="${2:?}"; shift 2 ;;
    --unpatch)      UNPATCH=1; shift ;;
    --doctor)       DOCTOR=1; shift ;;
    --force)        FORCE=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
[[ -z "$MODE" && -z "$UNPATCH" && -z "$DOCTOR" ]] && { echo "Usage: $0 --from-venv <venv> | --from-zip <libtorch.zip> | --unpatch | --doctor [--force]"; exit 1; }

# The extracted assets (kit-lib.sh, hip-compat/, hip-shim.cmake,
# cuda-hip-rocm.R) live next to this script, whether or not --tree is used.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*"; exit 1; }

# >>> kit-functions (test-extractable block: pure functions only) --------------
# 3.1 -- the helpers live in tools/kit-lib.sh so tools/test-kit.sh can source
# them without arming the EXIT trap or parsing the CLI.
# shellcheck source=kit-lib.sh
source "$SCRIPT_DIR/kit-lib.sh"
# <<< kit-functions -------------------------------------------------------------

# --- Resolve the {torch} tree root --------------------------------------------
if [[ -n "$TREE_ARG" ]]; then
  TREE="$(cd "$TREE_ARG" && pwd -P)"
else
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
    rm -f "$LANTERN_SRC/src/Cuda.cpp.kit-applied"
    echo "restored $LANTERN_SRC/src/Cuda.cpp from the kit backup"
  fi
  if grep -qF -- "$MARK_R_A" "$TREE/R/cuda.R" 2>/dev/null; then
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

case "$MODE" in
  venv) stage_from_venv "$SRC" ;;
  zip)  stage_from_zip  "$SRC" ;;
  *)    die "unknown staging mode: $MODE" ;;
esac

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
# 1.9 -- the same nvcc branches can also be selected through these variables
# in torch's LoadHIP.cmake, so clear them together (unset -v works under set -u).
unset -v CUDA CUDA_HOME CUDA_PATH TORCH_CUDA_ARCH_LIST 2>/dev/null || true

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
  grep -qF -- 'hip-compat' "$FLAGS_MAKE" \
    || die "hip-compat include dir missing from lantern's compile flags -- hip-shim block did not apply"
  echo "    flags check: USE_ROCM + hip-compat include dir present (ok)"
  # v17: pre-build gate with the REAL compile flags (the exact -D/-I/-std set
  # make will use): syntax-compile the torch cuda-header closure NOW, so any
  # residual breakage dies with the full g++ output instead of at 51% inside
  # make (field: hipsparse.h died exactly there in the v16 run). Skipped when
  # the staged tree ships NO torch cuda headers at all (dummy stagings have
  # nothing to probe; real torch wheels always carry ATen/cuda/CUDAContext.h --
  # exactly the surface that died at 51%).
  # (top-level code: plain variables, no `local`)
  if [[ ! -f "$LIBTORCH_DIR/include/ATen/cuda/CUDAContext.h" ]]; then
    echo "    flags probe: skipped (staged tree ships no ATen/cuda headers -- nothing to probe)"
  elif ! command -v g++ >/dev/null 2>&1; then
    warn "g++ not found -- flags probe skipped (make will judge)"
  else
    KIT_PROBE_FILE="$BUILD_DIR/hip-compat/compat-probe.cpp"
    [[ -f "$KIT_PROBE_FILE" ]] || die "compat probe missing ($KIT_PROBE_FILE) -- ensure_hip_compat did not run on this staging?"
    KIT_D=(); KIT_I=()
    read -r -a KIT_D < <(sed -n 's/^CXX_DEFINES = //p' "$FLAGS_MAKE") || true
    read -r -a KIT_I < <(sed -n 's/^CXX_INCLUDES = //p' "$FLAGS_MAKE") || true
    KIT_STD="$(sed -n 's/^CXX_FLAGS = //p' "$FLAGS_MAKE" | grep -o -- '-std=[a-z0-9+]*' | head -n1 || true)"
    if g++ "${KIT_D[@]}" "${KIT_I[@]}" ${KIT_STD:+"$KIT_STD"} -fsyntax-only "$KIT_PROBE_FILE" \
         2>"$BUILD_DIR/hip-compat/flags-probe.log"; then
      echo "    flags probe: torch cuda-header closure compiles with the REAL lantern flags (ok)"
    else
      die "the torch cuda-header closure does NOT compile with the real lantern flags
     (this is exactly how make would fail at ~51% on src/Cuda.cpp.o). Full g++
     output: $BUILD_DIR/hip-compat/flags-probe.log
     A missing hip*/roc* header here means the HIP SDK headers could not be
     snapshotted: install them (sudo amdgpu-install --usecase=rocm / apt
     install rocm-dev) or set ROCM_PATH, then rerun this script."
    fi
    unset KIT_D KIT_I KIT_STD KIT_PROBE_FILE
  fi
fi
cmake --build "$BUILD_DIR" --target lantern --parallel "$(nproc 2>/dev/null || echo 2)"

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

# 1.6 -- arm "keep the audited artifacts" only once BOTH copies are really
# there and non-empty. A full disk / permission error at cp time must still
# quarantine, otherwise an incomplete artifact would survive.
if [[ -s "$TREE/inst/lib/liblantern.so" && -s "$KIT_DIR/lantern/lib/liblantern.so" ]]; then
  FRESH_OK=1
else
  die "fresh liblantern copies are missing/empty after staging -- FRESH_OK not armed"
fi

# --- Summary -------------------------------------------------------------------
cat <<EOF

make_rocm_kit.sh -- build OK, artifacts aligned and audited.

  build-lantern/libtorch/     -> $LIBTORCH_DIR   (TORCH_URL candidate)
  build-lantern/hip-compat/   -> cuda->HIP shims + HIP SDK math headers + probes
  build-lantern/liblantern.so -> $LIBLANTERN
  inst/lib/liblantern.so      -> dev loading (load_all)
  torch-rocm-kit/lantern/     -> $KIT_DIR/lantern   (LANTERN_URL candidate)

  staged tree is hermetic (DT_RPATH=\$ORIGIN); the source patches live in
  src/lantern/src/Cuda.cpp and R/cuda.R (remove with --unpatch).

  Full notes (load-time, failure handling, Path A/B, startup prompt, GPU ISA):
    tools/README-rocm.md
  Design history (one section per version):
    tools/make_rocm_kit.CHANGELOG.md

  If loading fails: ./tools/make_rocm_kit.sh --doctor
EOF

warn "Reminder: a successful build does NOT prove your GPU ISA is covered by the
wheel (see doctor::gpu_arch). If the smoke test dies with 'invalid device
function', try HSA_OVERRIDE_GFX_VERSION=9.0.6 for gfx900 (elementwise usually
OK, GEMM may trap); full BLAS on Vega10 needs the ROCm 5.7 container. Never mix
ROCm 6.4 and 7.2 apt packages, and never define __HIP_PLATFORM_NVIDIA__ together
with __HIP_PLATFORM_AMD__ (the shim defines the AMD pair for lantern)."
