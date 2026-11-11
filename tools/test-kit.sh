#!/usr/bin/env bash
# =============================================================================
# test-kit.sh -- unit tests for make_rocm_kit.sh (Linux, bash 4+)
#
# The kit's pure/diagnostic helpers live in the marked "kit-functions" block so
# they can be tested without arming the EXIT trap or parsing the CLI. This
# script extracts that block and exercises it on synthetic fixtures.
#
# Run:  ./tools/test-kit.sh
#
# Last change: v23 -- tests for install_nvrtc_bridge and the effective-arch
# verdict.
# =============================================================================
set -uo pipefail
# binutils output is parsed by sed patterns written for the C locale.
export LC_ALL=C LANG=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT="$HERE/make_rocm_kit.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  ok   %s\n' "$*"; }
bad() { FAIL=$((FAIL + 1)); printf '  FAIL %s\n' "$*"; }
skip(){ printf '  skip %s\n' "$*"; }

# --- minimal harness for the sourced block -----------------------------------
log()  { :; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die()  { printf 'die: %s\n' "$*" >&2; exit 1; }

# SCRIPT_DIR is used by apply_hip_shim/apply_cuda_r_block/ensure_hip_compat.
SCRIPT_DIR="$HERE"

# The helpers are a standalone, side-effect-free library.
# shellcheck source=kit-lib.sh
source "$HERE/kit-lib.sh" || { printf 'FATAL: cannot source %s\n' "$HERE/kit-lib.sh" >&2; exit 1; }

for helper in strip_hip_shim apply_hip_shim patch_cuda_hip verdict_for_arch; do
  if ! declare -F "$helper" >/dev/null 2>&1; then
    printf 'FATAL: helper %s is not a function after sourcing kit-lib.sh\n' "$helper" >&2
    exit 1
  fi
done

have() { command -v "$1" >/dev/null 2>&1; }

# -----------------------------------------------------------------------------
# 1) strip_hip_shim: idempotent, keeps the rest, cleans trailing blanks
# -----------------------------------------------------------------------------
f="$TMP/CMakeLists.txt"
printf 'cmake_minimum_required(VERSION 3.18)\n\n%s\nbody\n%s\n\n\n' "$MARK_A" "$MARK_B" > "$f"
strip_hip_shim "$f"
[[ "$(cat "$f")" == "cmake_minimum_required(VERSION 3.18)" ]] \
  && ok "strip_hip_shim removes the block and trailing blanks" \
  || bad "strip_hip_shim left unexpected content: $(cat "$f")"
cp "$f" "$TMP/after1"
strip_hip_shim "$f"
cmp -s "$f" "$TMP/after1" && ok "strip_hip_shim is idempotent" \
  || bad "strip_hip_shim changed an already-stripped file"
printf 'plain\n' > "$f"
strip_hip_shim "$f"
[[ "$(cat "$f")" == "plain" ]] && ok "strip_hip_shim leaves marker-less files alone" \
  || bad "strip_hip_shim altered a marker-less file"

# -----------------------------------------------------------------------------
# 1.1/2.3) apply_hip_shim: single block on double apply + backup never clobbered
# -----------------------------------------------------------------------------
f="$TMP/CMakeLists.txt"
printf 'cmake_minimum_required(VERSION 3.18)\n' > "$f"
apply_hip_shim "$f"
apply_hip_shim "$f"
if [[ "$(grep -cF "$MARK_A" "$f")" -eq 1 ]]; then
  ok "apply_hip_shim appends exactly one block"
else
  bad "apply_hip_shim produced $(grep -cF "$MARK_A" "$f") blocks"
fi
if [[ -f "$f.orig-make-rocm-kit" ]] && grep -q '^cmake_minimum_required' "$f.orig-make-rocm-kit"; then
  ok "apply_hip_shim keeps a pristine backup"
else
  bad "apply_hip_shim backup missing or wrong"
fi
# drift: alter the working file, rerun, backup must stay pristine
printf 'upstream-edit\n' >> "$f"
apply_hip_shim "$f"
if ! grep -q 'upstream-edit' "$f.orig-make-rocm-kit"; then
  ok "apply_hip_shim never overwrites the pristine backup"
else
  bad "apply_hip_shim overwrote the pristine backup on a later run"
fi

# -----------------------------------------------------------------------------
# 1.4/2.7) ensure_soname_links + direct_needed/is_system_lib (real ELF)
# -----------------------------------------------------------------------------
if have gcc && have objdump; then
  d="$TMP/soname"; mkdir -p "$d"
  printf '#include <stdio.h>\nint kit_f(void){return puts("x");}\n' > "$TMP/x.c"
  gcc -shared -fPIC -Wl,--no-as-needed -lc -Wl,-soname,libfoo.so.1 \
      -o "$d/libfoo.so.1.2.3" "$TMP/x.c"
  ensure_soname_links "$d"
  if [[ -L "$d/libfoo.so.1" && "$(readlink "$d/libfoo.so.1")" == "libfoo.so.1.2.3" ]]; then
    ok "ensure_soname_links creates the SONAME symlink"
  else
    bad "ensure_soname_links did not create libfoo.so.1"
  fi
  if direct_needed "$d/libfoo.so.1.2.3" | grep -qx 'libc.so.6'; then
    ok "direct_needed extracts DT_NEEDED"
  else
    bad "direct_needed did not report libc.so.6"
  fi
  is_system_lib libc.so.6 && ok "is_system_lib recognizes libc" || bad "is_system_lib missed libc"
  is_system_lib libamdhip64.so.6 && bad "is_system_lib accepted libamdhip64" || ok "is_system_lib rejects ROCm libraries"
else
  skip "ensure_soname_links/direct_needed (gcc or objdump missing)"
fi

# -----------------------------------------------------------------------------
# 1.3) hermetize_libs: ignores non-ELF, sets DT_RPATH on real ELF
# -----------------------------------------------------------------------------
if have patchelf && have readelf && have gcc; then
  d="$TMP/herm"; mkdir -p "$d"
  printf 'not an elf\n' > "$d/libnotelf.so.0"
  printf 'int kit_h(void){return 0;}\n' > "$TMP/h.c"
  gcc -shared -fPIC -o "$d/libherm.so.1" "$TMP/h.c"
  hermetize_libs "$d"
  if readelf -d "$d/libherm.so.1" 2>/dev/null | grep -q 'Library rpath: \[\$ORIGIN\]'; then
    ok "hermetize_libs sets DT_RPATH=\$ORIGIN"
  else
    bad "hermetize_libs did not set DT_RPATH"
  fi
  # non-ELF must be untouched (no rpath can be added)
  if readelf -d "$d/libnotelf.so.0" >/dev/null 2>&1; then
    bad "hermetize_libs touched a non-ELF file"
  else
    ok "hermetize_libs ignores non-ELF files"
  fi
else
  skip "hermetize_libs (patchelf/readelf/gcc missing)"
fi

# -----------------------------------------------------------------------------
# 2.7) install_nvrtc_bridge: copies into both layouts with the right DT_RPATH
# -----------------------------------------------------------------------------
if have patchelf && have readelf && have gcc; then
  d="$TMP/nvrtc"; mkdir -p "$d/staged/lib" "$d/dev" "$d/kit"
  printf 'int kit_n(void){return 0;}\n' > "$TMP/n.c"
  gcc -shared -fPIC -o "$d/staged/lib/libcaffe2_nvrtc.so" "$TMP/n.c"
  LIBTORCH_DIR="$d/staged" install_nvrtc_bridge "$d/dev" dev >/dev/null 2>&1
  LIBTORCH_DIR="$d/staged" install_nvrtc_bridge "$d/kit" kit >/dev/null 2>&1
  if [[ -s "$d/dev/libcaffe2_nvrtc.so" && -s "$d/kit/libcaffe2_nvrtc.so" ]] \
     && readelf -d "$d/kit/libcaffe2_nvrtc.so" 2>/dev/null | grep -q 'Library rpath: \[\$ORIGIN\]' \
     && readelf -d "$d/dev/libcaffe2_nvrtc.so" 2>/dev/null | grep -qF "Library rpath: [\$ORIGIN:$d/staged/lib]"; then
    ok "install_nvrtc_bridge copies into both layouts with the right DT_RPATH"
  else
    bad "install_nvrtc_bridge did not stage both copies correctly"
  fi
else
  skip "install_nvrtc_bridge (patchelf/readelf/gcc missing)"
fi

# -----------------------------------------------------------------------------
# 1.5) audit_closure: a dependency resolving into ROCM_MARK is fatal
# -----------------------------------------------------------------------------
if have gcc && have ldd; then
  mkdir -p "$TMP/fakerocm"
  printf 'int kit_fake(void){return 0;}\n' > "$TMP/fake.c"
  gcc -shared -fPIC -Wl,-soname,libfake.so.1 -o "$TMP/fakerocm/libfake.so.1" "$TMP/fake.c"
  printf 'int kit_fake(void);\nint kit_bad(void){return kit_fake();}\n' > "$TMP/bad.c"
  gcc -shared -fPIC -o "$TMP/libbad.so.1" "$TMP/bad.c" \
      -L"$TMP/fakerocm" -l:libfake.so.1 -Wl,-rpath,"$TMP/fakerocm"
  out=$(
    ROCM_MARK="$TMP/fakerocm" ROCM_HOSTILE_LIB="" \
      audit_closure "$TMP/libbad.so.1" "$TMP/fakerocm" 2>&1
  ) && rc=0 || rc=$?
  if [[ "$rc" -ne 0 && "$out" == *"resolves dependencies from"* ]]; then
    ok "audit_closure rejects a dependency resolved from ROCM_MARK"
  else
    bad "audit_closure did not reject ROCM_MARK resolution (rc=$rc)"
  fi
else
  skip "audit_closure (gcc or ldd missing)"
fi

# -----------------------------------------------------------------------------
# 2.6) verdict_for_arch: override suggestion / covered / unknown
# -----------------------------------------------------------------------------
out=$(verdict_for_arch gfx900 "gfx906 gfx1100")
[[ "$out" == *"HSA_OVERRIDE_GFX_VERSION=9.0.6"* && "$out" == *"EXPERIMENTAL"* ]] \
  && ok "verdict_for_arch suggests the gfx900 experimental override" \
  || bad "verdict_for_arch missed the gfx900 override"
out=$(verdict_for_arch gfx90a "gfx90a gfx1100")
[[ "$out" == *"COVERED"* ]] && ok "verdict_for_arch reports covered archs" \
  || bad "verdict_for_arch did not report coverage"
out=$(verdict_for_arch gfx999 "gfx1100")
[[ "$out" == *"TORCH_ROCM_ARCH=gfx999"* ]] && ok "verdict_for_arch falls back to a source rebuild" \
  || bad "verdict_for_arch missed the rebuild fallback"
out=$(HSA_OVERRIDE_GFX_VERSION=9.0.0 verdict_for_arch gfx90c "gfx900 gfx1100" gfx900)
[[ "$out" == *"maps it to gfx900"* && "$out" == *"~/.Renviron"* ]] \
  && ok "verdict_for_arch explains a hardware arch covered via HSA_OVERRIDE" \
  || bad "verdict_for_arch missed the hardware/effective arch distinction"

# -----------------------------------------------------------------------------
# 1.2/1.8) patch_cuda_hip: patch, idempotence, drift guard, missing guards
# -----------------------------------------------------------------------------
make_cuda_cpp() {
  local guards="$1" path="$2" i
  {
    # include block (upstream: #ifdef __NVCC__ around the ATen/cuda headers)
    printf '#ifdef __NVCC__\n#include <torch/csrc/cuda/memory_snapshot.h>\n#endif\n\n'
    for i in $(seq 1 "$guards"); do
      printf '#ifdef __NVCC__\nint dummy_%d() { return %d; }\n#endif\n\n' "$i" "$i"
    done
    # upstream layout: the guard is INSIDE the function body
    printf 'int _lantern_cuda_get_runtime_version() {\n  LANTERN_FUNCTION_START\n#ifdef __NVCC__\n  int runtimeVersion;\n  cudaRuntimeGetVersion(&runtimeVersion);\n  return runtimeVersion;\n#else\n  throw std::runtime_error(\n      "`cuda_device_stats` is only supported on CUDA runtimes.");\n#endif\n  LANTERN_FUNCTION_END\n}\n\n'
    for i in 2 3 4 5 6; do
      printf 'const char* m%i = "msg is only supported on CUDA runtimes.";\n' "$i"
    done
  } > "$path"
}

if have python3; then
  LANTERN_SRC="$TMP/lanternsrc"
  LIBTORCH_DIR="$TMP/libtorch"
  mkdir -p "$LANTERN_SRC/src" "$LIBTORCH_DIR/lib"
  touch "$LIBTORCH_DIR/lib/libtorch_hip.so"
  FORCE=""
  # include guard + 3 dummies + runtime guard = 5 (< 6 requested)
  make_cuda_cpp 3 "$LANTERN_SRC/src/Cuda.cpp"
  out=$(patch_cuda_hip 2>&1) && rc=0 || rc=$?
  if [[ "$rc" -ne 0 && "$out" == *"found sites"* ]]; then
    ok "patch_cuda_hip reports the guard sites on a short tree (v1.2)"
  else
    bad "patch_cuda_hip short-tree failure was not informative (rc=$rc)"
  fi
  rm -f "$LANTERN_SRC/src/Cuda.cpp"* "$LIBTORCH_DIR/lib/libtorch_hip.so"
  touch "$LIBTORCH_DIR/lib/libtorch_hip.so"
  # include guard + 5 dummies + runtime guard = 7 (matches upstream torch 2.8)
  make_cuda_cpp 5 "$LANTERN_SRC/src/Cuda.cpp"
  if ( patch_cuda_hip ) >/dev/null 2>&1; then
    if grep -q 'hipRuntimeGetVersion' "$LANTERN_SRC/src/Cuda.cpp" \
        && [[ -f "$LANTERN_SRC/src/Cuda.cpp.kit-applied" ]]; then
      ok "patch_cuda_hip applies the LANTERN_HIP patch and writes the hash"
    else
      bad "patch_cuda_hip did not produce the expected output"
    fi
  else
    bad "patch_cuda_hip failed on a valid fixture"
  fi
  h1="$(file_hash "$LANTERN_SRC/src/Cuda.cpp")"
  ( patch_cuda_hip ) >/dev/null 2>&1 || true
  h2="$(file_hash "$LANTERN_SRC/src/Cuda.cpp")"
  if [[ "$h1" == "$h2" ]]; then
    ok "patch_cuda_hip is idempotent"
  else
    bad "patch_cuda_hip changed the file on rerun"
  fi
  # drift: local edit + rerun without --force must be preserved
  printf '/* local edit */\n' >> "$LANTERN_SRC/src/Cuda.cpp"
  ( patch_cuda_hip ) >/dev/null 2>&1 || true
  if grep -q 'local edit' "$LANTERN_SRC/src/Cuda.cpp"; then
    ok "patch_cuda_hip refuses to overwrite local edits without --force (v1.8)"
  else
    bad "patch_cuda_hip overwrote local edits without --force"
  fi
  FORCE=1
  ( patch_cuda_hip ) >/dev/null 2>&1 || true
  if ! grep -q 'local edit' "$LANTERN_SRC/src/Cuda.cpp"; then
    ok "patch_cuda_hip reapplies over local edits with --force"
  else
    bad "patch_cuda_hip ignored --force"
  fi
  FORCE=""
else
  skip "patch_cuda_hip (python3 missing)"
fi

# -----------------------------------------------------------------------------
printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
if [[ "$PASS" -eq 0 ]]; then
  printf 'FATAL: zero tests passed; refusing to report a green run\n' >&2
  exit 1
fi
[[ "$FAIL" -eq 0 ]] || exit 1
exit 0
