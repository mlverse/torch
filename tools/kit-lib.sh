#!/usr/bin/env bash
# =============================================================================
# kit-lib.sh -- test-extractable helper library for make_rocm_kit.sh
#
# Sourced by make_rocm_kit.sh and by tools/test-kit.sh. It deliberately
# defines NO exit trap, NO CLI parsing and NO top-level side effects, so the
# tests can call the helpers directly on synthetic fixtures.
#
# Requires the caller to provide: log/warn/die (plus SCRIPT_DIR for the
# functions that read the extracted assets). Globals such as LIBTORCH_DIR,
# BUILD_DIR and LANTERN_SRC are read at call time.
# =============================================================================
# --- hip-shim block: markers + idempotent handling ----------------------------
MARK_A='# >>> hip-shim-rocm (make_rocm_kit.sh)'
MARK_B='# <<< hip-shim-rocm'

# 1.8 -- drift detection for the source patches: hash the PATCHED file so a
# later run can tell "still our output" from "the user edited it".
file_hash() {
  local h=""
  if command -v sha256sum >/dev/null 2>&1; then
    read -r h _ < <(sha256sum "$1")
  else
    read -r h _ < <(cksum "$1")
  fi
  printf '%s' "$h"
}

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
  local f="$1" shim="$SCRIPT_DIR/hip-shim.cmake"
  [[ -f "$f" ]] || die "CMakeLists.txt not found: $f"
  [[ -f "$shim" ]] || die "hip-shim.cmake not found next to this script: $shim"
  strip_hip_shim "$f"
  # 1.1 -- NEVER overwrite an existing pristine backup. The first run saves the
  # unpatched CMakeLists; a later run that restored/edited the file must not
  # silently replace that original with a post-restore version. Warn on drift
  # instead (upstream edits since the first run) and keep the backup as is.
  if [[ -f "$f.orig-make-rocm-kit" ]]; then
    cmp -s "$f" "$f.orig-make-rocm-kit" \
      || warn "$(basename "$f") differs from its pristine backup ($(basename "$f").orig-make-rocm-kit) -- upstream edits since the first kit run are preserved in the current file; the backup is kept unchanged."
  else
    cp "$f" "$f.orig-make-rocm-kit"
  fi
  # 2.3 -- the CMake block lives in tools/hip-shim.cmake (versioned, lintable,
  # testable); MARK_A/MARK_B stay at its boundaries so strip_hip_shim works.
  cat "$shim" >> "$f"
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
# v17: also snapshots the HIP SDK math headers the shims forward to (the
# wheel bundles only the hip/ runtime subset) and self-heals the closure via
# a g++ -fsyntax-only probe -- see CHANGELOG v17.
ensure_hip_compat() {
  [[ -f "$LIBTORCH_DIR/lib/libtorch_hip.so" ]] || return 0
  local dir="$BUILD_DIR/hip-compat"
  rm -rf "$dir"                       # clean regeneration on every kit run
  mkdir -p "$dir"
  # 2.2 -- the six shims live in tools/hip-compat/ (versioned, lintable,
  # testable); copy them into the build dir on every run.
  local hdr src_hdr
  local -a hdrs=(cuda.h cuda_runtime_api.h cublas_v2.h cublasLt.h cusparse.h cusolverDn.h)
  for hdr in "${hdrs[@]}"; do
    src_hdr="$SCRIPT_DIR/hip-compat/$hdr"
    [[ -f "$src_hdr" ]] || die "compat shim missing: $src_hdr (tools/hip-compat/ ships with the kit)"
    cp "$src_hdr" "$dir/$hdr"
  done

  # v17 -- the wheel ships NO math-library headers: the shims forward to
  # <hipsparse.h>/<hipblas.h>/<hipblaslt.h>/<hipsolver.h>, which exist only in
  # a system HIP SDK (field evidence, v16 run: the <hip/hip_runtime_api.h> of
  # the cuda_runtime_api.h shim resolved from the wheel's bundled
  # libtorch/include/hip/ one line BEFORE hipsparse.h died). Snapshot the SDK
  # copies into $dir (FIRST on lantern's include path): ordering-independent,
  # never shadows a header the wheel ships, and compile-time ONLY -- opaque
  # handles, nothing gets linked, the hermetic runtime chain is untouched.
  local sdk="" cand sdkver="(unknown version)" h missing_req="" copied=0
  local -a sdk_cands=()
  [[ -n "${ROCM_PATH:-}" ]] && sdk_cands+=("$ROCM_PATH/include")
  [[ -n "${HIP_PATH:-}" ]]  && sdk_cands+=("$HIP_PATH/include")
  sdk_cands+=("/opt/rocm/include")
  # any other local ROCm install (lexicographic reverse = newest first)
  while IFS= read -r cand; do sdk_cands+=("$cand"); done \
    < <(compgen -G '/opt/rocm-*/include' 2>/dev/null | sort -r || true)
  sdk_cands+=("/usr/include")
  for cand in "${sdk_cands[@]}"; do
    [[ -f "$cand/hipsparse.h" ]] && { sdk="$cand"; break; }
  done
  [[ -n "$sdk" ]] || die "no HIP SDK headers found (looked in: ${sdk_cands[*]})
     The cuda->HIP shims forward to hipsparse.h/hipblas.h/hipblaslt.h, which
     the wheel does not bundle. Install the ROCm SDK dev headers:
       sudo amdgpu-install --usecase=rocm    (or: sudo apt install rocm-dev)
     or set ROCM_PATH to an install that has <prefix>/include/hipsparse.h."
  [[ -f "$sdk/../.info/version" ]] && sdkver="$(<"$sdk/../.info/version")"

  for h in hipsparse.h hipblas.h hipblaslt.h; do
    if [[ -f "$LIBTORCH_DIR/include/$h" ]]; then
      echo "    [compat] $h already bundled by the wheel -- no snapshot needed"
    elif [[ -f "$sdk/$h" ]]; then
      cp "$sdk/$h" "$dir/$h"; copied=$((copied + 1))
    else
      missing_req+=" $h"
    fi
  done
  [[ -z "$missing_req" ]] || die "HIP SDK headers missing from $sdk:$missing_req
     (hipsparse.h + hipblas.h + hipblaslt.h are hard requirements: torch 2.8.0
     includes <cusparse.h>/<cublas_v2.h>/<cublasLt.h> unguarded and the shims
     forward to these). Install the ROCm SDK dev headers on this host
     (sudo amdgpu-install --usecase=rocm / apt install rocm-dev) or point
     ROCM_PATH at an install that has them."
  # hipsolver.h: the cusolverDn shim only fires under "#ifdef CUDART_VERSION"
  # (dead on HIP) -- snapshot it when available, warn otherwise.
  if [[ ! -f "$LIBTORCH_DIR/include/hipsolver.h" && -f "$sdk/hipsolver.h" ]]; then
    cp "$sdk/hipsolver.h" "$dir/hipsolver.h"; copied=$((copied + 1))
  elif [[ ! -f "$sdk/hipsolver.h" ]]; then
    warn "hipsolver.h not in $sdk -- fine unless some header includes <cusolverDn.h> outside CUDART_VERSION"
  fi
  (( copied > 0 )) && warn "snapshotted $copied math-header(s) from the system HIP SDK $sdk ($sdkver) into hip-compat -- compile-time ONLY (opaque handles), the wheel's own runtime chain stays authoritative"

  # v17 -- syntax-compile the EXACT torch cuda-header closure NOW and self-heal
  # every remaining missing SDK header (math headers pull more SDK headers
  # transitively, e.g. the roc* family or hip/hip_version.h). The probe and
  # its logs stay in hip-compat/ for manual reruns.
  local probe="$dir/compat-probe.cpp" plog="$dir/compat-probe.log"
  cat > "$probe" <<'EOF'
/* make_rocm_kit.sh v17 -- cuda->HIP compat probe: the torch 2.8.0 header
 * closure of lantern's src/Cuda.cpp. Compiled with -fsyntax-only by the kit
 * (g++ <defines> <includes> compat-probe.cpp) so a missing/broken HIP SDK
 * header dies HERE, in seconds, instead of at 51% inside make. */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
EOF
  # cusolverDn.h is a dead include on HIP (CUDART_VERSION undefined): probe it
  # only when its hipsolver.h counterpart is actually resolvable.
  if [[ -f "$dir/hipsolver.h" || -f "$LIBTORCH_DIR/include/hipsolver.h" ]]; then
    echo '#include <cusolverDn.h>' >> "$probe"
  fi
  # and only the closure headers the STAGED tree actually ships (torch 2.8.0
  # wheels carry all of them; fake/partial stagings must not false-kill the
  # post-configure gate on headers that would never be compiled anyway).
  local _ti
  for _ti in ATen/cuda/CUDAContext.h ATen/cuda/detail/CUDAHooks.h \
             c10/cuda/CUDAFunctions.h c10/cuda/CUDAStream.h \
             c10/cuda/CUDACachingAllocator.h; do
    [[ -f "$LIBTORCH_DIR/include/$_ti" ]] && echo "#include <$_ti>" >> "$probe"
  done
  echo 'int main() { return 0; }' >> "$probe"
  if command -v g++ >/dev/null 2>&1; then
    local -a pflags=(-std=c++17 -fsyntax-only "-I$dir"
                     "-I$LIBTORCH_DIR/include"
                     "-I$LIBTORCH_DIR/include/torch/csrc/api/include"
                     -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__
                     -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE -DUSE_ROCM -DLANTERN_HIP=1)
    local attempt=0 probe_ok=0 mh src_h
    while (( attempt < 15 )); do
      if g++ "${pflags[@]}" "$probe" 2>"$plog"; then probe_ok=1; break; fi
      mh="$(grep -m1 -oE 'fatal error: [^ :]+: No such file or directory' "$plog" \
            | sed -E 's/^fatal error: //; s/: No such file or directory$//' || true)"
      [[ -n "$mh" ]] || break
      src_h=""
      [[ -f "$sdk/$mh" ]] && src_h="$sdk/$mh"
      [[ -n "$src_h" ]] || src_h="$(find "$sdk" -name "$(basename "$mh")" 2>/dev/null | head -n1 || true)"
      [[ -n "$src_h" ]] || break
      mkdir -p "$dir/$(dirname "$mh")"
      cp "$src_h" "$dir/$mh"
      echo "    [compat] self-heal: snapshot $mh <- $src_h"
      copied=$((copied + 1)); attempt=$((attempt + 1))
    done
    if (( probe_ok )); then
      echo "    [compat] probe: torch cuda-header closure compiles (g++ -fsyntax-only; $copied SDK header(s) snapshotted in total)"
    else
      warn "compat probe still failing -- full g++ output in $plog (informational: the post-configure gate re-judges with the real lantern flags)"
    fi
  else
    warn "g++ not found -- compat probe skipped (cmake will judge)"
  fi

  echo "    [compat] wrote 6 cuda->HIP shim headers in $dir + HIP SDK snapshot/probe (torch 2.8.0 closure: 26 cuda* tokens + 4 handle aliases)"
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
  local -A seen=()
  [[ -d "$dir" ]] || return 0
  for f in "$dir"/*.so*; do
    [[ -e "$f" ]] || continue                # unmatched glob or dangling link
    [[ -L "$f" ]] && continue                # only real files carry a SONAME
    base="$(basename "$f")"
    soname="$(objdump -p "$f" 2>/dev/null | awk '$1=="SONAME"{print $2; exit}' || true)"
    [[ -n "$soname" ]] || continue
    # 1.4 -- two real files may (rarely) carry the same SONAME; the loop order
    # would silently decide which one the symlink points at. Name them.
    if [[ -n "${seen[$soname]:-}" && "${seen[$soname]}" != "$base" ]]; then
      warn "duplicate SONAME $soname in $dir: ${seen[$soname]} and $base (first one wins)"
    fi
    seen[$soname]="$base"
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

# 2.7 -- shared helpers: the two gates below used to duplicate the readelf
# extraction and the system-library pattern list (a divergence there is a
# subtle bug: one gate would reject what the other accepts).
direct_needed() {
  readelf -d "$1" 2>/dev/null | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p'
}

is_system_lib() {
  case "$1" in
    libc.so*|libm.so*|libdl.so*|librt.so*|libpthread.so*|libstdc++.so*|libgcc_s.so*|ld-linux*) return 0 ;;
  esac
  return 1
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
    is_system_lib "$needed" && continue
    if [[ ! -e "$libdir/$needed" ]]; then
      missing+=" $needed"
    fi
  done < <(direct_needed "$lib")
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
    is_system_lib "$needed" && continue
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
  done < <(direct_needed "$lib")
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
  local applied="$LANTERN_SRC/src/Cuda.cpp.kit-applied"
  [[ -f "$f" ]] || { echo "    [patch] $f not found -- skipped (older lantern layout)"; return 0; }
  [[ -f "$LIBTORCH_DIR/lib/libtorch_hip.so" ]] || { echo "    [patch] not a ROCm wheel staging -- Cuda.cpp untouched"; return 0; }
  if grep -q 'LANTERN_HIP' "$f"; then
    if [[ -f "$bak" ]]; then
      # 1.8 -- cp "$bak" "$f" would mask edits made between two kit runs. A
      # hash of the last patched output distinguishes "our own artifact" from
      # "the user touched it": refuse unless --force was passed.
      if [[ -f "$applied" && -z "$FORCE" ]]; then
        if [[ "$(file_hash "$f")" != "$(<"$applied")" ]]; then
          warn "Cuda.cpp was modified after the last kit run (hash mismatch) -- refusing to overwrite your edits. Reapply with --force (restores the pristine backup first), or restore your VCS copy."
          return 0
        fi
      fi
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

def guard_sites(text):
    """Dump every '#ifdef __NVCC__' site with the signature that follows it."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if line.startswith("#ifdef __NVCC__"):
            sig = "?"
            for j in range(i + 1, min(i + 40, len(lines))):
                s = lines[j].strip()
                if s and not s.startswith(("#", "//", "/*", "*")):
                    sig = s
                    break
            out.append("  line %d: %s  ->  %s" % (i + 1, line, sig))
    return "\n".join(out) or "  (none)"

# 1) widen every __NVCC__ guard (include block + 6 function bodies)
src, n_guards = re.subn(r'^#ifdef __NVCC__',
                        '#if defined(__NVCC__) || defined(LANTERN_HIP)',
                        src, flags=re.M)
if n_guards < 6:
    fail("expected >= 6 '#ifdef __NVCC__' guards, found %d\n"
         "found sites:\n%s\n"
         "(tree layout changed? restore the backup and report)" % (n_guards, guard_sites(open(path, encoding="utf-8").read())))

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
    found = re.findall(r'[^\n]*is only supported on CUDA runtimes\.', src)
    fail("expected exactly 6 error messages to reword, got %d\n"
         "found sites:\n%s" % (n_msgs, "\n".join("  " + s.strip() for s in found) or "  (none)"))

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
  file_hash "$f" > "$applied"          # drift marker for the next run (1.8)
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
  # 2.4 -- the R block lives in tools/cuda-hip-rocm.R (versioned, parseable).
  cat "$SCRIPT_DIR/cuda-hip-rocm.R" >> "$f"
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
# there is no hiprtc at all (RNG state + JIT stay broken; doctor::nvrtc).
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
  # 1.3 -- one pass (the old code ran readelf -h twice per library; it is the
  # expensive part on 200+ files).
  for f in "$dir"/*.so*; do
    [[ -f "$f" && ! -L "$f" ]] || continue
    readelf -h "$f" >/dev/null 2>&1 || continue
    tot=$((tot+1))
    if ! patchelf --force-rpath --set-rpath '$ORIGIN' "$f" 2>/dev/null; then
      warn "hermetize: patchelf failed on $(basename "$f") -- the audits below will judge"
      continue
    fi
    n=$((n+1))
    [[ $((n % 20)) -eq 0 ]] && echo "    ... $n hermetized so far"
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
# 1.5 -- scrub hostile loader hooks from ALL regimes: LD_PRELOAD / LD_AUDIT
# inherited from the caller's shell would instrument (or fake) the ldd run.
run_ldd() {
  local spec="${1:-}" lib="$2"
  if [[ -n "$spec" ]]; then
    env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT "$spec" ldd "$lib" 2>&1 || true
  else
    env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT ldd "$lib" 2>&1 || true
  fi
}

# 2.5 -- shared trace checks for the three regimes.
check_trace() {
  local lib="$1" regime="$2" trace="$3" nf bad
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
}

audit_closure() {
  local lib="$1" libdir="$2" regime trace
  for regime in pristine forced hostile; do
    case "$regime" in
      pristine) trace=$(run_ldd "" "$lib") ;;
      forced)   trace=$(run_ldd "LD_LIBRARY_PATH=$libdir" "$lib") ;;
      hostile)
        if [[ -z "${ROCM_HOSTILE_LIB:-}" || ! -d "${ROCM_HOSTILE_LIB:-}" ]]; then
          warn "hostile audit skipped (ROCM_HOSTILE_LIB=${ROCM_HOSTILE_LIB:-<unset>} not present)"
          continue
        fi
        trace=$(run_ldd "LD_LIBRARY_PATH=$ROCM_HOSTILE_LIB" "$lib") ;;
    esac
    check_trace "$lib" "$regime" "$trace"
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
# 2.6 -- one device's verdict (HSA_OVERRIDE_MAP is a global associative array).
# Extracted so it can be tested in isolation: verdict_for_arch gfx900 "gfx90a ..."
verdict_for_arch() {
  local arch="$1" wa="$2" ov tag sib
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
}

doctor_gpu_arch() {
  local rinfo gpus="" arch line wa="" files=() labels=() arcs=() lst i f wv="" rf seen_rf=""
  echo "  [gpu_arch] GPU ISA vs wheel code objects ('invalid device function' layer)"
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
    verdict_for_arch "$arch" "$wa"
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
# real Rscript library(torch). v13 adds doctor_gpu_arch: GPU ISA vs the code objects
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
        echo "      (real failure -- culprit extraction in [hostile_probe])"
      fi
    fi
  done
  echo "  [hostile_probe] dlopen probe (RTLD_NOW), hostile LD_LIBRARY_PATH=${ROCM_HOSTILE_LIB:-/opt/rocm/lib}"
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
  echo "  [nvrtc] nvrtc/hiprtc bridge (device RNG state + JIT kernel compilation)"
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
  echo "  [install] installed {torch} package + install env vars (shell view; vars set in ~/.Renviron are R-only)"
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
  echo "  [r_probe] R-faithful probe (Rscript + LD_DEBUG)"
  if command -v Rscript >/dev/null 2>&1 && [[ -n "${cands[0]:-}" && -f "${cands[0]}" ]]; then
    env LD_DEBUG=libs,files Rscript --vanilla -e 'library(torch)' >/dev/null 2>"$dbg" || true
    { grep -E 'needed by|required by' "$dbg" 2>/dev/null | grep -E 'amdhip64|hsa-runtime64' ;
      grep -E 'needed by|required by' "$dbg" 2>/dev/null | grep -F "$ROCM_MARK" ; } | sort -u | sed 's/^/      /' | head -20 || echo "      (no rocm chain in the R trace)"
    echo "      -- full log: $dbg"
  else
    echo "      (Rscript or installed torch not available)"
  fi
  log "DOCTOR done. If hostile_probe/r_probe name a culprit, rerun the kit (v10 hermetizes the tree). If they stay clean but gpu_arch says NOT covered for your gfx target, that is the layer ABOVE loading: the wheel has no code object for your GPU ISA (HSA_OVERRIDE_GFX_VERSION with a same-generation sibling, or rebuild libtorch with TORCH_ROCM_ARCH=<target> -- nothing the kit can fix). nvrtc judges the nvrtc/hiprtc bridge, install the installed package + install env vars. Paste this whole output if unsure."
}

# --- 3.2: staging dispatch ----------------------------------------------------
# The MODE/SRC pair selects one of these (tiny table in the 1/4 step below).
stage_from_venv() {
  local venv torch_pkg py
  venv="$(cd "$1" && pwd -P)"
  torch_pkg=$(compgen -G "$venv/lib/python3*/site-packages/torch" | head -n1)
  [[ -n "$torch_pkg" ]] || die "torch package not found in $venv"
  py="$venv/bin/python"
  [[ -x "$py" ]] || die "python interpreter missing: $py"
  echo "torch $( "$py" -c 'import torch; print(torch.__version__)' ) / HIP $( "$py" -c 'import torch; print(getattr(torch.version, "hip", None))' )"
  rsync -a --exclude "__pycache__" "$torch_pkg/" "$LIBTORCH_DIR/"
  # install.R expects lib/libtorch.so; pip wheels only ship libtorch_cpu.so.
  ln -sf libtorch_cpu.so "$LIBTORCH_DIR/lib/libtorch.so"
}

stage_from_zip() {
  local src="$1" tmp sub
  [[ -f "$src" ]] || die "Zip not found: $src"
  tmp=$(mktemp -d)
  unzip -q "$src" -d "$tmp"
  sub=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n1)
  # empty the staging dir then move the zip content into it
  rm -rf "$LIBTORCH_DIR" && mkdir -p "$LIBTORCH_DIR"
  # v12: dotglob+nullglob inside a subshell -- the old `mv "$sub"/* "$sub"/.[!.]*`
  # died on wheels without dotfiles (the literal glob fails the whole mv), and
  # an empty archive must fail LOUDLY instead of tripping set -e mid-glob.
  if [[ -n "$sub" ]]; then
    ( shopt -s dotglob nullglob; entries=("$sub"/*); [[ ${#entries[@]} -gt 0 ]] || exit 3; mv "${entries[@]}" "$LIBTORCH_DIR/" ) \
      || die "zip staging failed (empty or unreadable archive?)"
  else
    ( shopt -s nullglob; entries=("$tmp"/*); [[ ${#entries[@]} -gt 0 ]] || exit 3; mv "${entries[@]}" "$LIBTORCH_DIR/" ) \
      || die "zip staging failed (empty or unreadable archive?)"
  fi
  rm -rf "$tmp"
}

