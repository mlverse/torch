
# >>> hip-shim-rocm (make_rocm_kit.sh) -- self-managed block (remove with: tools/make_rocm_kit.sh --unpatch)
# Last change: v16 -- cuda->HIP compat headers first on the include path (+ USE_ROCM).
# ALIGN the HIP runtime and the roc* math libraries with the staged wheel:
# torch's CMake config (LoadHIP) points hip::* at the SYSTEM ROCm, while
# libc10_hip/libtorch_cuda from the wheel embed their own ROCm chain. Linking
# the system libamdhip64 on top mixes two runtimes: at load time, the
# already-loaded libhsa does not export the recent ROCR_1 symbol required by
# the system library -> "undefined symbol:
# hsa_amd_memory_get_preferred_copy_engine". The runtime and the math libs
# (hipblas, hipsparse, hipsolver, hipblaslt, ...) are therefore re-aligned to
# the wheel: the staged lib dir is searched FIRST for plain -l<name> items,
# bare-name imported targets are created/redirected to the wheel files, and
# any imported target still pointing outside the wheel (roc::X, hip::X,
# full-path locations) is re-aligned to the same-family wheel library. If the
# targets do not exist (wheel-only machine), create them: wheel first, system
# ROCm as a last resort.
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
# make sure plain -l<name> items resolve to the staged wheel first.
if(TARGET lantern)
  target_link_directories(lantern BEFORE PRIVATE "${TORCH_PATH}/lib")
endif()
# bare-name imported targets for the known ROCm math libraries.
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
# sweep every imported target still pointing outside the staged
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
# COMPILE definitions for a ROCm wheel. Two gaps the wheel cannot fill,
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
# enable lantern's CUDA-flavored implementations on the ROCm wheel.
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
# cuda->HIP compat headers FIRST on lantern's include path + USE_ROCM.
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
