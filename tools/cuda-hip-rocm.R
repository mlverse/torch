
# >>> cuda-hip-rocm (make_rocm_kit.sh) -- self-managed block (remove with: tools/make_rocm_kit.sh --unpatch)
# Last change: v15 -- ROCm/HIP fixes for the cuda_* API; these definitions override the
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
