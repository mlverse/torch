#' Autocast context manager
#'
#' Allow regions of your code to run in mixed precision.
#' In these regions, ops run in an op-specific dtype chosen by autocast
#' to improve performance while maintaining accuracy.
#' 
#' When entering an autocast-enabled region, Tensors may be any type.
#' You should not call `half()` or `bfloat16()` on your model(s) or inputs 
#' when using autocasting.
#' 
#' `autocast` should only be enabled during the forward pass(es) of your network, 
#' including the loss computation(s).  Backward passes under autocast are not 
#' recommended. Backward ops run in the same type that autocast used for 
#' corresponding forward ops.
#' 
#' @param device_type a character string indicating whether to use 'cuda' or 'cpu' device
#' @param enabled a logical value indicating whether autocasting should be enabled in the region. Default: TRUE
#' @param dtype a torch data type indicating whether to use `torch_float16()` or `torch_bfloat16()`.
#' @param cache_enabled a logical value indicating whether the weight cache inside autocast should be enabled.
#' @param ... currently unused.
#' @inheritParams with_no_grad
#' @examples 
#' x <- torch_randn(5, 5, dtype = torch_float32())
#' y <- torch_randn(5, 5, dtype = torch_float32())
#' 
#' foo <- function(x, y) {
#'   local_autocast(device = "cpu")
#'   z <- torch_mm(x, y)
#'   w <- torch_mm(z, x)
#'   w
#' }
#' 
#' out <- foo(x, y)
#' @export
local_autocast <- function(device_type, dtype = NULL, enabled = TRUE, cache_enabled = NULL, ..., .env = parent.frame()) {
  device <- device_type
  
  fast_dtype <- if (!is.null(dtype)) {
    dtype
  } else if (device == "cpu") {
    cpp_amp_autocast_get_cpu_dtype()
  } else if (device == "cuda") {
    cpp_amp_autocast_get_gpu_dtype()
  } else {
    cli::cli_abort("Unsupported device {.val {device}}.")
  }
  
  cache_enabled <- if (!is.null(cache_enabled)) {
    cache_enabled
  } else {
    cpp_amp_autocast_is_cache_enabled()
  }
  
  if (device == "cpu") {
    prev_enabled <- cpp_amp_is_autocast_cpu_enabled()
    prev_fast_dtype <- cpp_amp_autocast_get_cpu_dtype()
    
    cpp_amp_autocast_set_cpu_enabled(enabled)
    cpp_amp_autocast_set_cpu_dtype(fast_dtype)
    cpp_amp_autocast_increment_nesting()
  } else if (device == "cuda") {
    prev_enabled <- cpp_amp_is_autocast_gpu_enabled()
    prev_fast_dtype <- cpp_amp_autocast_get_gpu_dtype()
    
    cpp_amp_autocast_set_gpu_enabled(enabled)
    cpp_amp_autocast_set_gpu_dtype(fast_dtype)
    cpp_amp_autocast_increment_nesting()
  } else {
    cli::cli_abort("Unsupported device {.val {device}}.")
  }
  
  prev_cache_enabled <- cpp_amp_autocast_is_cache_enabled()
  cpp_amp_autocast_set_cache_enabled(cache_enabled)
  
  withr::defer({
    if (device == "cpu") {
      if (cpp_amp_autocast_decrease_nesting() == 0) {
        cpp_amp_autocast_clear_cache()
      }
      cpp_amp_autocast_set_cpu_enabled(prev_enabled)
      cpp_amp_autocast_set_cpu_dtype(prev_fast_dtype)
    } else if (device == "cuda") {
      if (cpp_amp_autocast_decrease_nesting() == 0) {
        cpp_amp_autocast_clear_cache()
      }
      cpp_amp_autocast_set_gpu_enabled(prev_enabled)
      cpp_amp_autocast_set_gpu_dtype(prev_fast_dtype)
    }
  }, envir = .env)
}

#' @describeIn local_autocast A with context for automatic mixed precision.
with_autocast <- function(code, ... , device_type, dtype = NULL, enabled = TRUE, cache_enabled = NULL) {
  local_autocast(device_type, dtype = dtype, enabled = enabled, cache_enabled = cache_enabled)
  force(code)
}