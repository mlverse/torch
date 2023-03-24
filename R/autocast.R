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

amp_GradScaler <- R6Class(
  "AmpGradScaler", 
  public = list(
    initialize = function(init_scale=2.^16, growth_factor=2.0, backoff_factor=0.5,
                          growth_interval=2000, enabled=TRUE) {
      self$.enabled <- enabled
      if (self$.enabled) {
        
        if (growth_factor <= 1) 
          cli::cli_abort("{.var growth_factor} should be > 1 but got {.val {growth_factor}}.")
        
        if (backoff_factor >= 1)
          cli::cli_abort("{.var backoff_factor} should be < 1 but got {.val {backoff_factor}}.")
        
        self$.init_scale <- init_scale
        self$.scale <- NULL
        self$.growth_factor <- growth_factor
        self$.backoff_factor <- backoff_factor
        self$.growth_interval <- growth_interval
        self$.init_growth_tracker <- 0
        # sel$._growth_tracker will be lazily initialized during the first call to scale()
        self$.growth_tracker <- NULL
        self$.per_optimizer_states <- amp_OptState$new()
      }
    },
    scale = function(outputs) {
      # Short-circuit for the common case.
      if (inherits(outputs, "torch_tensor")) {
        if (!outputs$is_cuda)
          cli::cli_abort("{.var outputs} device must be {.val cuda}, got {.val {outputs$device$type}}.")
        
        if (is.null(self$.scale)) {
          self$.lazy_init_scale_growth_tracker(outputs$device)
        }
        
        return(outputs * self$.scale$to(device = outputs$device, non_blocking = TRUE))
      }
      
      # Invoke the more complex machinery only if we're treating multiple outputs.
      if (is.list(outputs))
        lapply(outputs, self$.scale)
      else
        cli::cli_abort("{.var outputs} must be a tensor or a list of tensors, got {.cls {class(outputs)}}.")
    },
    unscale_ = function(optimizer) {
      if (!self$.enabled) return(invisible(NULL))
      self$.check_scale_growth_tracker("unscale_")
      
      optimizer_state <- self$.per_optimizer_states[[rlang::obj_address(optimizer)]]
      
      if (optimizer_state[["stage"]] == "unscaled") {
        cli::cli_abort("{.fn unscale_} has already been called on this optimizer since the last {.fn update}.")
      } else if (optimizer_state[["stage"]] == "stepped") {
        cli::cli_abort("{.fn unscale_} is being called after {.fn step}.")
      }
      
      # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
      inv_scale <- self$.scale$double()$reciprocal()$float()
      found_inf <- torch_full(list(), 0.0, dtype=torch_float32(), device=self$.scale$device)
      
      optimizer_state[["found_inf"]] <- self$.unscale_grads_(optimizer, inv_scale, found_inf, FALSE)
      optimizer_state[["stage"]] <- "unscaled"
    },
    step = function(optimizer, ...) {
      optimizer_state <- self._per_optimizer_states[rlang::obj_address(optimizer)]
      if (optimizer_sate$stage == "stepped") {
        cli::cli_abort("{.fn step} has already been called since the last {.fn update}.")
      }

      if (optimizer_state$stage == "ready") {
        self$unscale_(optimizer)
      }
    
      retval <- self.maybe_opt_step(optimizer, optimizer_state, ...)
      optimizer_state$stage <- "stepped"
      retval
    },
    update = function(new_scale = NULL) {

    }
    .lazy_init_scale_growth_tracker = function(dev) {
      if (!is.null(self$.growth_tracker))
        cli::cli_abort("{.var .growth_tracker} initialized before {.var .scale}")
      
      self$.scale <- torch_full(size = list(), self$.init_scale, dtype = torch_float32(), device = dev)
      self$.growth_tracker <- torch_full(size = list(), self$.init_growth_tracker, dtype = torch_int32(), device = dev)
    },
    .check_scale_growth_tracker = function(funcname) {
      fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration."
      if (is.null(self$.scale)) {
        cli::cli_abort(c(
          "Attempted {.fn {funcname}} but {.var .scale} is {.val NULL}.",
          fix
        ))
      }
      if (is.null(self$.growth_tracker)) {
        cli::cli_abort(c(
          "Attempted {.fn {funcname}} but {.var .growth_tracker} is {.val NULL}.",
          fix
        ))
      }
      list(self$.scale, self$.growth_tracker)
    },
    .unscale_grads = function(optimizer, inv_scale, found_inf, allow_fp16) {
      local_no_grad()
      found <- 0
      for (group in optimizer$param_groups) {
        found <- found + cpp_amp_foreach_non_finite_check_and_unscale(group$params)
      }
      found
    },
    .maybe_opt_step = function(optimizer, optimizer_state, ...) {
      if (!(optimizer_state$found_inf > 0)) {
        optimizer$step(...)
      } else {
        invisible(NULL)
      }      
    }
  )
)


amp_OptState <- R6::R6Class(
  "AmpOptState",
  lock_objects = FALSE,
  public = list(
    initialize = function() {
      self$stage <- "ready"
      self$found_inf <- FALSE
    }
  )
)