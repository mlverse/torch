Generator <- R7Class(
  classname = "torch_generator",
  public = list(
    ptr = NULL,
    initialize = function(ptr = NULL) {
      if (!is.null(ptr)) {
        return(ptr)
      }

      cpp_torch_generator()
    },
    print = function() {
      cat("torch_generator()")
    },
    current_seed = function() {
      if (!requireNamespace("bit64")) {
        warning("bit64 is required to correctly show the seed.")
      }

      bit64::as.integer64(cpp_generator_current_seed(self$ptr))
    },
    set_current_seed = function(seed) {
      if ((!is.integer(seed)) && (!inherits(seed, "integer64"))) {
        stop("Seed must an integer or integer64.")
      }

      seed <- as.character(seed)

      cpp_generator_set_current_seed(self$ptr, seed)
    }
  ),
  active = list(
    ptr = function() {
      self
    }
  )
)

#' Create a Generator object
#'
#' A `torch_generator`  is an object which manages the state of the algorithm
#' that produces pseudo random numbers. Used as a keyword argument in many
#' In-place random sampling functions.
#'
#' @examples
#'
#' # Via string
#' generator <- torch_generator()
#' generator$current_seed()
#' generator$set_current_seed(1234567L)
#' generator$current_seed()
#'
#' @export
torch_generator <- function() {
  Generator$new()
}

is_torch_generator <- function(x) {
  inherits(x, "torch_generator")
}

#' Sets the seed for generating random numbers.
#'
#' @param seed integer seed.
#' @param code expression to run in the context of the seed
#' @param .env environment that will take the modifications from manual_seed.
#' @param ... unused currently.
#' 
#' @note Currently the `local_torch_manual_seed` and `with_torch_manual_seed` won't
#'   work with Tensors in the MPS device. You can sample the tensors on CPU and 
#'   move them to MPS if reproducibility is required.
#'
#' @export
torch_manual_seed <- function(seed) {
  cpp_torch_manual_seed(as.character(seed))
  # update the null generator
  if (torch_option("old_seed_behavior", FALSE)) {
    .generator_null$set_current_seed(
      seed = as.integer(torch::torch_randint(low = 1, high = 1e6, size = 1)$item())
    )
  }
}

#' @describeIn torch_manual_seed Modifies the torch seed in the environment scope.
local_torch_manual_seed <- function(seed, .env = parent.frame()) {
  current_state <- list()
  current_state[["cpu"]] <- torch_get_rng_state()
  if (cuda_is_available())
    current_state[["cuda"]] <- cuda_get_rng_state()
  
  torch_manual_seed(seed)
  withr::defer({
    torch_set_rng_state(current_state$cpu)
    if (!is.null(current_state$cuda)) cuda_set_rng_state(current_state$cuda)
  }, envir = .env)
}

#' @describeIn torch_manual_seed A with context to change the seed during the function execution.
with_torch_manual_seed <- function(code, ..., seed) {
  ellipsis::check_dots_empty()
  local_torch_manual_seed(seed)
  force(code)
}

#' RNG state management
#' 
#' Low level functionality to set and change the RNG state.
#' It's recommended to use [torch_manual_seed()] for most cases.
#' 
#' @param state A tensor with the current state or a list containing the state 
#'   for each device - (for CUDA).
#' @param device The cuda device index to get or set the state. If `NULL` gets the state
#'   for all available devices.
#'
#' @export
torch_get_rng_state <- function() {
  cpp_torch_get_rng_state()
}

#' @describeIn torch_get_rng_state Sets the RNG state for the CPU
torch_set_rng_state <- function(state) {
  cpp_torch_set_rng_state(state)
}

#' @describeIn torch_get_rng_state Gets the RNG state for CUDA. 
cuda_get_rng_state <- function(device = NULL) {
  if (!is.null(device)) {
    return(cpp_torch_cuda_get_rng_state(device))
  }
  
  devices <- cuda_device_count()
  states <- list()
  for (i in seq_len(devices)) {
    states[[i]] <- cpp_torch_cuda_get_rng_state(i - 1)
  }
  states
}

#' @describeIn torch_get_rng_state Sets the RNG state for CUDA. 
cuda_set_rng_state <- function(state, device = NULL) {
  if (!is.null(device)) {
    return(cpp_torch_cuda_set_rng_state(device, state))
  }
  
  if (length(state) != cuda_device_count()) {
    cli::cli_abort("Expected length {.var state} ({.val {length(state)}}) equal to the number of cuda devices ({.val {cuda_device_count()}}).")
  }
  
  for (i in seq_along(state)) {
    cpp_torch_cuda_set_rng_state(i-1, state[[i]])
  }
}
