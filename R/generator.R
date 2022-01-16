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
