nn_Module <- R6::R6Class(
  classname = "nn_Module",
  lock_objects = FALSE,
  active = list(
    parameters = function() {
      nms <- names(self)
      nms <- nms[nms != "parameters"]
      pars <- Filter(function(nm) is_nn_parameter(self[[nm]]), nms)
      lapply(pars, function(x) self[[x]])
    }
  )
)

nn_parameter <- function(x, requires_grad = TRUE) {
  if (!is_torch_tensor(x))
    stop("`x` must be a tensor.")
  x$requires_grad_(requires_grad)
  class(x) <- c(class(x), "nn_parameter")
  x
}

is_nn_parameter <- function(x) {
  inherits(x, "nn_parameter")
}

nn_module <- function(initialize, forward) {
  R6::R6Class(
    inherit = nn_Module,
    lock_objects = FALSE,
    public = list(
      initialize = initialize,
      forward = forward
    )
  )
}


