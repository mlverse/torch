nn_Module <- R6::R6Class(
  classname = "nn_module",
  lock_objects = FALSE,
  active = list(
    parameters = function() {
      nms <- names(self)
      nms <- nms[nms != "parameters"]
      pars <- lapply(
        nms,
        function(nm) {
          x <- self[[nm]]
          
          if (is_nn_parameter(x))
            return(x)
          
          if (is_nn_module(x))
            return(x$parameters)
          
        }
      )
      unlist(pars, recursive = TRUE, use.names = TRUE)
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

is_nn_module <- function(x) {
  inherits(x, "nn_module")
}

nn_module <- function(classname = NULL, initialize, forward, ...) {
  R6::R6Class(
    classname = classname,
    inherit = nn_Module,
    lock_objects = FALSE,
    public = list(
      initialize = initialize,
      forward = forward,
      ...
    )
  )
}




