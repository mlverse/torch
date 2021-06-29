
#' Adds the 'jit_tuple' class to the input
#' 
#' Allows specifying that an output or inputmust be considered a jit
#' tuple and instead of a list or dictionary when tracing.
#' 
#' @param x the list opbject that will be converted to a tuple.
#' 
#' @export
jit_tuple <- function(x) {
  if (!is.list(x))
    runtime_error("Argument 'x' must be a list.")
  
  class(x) <- c(class(x), "jit_tuple")
  x
}