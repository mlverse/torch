#' @include tensor.R

#' With no grad
#'
#' Temporarily modify gradient recording.
#'
#' @param code code to be executed with no gradient recording.
#'
#' @examples
#' x <- torch_tensor(runif(5), requires_grad = TRUE)
#' with_no_grad({
#'   x$sub_(torch_tensor(as.numeric(1:5)))
#' })
#' x
#' x$grad()
#'
#' @export
with_no_grad <- function(code) {
  withr::with_(
    set = function() {
      cpp_autograd_set_grad_mode(FALSE)
    },
    reset = function(old) {
      cpp_autograd_set_grad_mode(TRUE)
    }
  )(code)
}


#' Enable grad
#' 
#' Context-manager that enables gradient calculation.
#' Enables gradient calculation, if it has been disabled via [with_no_grad] or [set_grad_enabled].
#' 
#' This context manager is thread local; it will not affect computation in 
#' other threads.
#' 
#' @param code code to be executed with gradient recording.
#' 
#' @examples 
#' 
#' x <- torch_tensor(1, requires_grad=TRUE)
#' with_no_grad({
#'   with_enable_grad({
#'     y = x * 2
#'   })
#' })
#' y$backward()
#' x$grad()
#' 
#' @export
with_enable_grad <- function(code) {
  withr::with_(
    set = function() {
      cpp_autograd_set_grad_mode(TRUE)
    },
    reset = function(old) {
      cpp_autograd_set_grad_mode(FALSE)
    }
  )(code)
}

Tensor$set("public", "grad", function() {
  Tensor$new(ptr = cpp_tensor_grad(self$ptr))
})

Tensor$set("public", "requires_grad", function() {
  cpp_tensor_requires_grad(self$ptr)
})

Tensor$set("public", "backward", function(gradient = list(), keep_graph = FALSE, 
                                          create_graph = FALSE) {
  invisible(private$`_backward`(gradient, keep_graph, create_graph))
})

Tensor$set("public", "register_hook", function(hook) {
  aux <- function(grad) {
    out <- hook(Tensor$new(ptr = grad))
    if (!is_torch_tensor(out))
      cpp_tensor_undefined()
    else
      out$ptr
  }
  cpp_tensor_register_hook(self$ptr, aux)
})


#' Set grad mode
#' 
#' Sets or disables gradient history.
#'
#' @param enabled bool wether to enable or disable the gradient recording.
#' 
#' @export
autograd_set_grad_mode <- function(enabled) {
  cpp_autograd_set_grad_mode(enabled)
}