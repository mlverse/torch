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

torch_hook <- R6::R6Class(
  classname = "torch_hook",
  public = list(
    x = NULL,
    pos = NULL,
    initialize = function(x, pos) {
      self$x <- x
      self$pos <- pos
    },
    remove = function() {
      cpp_tensor_remove_hook(self$x$ptr, self$pos)
    },
    print = function() {
      cat("<torch_hook>")
    }
  )
)

Tensor$set("public", "register_hook", function(hook) {
  wrap <- function(grad) {
    out <- hook(Tensor$new(ptr = grad))
    if (!is_torch_tensor(out))
      cpp_tensor_undefined()
    else
      out$ptr
  }
  pos <- cpp_tensor_register_hook(self$ptr, wrap)
  torch_hook$new(self, pos)
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

AutogradContext <- R6::R6Class(
  classname = "torch_autograd_context",
  public = list(
    
    ptr = NULL,
    
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    
    save_for_backward = function(vars) {
      cpp_autograd_context_save_for_backward(self$ptr, torch_variable_list(vars)$ptr)
    },
    
    get_saved_variables = function() {
      vl <- variable_list$new(ptr = cpp_autograd_context_get_saved_variables(self$ptr))
      vl$to_r()
    },
    
    set_arguments = function(names, needs_grad) {
      cpp_autograd_context_set_arguments(self$ptr, names, needs_grad)
    },
    
    get_argument_names = function() {
      cpp_autograd_context_get_argument_names(self$ptr)
    },
    
    get_argument_needs_grad = function() {
      cpp_autograd_context_get_argument_needs_grad(self$ptr)
    }

  )
)

autograd_function <- function(forward, backward) {
  
  
  other <- NULL
  variables <- NULL
  argument_names <- NULL
  argument_needs_grad <- NULL
  
  f <- function(ctx, inputs) {
    inputs <- variable_list$new(ptr = inputs)$to_r()
    names(inputs) <- names(variables)
    args <- append(inputs, other)
    
    args$ctx <- AutogradContext$new(ctx)
    args$ctx$set_arguments(argument_names, argument_needs_grad)
    
    res <- do.call(forward, args)
    
    if (!is.list(res))
      res <- list(res)
    
    torch_variable_list(res)$ptr
  }
  
  b <- function(ctx, grad_output) {
    ctx <- AutogradContext$new(ctx)
    grad_output <- variable_list$new(ptr = grad_output)$to_r()
    res <- backward(ctx, grad_output)
    
    if (!is.list(res))
      res <- list(res)
    
    torch_variable_list(res)$ptr
  }
  
  f_ <- cpp_Function_lambda(f)
  b_ <- cpp_Function_lambda(b)
  
  
  function(...) {
    
    args <- list(...)
    
    variables <<- Filter(
      function(arg) {is_torch_tensor(arg) && arg$requires_grad()}, 
      args
    )
    
    other <<- Filter(
      function(arg) {!(is_torch_tensor(arg) && arg$requires_grad())}, 
      args
    )
    
    argument_names <<- names(args)
    argument_needs_grad <<- names(args) %in% names(variables)
    
    res <- cpp_Function_apply(torch_variable_list(variables)$ptr, f_, b_)
    res <- variable_list$new(ptr = res)$to_r()
    res  
  }
  
}