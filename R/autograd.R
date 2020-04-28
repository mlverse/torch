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
    
    initialize = function(ptr, env) {
      self$ptr <- ptr
      private$.env <- env
    },
    
    save_for_backward = function(...) {
      
      args <- rlang::list2(...)
  
      private$.env$.is_torch_tensor <- as.logical(sapply(args, is_torch_tensor))
      
      vars <- args[private$.env$.is_torch_tensor]
      other <- args[!private$.env$.is_torch_tensor]
      
      cpp_autograd_context_save_for_backward(self$ptr, torch_variable_list(vars)$ptr)
      private$.env$.other <- other
      
      if (is.null(names(vars)))
        nms <- rep("", length(vars))
      else
        nms <- names(vars)
      
      cpp_autograd_context_set_saved_variables_names(self$ptr, nms)
    },
    
    get_saved_variables = function() {
      
      # retrieve variables
      vars <- variable_list$new(ptr = cpp_autograd_context_get_saved_variables(self$ptr))
      vars <- vars$to_r()
      
      nms <- cpp_autograd_context_get_saved_variables_names(self$ptr)
      if (!all(nms == ""))
        names(vars) <- nms
      
      # retrieve other
      other <- private$.env$.other
      
      # retrieve order
      is_tensors <- private$.env$.is_torch_tensor
      indexes <- integer(length = length(is_tensors))
      indexes[is_tensors] <- cumsum(is_tensors[is_tensors])
      indexes[!is_tensors] <- cumsum(!is_tensors[!is_tensors])
      
      mapply(
        FUN = function(is_tensor, index) {
          if (is_tensor)
            vars[index]
          else
            other[index]
        }, 
        is_tensors,
        indexes,
        USE.NAMES = FALSE
      )
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
  ),
  active = list(
    needs_input_grad = function() {
      setNames(as.list(self$get_argument_needs_grad()), self$get_argument_names())
    },
    saved_variables = function() {
      self$get_saved_variables()
    }
  ),
  private  = list(
    .env = NULL
  )
)

autograd_function <- function(forward, backward) {
  rlang::new_function(
    args = rlang::fn_fmls(forward)[-1],
    body = rlang::expr({
      
      # environment to transfer info from this function to
      # the forward/backward function
      .env <- rlang::new_environment()
      .env$forward_returns_list <- TRUE
      
      # create the c++ lambda wrapping the R function
      .f <- function(ctx, inputs) {
        inputs <- variable_list$new(ptr = inputs)$to_r()
        names(inputs) <- names(.env$variables)
        args <- append(inputs, .env$other)
        
        args$ctx <- AutogradContext$new(ctx, .env)
        args$ctx$set_arguments(.env$argument_names, .env$argument_needs_grad)
        
        res <- do.call(forward, args)
        
        if (!is.list(res)) {
          .env$forward_returns_list <- FALSE
          res <- list(res)
        }
        
        torch_variable_list(res)$ptr
      }
      .b <- function(ctx, grad_output) {
        
        # parse pointers to R objects
        ctx <- AutogradContext$new(ctx, .env)
        grad_output <- variable_list$new(ptr = grad_output)$to_r()
        
        # destructure the grad_output list
        fmls <- rlang::fn_fmls_names(backward)[-1] # remove the context
        if (length(grad_output) > length(fmls)) {
          if (length(fmls) == 1) # and length(grad_output) > 1
            grad_output <- list(grad_output)
          else {
            d <- length(grad_output) - length(fmls)
            grad_output <- append(
              grad_output[1:(length(grad_output) - (d + 1))],
              list(grad_output[(length(grad_output) - d):length(grad_output)])
            )  
          }
        }
        args <- append(list(ctx), grad_output)
        res <- do.call(backward, args)
        
        argument_names <- ctx$get_argument_names()
        argument_needs_grad <- ctx$get_argument_needs_grad()
        
        res <- res[argument_names[argument_needs_grad]]
        
        torch_variable_list(res)$ptr
      }
      
      .f_ <- cpp_Function_lambda(.f)
      .b_ <- cpp_Function_lambda(.b)
      
      # passing the variables through cpp_Function_apply
      # other arguments are passed through `.env`
      args <- rlang::list2(!!!rlang::fn_fmls_syms(forward)[-1])
      is_var <- sapply(args, function(arg) {is_torch_tensor(arg) && arg$requires_grad()})    
    
      .env$variables <- args[is_var]
      .env$other <- args[!is_var]
      
      .env$argument_names <- names(args)
      .env$argument_needs_grad <- names(args) %in% names(.env$variables)
      
      res <- cpp_Function_apply(torch_variable_list(.env$variables)$ptr, .f_, .b_)
      res <- variable_list$new(ptr = res)$to_r()
      
      # post processing of results
      if (!.env$forward_returns_list)
       res <- res[[1]]
      
      res
    })
  )
}