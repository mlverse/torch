#' @include tensor.R
NULL

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
#' x$grad
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
#' x$grad
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

Tensor$set("active", "grad", function() {
  Tensor$new(ptr = cpp_tensor_grad(self$ptr))
})

Tensor$set("active", "requires_grad", function() {
  cpp_tensor_requires_grad(self$ptr)
})

Tensor$set("public", "backward", function(gradient = list(), keep_graph = FALSE, 
                                          create_graph = FALSE) {
  invisible(private$`_backward`(gradient, keep_graph, create_graph))
})

Tensor$set("public", "retain_grad", function() {
  invisible(private$`_retain_grad`())
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

#' Class representing the context.
#'  
AutogradContext <- R6::R6Class(
  classname = "torch_autograd_context",
  public = list(
    
    #' @field ptr (Dev related) pointer to the context c++ object.
    ptr = NULL,
    
    #' @description 
    #' (Dev related) Initializes the context. Not user related.
    #' 
    #' @param ptr pointer to the c++ object
    #' @param env environment that encloses both forward and backward
    #' @param argument_names names of forward arguments
    #' @param argument_needs_grad whether each argument in forward needs grad.
    initialize = function(ptr, env, argument_names = NULL, argument_needs_grad = NULL) {
      self$ptr <- ptr
      private$.env <- env
      if (!is.null(argument_names) && !is.null(argument_needs_grad))
        private$set_arguments(argument_names, argument_needs_grad)
    },
    
    #' @description
    #' Saves given objects for a future call to backward().
    #'
    #' This should be called at most once, and only from inside the `forward()` 
    #' method.
    #' 
    #' Later, saved objects can be accessed through the `saved_variables` attribute. 
    #' Before returning them to the user, a check is made to ensure they weren’t used 
    #' in any in-place operation that modified their content.
    #' 
    #' Arguments can also be any kind of R object.
    #'
    #' @param ... any kind of R object that will be saved for the backward pass.
    #'   It's common to pass named arguments.
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
    
    #' @description
    #' Marks outputs as non-differentiable.
    #' 
    #' This should be called at most once, only from inside the `forward()` method, 
    #' and all arguments should be outputs.
    #' 
    #' This will mark outputs as not requiring gradients, increasing the efficiency 
    #' of backward computation. You still need to accept a gradient for each output 
    #' in `backward()`, but it’s always going to be a zero tensor with the same 
    #' shape as the shape of a corresponding output.
    #' 
    #' This is used e.g. for indices returned from a max Function.
    #' 
    #' @param ... non-differentiable outputs.
    mark_non_differentiable = function(...) {
      vars <- rlang::list2(...)
      var_list <- torch_variable_list(vars)
      cpp_autograd_context_mark_non_differentiable(self$ptr, var_list$ptr)
      invisible(NULL)
    },
    
    #' @description 
    #' Marks given tensors as modified in an in-place operation.
    #'
    #' This should be called at most once, only from inside the `forward()` method, 
    #' and all arguments should be inputs.
    #' 
    #' Every tensor that’s been modified in-place in a call to `forward()` should 
    #' be given to this function, to ensure correctness of our checks. It doesn’t 
    #' matter whether the function is called before or after modification.
    #' 
    #' @param ... tensors that are modified in-place.
    mark_dirty = function(...) {
      vars <- rlang::list2(...)
      var_list <- torch_variable_list(vars)
      cpp_autograd_context_mark_dirty(self$ptr, var_list$ptr)
      invisible(NULL)
    }
  ),
  active = list(
    
    #' @field needs_input_grad boolean listing arguments of `forward` and whether they require_grad.
    needs_input_grad = function() {
      setNames(as.list(private$get_argument_needs_grad()), private$get_argument_names())
    },
    
    #' @field saved_variables list of objects that were saved for backward via `save_for_backward`.
    saved_variables = function() {
      private$get_saved_variables()
    }
  ),
  private  = list(
    .env = NULL,
    set_arguments = function(names, needs_grad) {
      cpp_autograd_context_set_arguments(self$ptr, names, needs_grad)
    },
    get_argument_names = function() {
      cpp_autograd_context_get_argument_names(self$ptr)
    },
    get_argument_needs_grad = function() {
      cpp_autograd_context_get_argument_needs_grad(self$ptr)
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
    }
  )
)

#' Records operation history and defines formulas for differentiating ops.
#' 
#' Every operation performed on Tensor's creates a new function object, that 
#' performs the computation, and records that it happened. The history is 
#' retained in the form of a DAG of functions, with edges denoting data 
#' dependencies (input <- output). Then, when backward is called, the graph is 
#' processed in the topological ordering, by calling `backward()` methods of each 
#' Function object, and passing returned gradients on to next Function's.
#' 
#' @param forward Performs the operation. It must accept a context `ctx` as the first argument, 
#'   followed by any number of arguments (tensors or other types). The context can be 
#'   used to store tensors that can be then retrieved during the backward pass.
#'   See [AutogradContext] for more information about context methods.
#' @param backward Defines a formula for differentiating the operation. It must accept 
#'   a context `ctx` as the first argument, followed by as many outputs did `forward()` 
#'   return, and it should return a named list. Each argument is the gradient w.r.t 
#'   the given output, and each element in the returned list should be the gradient 
#'   w.r.t. the corresponding input. The context can be used to retrieve tensors saved 
#'   during the forward pass. It also has an attribute `ctx$needs_input_grad` as a 
#'   named list of booleans representing whether each input needs gradient. 
#'   E.g., `backward()` will have `ctx$needs_input_grad$input = TRUE` if the `input`
#'   argument to `forward()` needs gradient computated w.r.t. the output.
#'   See [AutogradContext] for more information about context methods.
#'   
#' @examples 
#' 
#' exp2 <- autograd_function(
#'   forward = function(ctx, i) {
#'     result <- i$exp()
#'     ctx$save_for_backward(result = result)
#'     result
#'   },
#'   backward = function(ctx, grad_output) {
#'     list(i = grad_output * ctx$saved_variable$result)
#'   }
#' )
#' 
#' @export
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
        
        args$ctx <- AutogradContext$new(ctx, .env, .env$argument_names, 
                                        .env$argument_needs_grad)
        
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
        
        needs_grad <- ctx$needs_input_grad
        argument_names <- names(needs_grad)
        argument_needs_grad <- as.logical(needs_grad)
        
        res <- res[argument_names[argument_needs_grad]]
        
        torch_variable_list(res)$ptr
      }
      
      .f_ <- cpp_Function_lambda(.f)
      .b_ <- cpp_Function_lambda(.b)
      
      # passing the variables through cpp_Function_apply
      # other arguments are passed through `.env`
      args <- rlang::list2(!!!rlang::fn_fmls_syms(forward)[-1])
      is_var <- sapply(args, function(arg) {is_torch_tensor(arg) && arg$requires_grad})    
    
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

Edge <- R6::R6Class(
  classname = "autograd_edge",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    func = function() {
      Node$new(cpp_autograd_edge_function(self$ptr))
    }
  )
)

Node <- R6::R6Class(
  classname = "autograd_node",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    print = function() {
      cat(cpp_autograd_node_name(self$ptr))
    }
  ),
  active = list(
    next_functions = function() {
      l <- lapply(cpp_autograd_node_next_edges(self$ptr), Edge$new)
      lapply(l, function(x) x$func())
    }
  )
)

Tensor$set("active", "grad_fn", function() {
  Node$new(cpp_tensor_grad_fn(self$ptr))
})

#' Computes the sum of gradients of given tensors w.r.t. graph leaves.
#' 
#' The graph is differentiated using the chain rule. If any of tensors are 
#' non-scalar (i.e. their data has more than one element) and require gradient, 
#' then the Jacobian-vector product would be computed, in this case the function 
#' additionally requires specifying `grad_tensors`. It should be a sequence of 
#' matching length, that contains the “vector” in the Jacobian-vector product, 
#' usually the gradient of the differentiated function w.r.t. corresponding 
#' tensors (None is an acceptable value for all tensors that don’t need gradient 
#' tensors).
#' 
#' This function accumulates gradients in the leaves - you might need to zero 
#' them before calling it.
#' 
#' @param tensors (list of Tensor) – Tensors of which the derivative will 
#' be computed.
#' @param grad_tensors (list of (Tensor or `NULL)) – The “vector” in the 
#' Jacobian-vector product, usually gradients w.r.t. each element of 
#' corresponding tensors. `NULL` values can be specified for scalar Tensors or 
#' ones that don’t require grad. If a `NULL` value would be acceptable for all 
#' grad_tensors, then this argument is optional.
#' @param retain_graph (bool, optional) – If `FALSE`, the graph used to compute 
#' the grad will be freed. Note that in nearly all cases setting this option to 
#' `TRUE` is not needed and often can be worked around in a much more efficient 
#' way. Defaults to the value of `create_graph`.
#' @param create_graph (bool, optional) – If `TRUE`, graph of the derivative will 
#' be constructed, allowing to compute higher order derivative products. 
#' Defaults to `FALSE`.
#' 
#' @examples
#' x <- torch_tensor(1, requires_grad = TRUE)
#' y <- 2 * x
#' 
#' a <- torch_tensor(1, requires_grad = TRUE)
#' b <- 3 * a
#' 
#' autograd_backward(list(y, b))
#' 
#' @export
autograd_backward <- function(tensors, grad_tensors = NULL, retain_graph = create_graph, 
                              create_graph = FALSE) {
  
  if (!is.list(tensors)) 
    tensors <- list(tensors)
  
  tensors_ <- torch_variable_list(tensors)
  
  if (is.null(grad_tensors))
    grad_tensors <- lapply(seq_along(tensors), function(x) NULL)
  else if (!is.list(grad_tensors))
    grad_tensors <- list(grad_tensors)
  
  null <- sapply(grad_tensors, is.null)
  if (length(null) > 0) {
    grad_tensors[null] <- lapply(
      seq_along(null), 
      function(x) Tensor$new(ptr = cpp_tensor_undefined())
    )  
  }
    
  grad_tensors_ <- torch_variable_list(grad_tensors)
  
  cpp_autograd_backward(
    tensors_$ptr,
    grad_tensors_$ptr,
    retain_graph,
    create_graph
  )
  
  invisible(NULL)
}

#' Computes and returns the sum of gradients of outputs w.r.t. the inputs.
#' 
#' `grad_outputs` should be a list of length matching output containing the “vector” 
#' in Jacobian-vector product, usually the pre-computed gradients w.r.t. each of 
#' the outputs. If an output doesn’t require_grad, then the gradient can be None).
#' 
#' If only_inputs is `TRUE`, the function will only return a list of gradients w.r.t 
#' the specified inputs. If it’s `FALSE`, then gradient w.r.t. all remaining leaves 
#' will still be computed, and will be accumulated into their `.grad` attribute.
#' @param outputs (sequence of Tensor) – outputs of the differentiated function.
#' @param inputs (sequence of Tensor) – Inputs w.r.t. which the gradient will be 
#' returned (and not accumulated into .grad).
#' @param grad_outputs (sequence of Tensor) – The “vector” in the Jacobian-vector 
#' product. Usually gradients w.r.t. each output. None values can be specified for 
#' scalar Tensors or ones that don’t require grad. If a None value would be acceptable 
#' for all `grad_tensors`, then this argument is optional. Default: None.
#' @param retain_graph (bool, optional) – If `FALSE`, the graph used to compute the 
#' grad will be freed. Note that in nearly all cases setting this option to `TRUE` is 
#' not needed and often can be worked around in a much more efficient way. 
#' Defaults to the value of `create_graph`.
#' @param create_graph (bool, optional) – If `TRUE, graph of the derivative will 
#' be constructed, allowing to compute higher order derivative products. 
#' Default: `FALSE`.
#' @param allow_unused (bool, optional) – If `FALSE`, specifying inputs that were 
#' not used when computing outputs (and therefore their grad is always zero) is an 
#' error. Defaults to `FALSE`
#' 
#' @examples 
#' w <- torch_tensor(0.5, requires_grad = TRUE)
#' b <- torch_tensor(0.9, requires_grad = TRUE)
#' x <- torch_tensor(runif(100))
#' y <- 2 * x + 1
#' loss <- (y - (w*x + b))^2
#' loss <- loss$mean()
#' 
#' o <- autograd_grad(loss, list(w, b))
#' o
#'  
#' @export
autograd_grad <- function(outputs, inputs, grad_outputs = NULL, retain_graph = create_graph,
                          create_graph = FALSE, allow_unused = FALSE) {
  
  if (!is.list(outputs))
    outputs <- list(outputs)
  
  outputs_ <- torch_variable_list(outputs)
  
  if (!is.list(inputs))
    inputs <- list(inputs)
  
  inputs_ <- torch_variable_list(inputs)
  
  if (is.null(grad_outputs)) {
    grad_outputs <- lapply(
      seq_along(outputs), 
      function(x) Tensor$new(ptr = cpp_tensor_undefined())
    )
  } else if (!is.list(grad_outputs)) {
    grad_outputs <- list(grad_outputs)
  }
    
  grad_outputs_ <- torch_variable_list(grad_outputs)
  
  out <- cpp_autograd_grad(
    outputs_$ptr,
    inputs_$ptr,
    grad_outputs_$ptr,
    retain_graph, 
    create_graph, 
    allow_unused
  )
  variable_list$new(ptr = out)$to_r()
}
