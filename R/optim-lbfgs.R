#' @include optim.R
NULL

optim_LBFGS <- R6::R6Class(
  "optim_lbfgs", 
  lock_objects = FALSE,
  inherit = Optimizer,
  public = list(
    initialize = function(params,
                          lr=1,
                          max_iter=20,
                          max_eval=NULL,
                          tolerance_grad=1e-7,
                          tolerance_change=1e-9,
                          history_size=100,
                          line_search_fn=NULL) {
      
      if (is.null(max_eval))
        max_eval <- as.integer(max_iter * 5 / 4)
      
      defaults <- list(
        lr = lr,
        max_iter= max_iter,
        max_eval = max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn=line_search_fn
      )
      
      super$initialize(params, defaults)
      
      if (length(self$param_groups) != 1)
        value_error("LBFGS doesn't support per-parameter options ",
                    "(parameter groups)")
      
      private$.params <- self$param_groups[[1]][['params']]
      private$.numel_cache <- NULL
      
    },
    step = function(closure) {
      with_no_grad({
        closure_ <- function() {
          with_enable_grad({
            closure()
          })
        }
        
        group <- self$param_groups[[1]]
        lr <- group[['lr']]
        max_iter <- group[['max_iter']]
        max_eval <- group[['max_eval']]
        tolerance_grad <- group[['tolerance_grad']]
        tolerance_change <- group[['tolerance_change']]
        line_search_fn <- group[['line_search_fn']]
        history_size <- group[['history_size']]
        
        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        if (is.null(private$.params[[1]]$state)) {
          private$.params[[1]]$state <- new.env(parent = emptyenv())
          private$.params[[1]]$state[["func_evals"]] <- 0
          private$.params[[1]]$state[["n_iter"]] <- 0
        }
        state <- private$.params[[1]]$state
          
        # evaluate initial f(x) and df/dx
        orig_loss <- closure_()
        loss <- orig_loss$item()
        current_evals <- 1
        state[['func_evals']] <- state[['func_evals']] + 1
        
        flat_grad <- private$.gather_flat_grad()
        opt_cond <- (flat_grad$abs()$max() <= tolerance_grad)$item()
        
        if (opt_cond)
          return(orig_loss)
        
        # tensors cached in state (for tracing)
        d <- state[['d']]
        t <- state[['t']]
        old_dirs <- state[['old_dirs']]
        old_stps <- state[['old_stps']]
        ro <- state[['ro']]
        H_diag <- state[['H_diag']]
        prev_flat_grad <- state[['prev_flat_grad']]
        prev_loss <- state[['prev_loss']]
        
        n_iter <- 0
        # optimize for a max of max_iter iterations
        while (n_iter < max_iter) {
          # keep track of nb of iterations
          n_iter <- n_iter + 1
          state[['n_iter']] <- state[['n_iter']] + 1
          
          ############################################################
          # compute gradient descent direction
          ############################################################
          
          if (state[['n_iter']] == 1) {
            d <- flat_grad$neg()
            old_dirs <- list()
            old_stps <- list()
            ro <- list()
            H_diag <- 1
          } else {
            # do lbfgs update (update memory)
            y <- flat_grad$sub(prev_flat_grad)
            s <- d$mul(t)
            ys <- y$dot(s)  # y*s
            
            if (!is.na(ys$item()) && ys$item() > 1e-10) {
              # updating memory
              if (length(old_dirs) == history_size) {
                # shift history by one (limited-memory)
                old_dirs <- old_dirs[-1]
                old_stps <- old_stps[-1]
                ro <- ro[-1]
              }
              
              # store new direction/step
              old_dirs[[length(old_dirs) + 1]] <- y
              old_stps[[length(old_stps) + 1]] <- s
              ro[[length(ro) + 1]] <- 1. / ys
              
              # update scale of initial Hessian approximation
              H_diag <- ys / y$dot(y)  # (y*y)
                
            }
            
            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            num_old <- length(old_dirs)
            
            if (is.null(state[["al"]]))
              state[["al"]] <- vector(mode = "list", length = history_size)
            
            al <- state[['al']]
            
            # iteration in L-BFGS loop collapsed to use just one buffer
            q <- flat_grad$neg()
            if (num_old >= 1) {
              for (i in seq(num_old, 1, by = -1)) {
                al[[i]] <- old_stps[[i]]$dot(q) * ro[[i]]
                q$add_(old_dirs[[i]], alpha=-al[[i]])
              }  
            }
          
            # multiply by initial Hessian
            # r/d is the final direction
            d <- r <- torch_mul(q, H_diag)
            for (i in seq_len(num_old)) {
              be_i <- old_dirs[[i]]$dot(r) * ro[[i]]
              r$add_(old_stps[[i]], alpha = al[[i]] - be_i)
            }
            
          }
          
          if (is.null(prev_flat_grad) || is_undefined_tensor(prev_flat_grad))
            prev_flat_grad <- flat_grad$clone(memory_format=torch_contiguous_format())
          else
            prev_flat_grad$copy_(flat_grad)
          prev_loss <- loss
          
          ############################################################
          # compute step length
          ############################################################
          # reset initial guess for step size
          if (state[['n_iter']] == 1) {
            t <- min(1., 1. / flat_grad$abs()$sum()$item()) * lr
          } else {
            t <- lr
          }
          
          # directional derivative
          gtd <- flat_grad$dot(d)  # g * d
          
          # directional derivative is below tolerance
          if (!is.na(gtd$item()) && gtd$item() > (-tolerance_change))
            break
          
          # optional line search: user function
          ls_func_evals = 0
          
          if (!is.null(line_search_fn)) {
            # perform line search, using user function
            value_error("Not yet supported in R")
          } else {
            # no line search, simply move with fixed-step
            private$.add_grad(t, d)
            if (n_iter != max_iter) {
              # re-evaluate function only if not in last iteration
              # the reason we do this: in a stochastic setting,
              # no use to re-evaluate that function here
              loss = closure_()$item()
              flat_grad <- private$.gather_flat_grad()
              opt_cond <- flat_grad$abs()$max()$item() <= tolerance_grad
              ls_func_evals <- 1
            }
          }
          
          # update func eval
          current_evals <- current_evals + ls_func_evals
          state[['func_evals']] <- state[['func_evals']] + ls_func_evals
          
          ############################################################
          # check conditions
          ############################################################
          if (n_iter == max_iter)
            break
          
          if (current_evals >= max_eval)
            break
          
          # optimal condition
          if (!is.na(opt_cond) && opt_cond)
            break
          
          # lack of progress
          d_ml <- d$mul(t)$abs()$max()$item()
          if (!is.na(d_ml) && d_ml <= tolerance_change)
            break
          
          if (!is.na(loss) && abs(loss - prev_loss) < tolerance_change)
            break
      
        }
        
        state[['d']] <- d
        state[['t']] <- t
        state[['old_dirs']] <- old_dirs
        state[['old_stps']] <- old_stps
        state[['ro']] <- ro
        state[['H_diag']] <- H_diag
        state[['prev_flat_grad']] <- prev_flat_grad
        state[['prev_loss']] <- prev_loss
        
      })
      
      orig_loss
    }
  ),
  private = list(
    .numel = function() {
      if (is.null(private$.numel_cache)) {
        private$.numel_cache <- sum(sapply(private$.params, function(p) p$numel()))
      }
      private$._numel_cache
    },
    .gather_flat_grad = function() {
      views <- list()
      for (i in seq_along(private$.params)) {
        p <- private$.params[[i]]
        if (is_undefined_tensor(p$grad)) {
          view <- .new_zeros_tensor(p)
        } else {
          view <- p$grad$view(-1)
        }
        views[[i]] <- view
      }
      torch_cat(views, dim = 1)
    },
    .add_grad = function(step_size, update) {
      offset <- 1
      for (p in private$.params) {
        numel <- p$numel()
        p$add_(update[offset:(offset + numel - 1)]$view_as(p), alpha = step_size)
        offset <- offset + numel
      }
      stopifnot(offset == private$.numel())
    },
    .clone_param = function() {
      lapply(private$.params, function(p) p$clone(memory_format = torch_contiguous_format()))
    },
    .set_param = function(params_data) {
      for(i in seq_along(private$.params)) {
        private$.params[[i]]$copy_(params_data[[i]])
      }
    },
    .directional_evaluate = function(closure, x, t, d) {
      private$.add_grad(t, d)
      loss <- closure()$item()
      flat_grad <- private$.gather_flat_grad()
      private$.set_param(x)
      list(loss, flat_grad)
    }
  )
)


.new_zeros_tensor <- function(p) {
  torch_zeros(p$numel(), dtype = p$dtype, device = p$device)
}


#' LBFGS optimizer
#' 
#' 
#' Implements L-BFGS algorithm, heavily inspired by 
#' [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
#' 
#' @section Warning:
#' 
#' This optimizer doesn't support per-parameter options and parameter
#' groups (there can be only one).
#' 
#' Right now all parameters have to be on a single device. This will be
#' improved in the future.
#' 
#' @note
#' This is a very memory intensive optimizer (it requires additional
#' `param_bytes * (history_size + 1)` bytes). If it doesn't fit in memory
#' try reducing the history size, or use a different algorithm.
#' 
#' @param lr (float): learning rate (default: 1)
#' @param max_iter (int): maximal number of iterations per optimization step
#'   (default: 20)
#' @param max_eval (int): maximal number of function evaluations per optimization
#'   step (default: max_iter * 1.25).
#' @param tolerance_grad (float): termination tolerance on first order optimality
#'   (default: 1e-5).
#' @param tolerance_change (float): termination tolerance on function
#'   value/parameter changes (default: 1e-9).
#' @param history_size (int): update history size (default: 100).
#' @param line_search_fn (str): either 'strong_wolfe' or None (default: None).
#' @inheritParams optim_sgd
#' 
#' 
#' @export  
optim_lbfgs <- function(params,
                        lr=1,
                        max_iter=20,
                        max_eval=NULL,
                        tolerance_grad=1e-7,
                        tolerance_change=1e-9,
                        history_size=100,
                        line_search_fn=NULL) {
  optim_LBFGS$new(params, lr = lr, max_iter = max_iter, max_eval = max_eval, 
                  tolerance_grad = tolerance_grad, tolerance_change = tolerance_change,
                  history_size = history_size, line_search_fn = NULL)
}
