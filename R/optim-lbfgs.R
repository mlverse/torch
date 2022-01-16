#' @include optim.R
NULL

# Compute minimum of interpolating polynomial based on function and derivative values
# ported from https://github.com/torch/optim/blob/master/polyinterp.lua
.cubic_interpolate <-
  function(x1, f1, g1, x2, f2, g2, bounds = NULL) {
    # Compute bounds of interpolation area
    if (!is.null(bounds)) {
      xmin_bound <- bounds[1]
      xmax_bound <- bounds[2]
    } else if (x1 <= x2) {
      xmin_bound <- x1
      xmax_bound <- x2
    } else {
      xmin_bound <- x2
      xmax_bound <- x1
    }

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 <- g1$item() + g2$item() - 3 * (f1 - f2) / (x1 - x2)
    d2_square <- d1^2 - g1 * g2
    if (d2_square$item() >= 0) {
      d2 <- sqrt(d2_square)
      min_pos <- if (x1 <= x2) {
        x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
      } else {
        x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
      }
      as.numeric(min(max(min_pos, xmin_bound), xmax_bound))
    } else {
      as.numeric((xmin_bound + xmax_bound) / 2)
    }
  }


# ported from https://github.com/torch/optim/blob/master/lswolfe.lua
#
# Parameters:
# - objfunc             a function (the objective) that takes as inputs the point of evaluation,
#                         the step size, and the descent direction, and returns f(X) and df/dX
# - x                   initial point / starting location
# - t                   initial step size
# - d                   descent direction
# - f                   initial function value
# - g                   gradient at initial location
# - gtd                 directional derivative at starting location
# - c1                  sufficient decrease parameter
# - c2                  curvature parameter
# - tolerance_change    minimum allowable step length
# - max_ls              maximum nb of iterations
#
# Return values:
# - f                   function value at x+t*d
# - g                   gradient value at x+t*d
# - x                   the next x (=x+t*d)
# - t                   the step length
# - ls_func_evals       the number of function evaluations
#
#
### Rationale ###
# 1 (sufficient decrease / Armijo condition):
#    function value should decrease at least as a fraction of how it would decrease with seepest descent
# 2 (curvature condition):
#    gradient should not be too steep, or else we could hope for stronger decrease
# 3 (strong Wolfe modification):
#    gradient should not be too positive either
# ´
.strong_wolfe <- function(obj_func,
                          x,
                          t,
                          d,
                          f,
                          g,
                          gtd,
                          c1 = 1e-4,
                          c2 = 0.9,
                          tolerance_change = 1e-9,
                          max_ls = 25) {
  d_norm <- d$abs()$max()
  g <- g$clone(memory_format = torch_contiguous_format())
  # evaluate objective and gradient using initial step
  ret <- obj_func(x, t, d)
  f_new <- ret[[1]]
  g_new <- ret[[2]]
  ls_func_evals <- 1
  gtd_new <- g_new$dot(d)

  # initial phase: find a solution point, or
  # bracket an initial interval containing a point satisfying the strong Wolfe criteria
  t_prev <- 0
  f_prev <- f
  g_prev <- g
  gtd_prev <- gtd
  ls_iter <- 0
  done <- FALSE

  while (ls_iter < max_ls) {
    # sufficient decrease (Armijo) condition violated
    # construct interval between previous (smaller) step size and current
    if ((f_new > f + c1 * t * gtd$item()) ||
      (ls_iter > 1 && f_new >= f_prev)) {
      bracket <- c(t_prev, t)
      bracket_f <- c(f_prev, f_new)
      bracket_g <- c(g_prev, g_new$clone(memory_format = torch_contiguous_format()))
      bracket_gtd <- c(gtd_prev, gtd_new)
      # cat("Initial search: sufficient decrease condition violated", "\n")
      break
    }

    # curvature condition satisfied (slope not too steep)
    # return current parameters, no further zoom stage
    if (abs(gtd_new$item()) <= -c2 * gtd$item()) {
      bracket <- c(t)
      bracket_f <- c(f_new)
      bracket_g <- c(g_new)
      done <- TRUE
      # cat("Initial search: strong Wolfe condition satisfied", "\n")
      break
    }

    # curvature condition (strong Wolfe 2) violated (gradient positive)
    # construct interval between previous (smaller) step size and current
    if (gtd_new$item() >= 0) {
      bracket <- c(t_prev, t)
      bracket_f <- c(f_prev, f_new)
      bracket_g <- c(g_prev, g_new$clone(memory_format = torch_contiguous_format()))
      bracket_gtd <- c(gtd_prev, gtd_new)
      # cat("Initial search: curvature condition violated (gradient positive)", "\n")
      break
    }

    # interpolate
    min_step <- t + 0.01 * (t - t_prev)
    max_step <- t * 10
    tmp <- t
    t <- .cubic_interpolate(t_prev,
      f_prev,
      gtd_prev,
      t,
      f_new,
      gtd_new,
      bounds = c(min_step, max_step)
    )

    # next step
    t_prev <- tmp
    f_prev <- f_new
    g_prev <- g_new$clone(memory_format = torch_contiguous_format())
    gtd_prev <- gtd_new
    ret <- obj_func(x, t, d)
    f_new <- ret[[1]]
    g_new <- ret[[2]]
    ls_func_evals <- ls_func_evals + 1
    gtd_new <- g_new$dot(d)
    ls_iter <- ls_iter + 1
  }

  # reached max number of iterations?
  if (ls_iter == max_ls) {
    bracket <- c(0, t)
    bracket_f <- c(f, f_new)
    bracket_g <- c(g, g_new)
  }

  # zoom phase: we now have a point satisfying the criteria, or
  # a bracket around it. We refine the bracket until we find the
  # exact point satisfying the criteria
  insuf_progress <- FALSE
  # find high and low points in bracket
  if (bracket_f[[1]] <= bracket_f[[length(bracket_f)]]) {
    low_pos <- 1
    high_pos <- 2
  } else {
    low_pos <- 2
    high_pos <- 1
  }

  while ((done != TRUE) && (ls_iter < max_ls)) {
    # line-search bracket is so small
    if (abs(bracket[[2]] - bracket[[1]]) * d_norm$item() < tolerance_change) {
      # cat("Zoom phase: bracket too small", "\n")
      break
    }


    # compute new trial value
    t <-
      .cubic_interpolate(
        bracket[[1]],
        bracket_f[[1]],
        bracket_gtd[[1]],
        bracket[[2]],
        bracket_f[[2]],
        bracket_gtd[[2]]
      )
    # test that we are making sufficient progress:
    # in case t is too close to boundary, we mark that we are making insufficient progress,
    # and if we have made insufficient progress in the last step,
    # or t is at one of the boundaries,
    # we will move t to a position which is `0.1 * len(bracket)` away from the nearest boundary point.
    eps <- 0.1 * (max(bracket) - min(bracket))
    if (min(max(bracket) - t, t - min(bracket)) < eps) {
      # interpolation close to boundary
      if ((insuf_progress == TRUE) ||
        (t >= max(bracket)) || (t <= min(bracket))) {
        # evaluate at 0.1 away from boundary
        if (abs(t - max(bracket)) < abs(t - min(bracket))) {
          t <- max(bracket) - eps
        } else {
          t <- min(bracket) + eps
        }
        insuf_progress <- FALSE
      } else {
        insuf_progress <- TRUE
      }
    } else {
      insuf_progress <- FALSE
    }

    # Evaluate new point
    ret <- obj_func(x, t, d)
    f_new <- ret[[1]]
    g_new <- ret[[2]]
    ls_func_evals <- ls_func_evals + 1
    gtd_new <- g_new$dot(d)
    ls_iter <- ls_iter + 1

    # Armijo condition violated or not lower than lowest point
    if ((f_new > (f + c1 * t * gtd$item())) ||
      (f_new >= bracket_f[[low_pos]])) {
      # cat("Zoom phase: sufficient decrease condition violated", "\n")
      bracket[[high_pos]] <- t
      bracket_f[[high_pos]] <- f_new
      bracket_g[[high_pos]] <- g_new
      bracket_gtd[[high_pos]] <- gtd_new
      if (bracket_f[[1]] <= bracket_f[[2]]) {
        low_pos <- 1
        high_pos <- 2
      } else {
        low_pos <- 2
        high_pos <- 1
      }
    } else {
      # Wolfe conditions satisfied
      if (abs(gtd_new$item()) <= -c2 * gtd$item()) {
        done <- TRUE
        # cat("Zoom phase: strong Wolfe condition satisfied", "\n")
      } else if (gtd_new$item() * (bracket[[high_pos]] - bracket[[low_pos]]) >= 0) {
        # old high becomes new low
        bracket[[high_pos]] <- bracket[[low_pos]]
        bracket_f[[high_pos]] <- bracket_f[[low_pos]]
        bracket_g[[high_pos]] <- bracket_g[[low_pos]]
        bracket_gtd[[high_pos]] <- bracket_gtd[[low_pos]]
      }

      # new point becomes new low
      bracket[[low_pos]] <- t
      bracket_f[[low_pos]] <- f_new
      bracket_g[[low_pos]] <- g_new
      bracket_gtd[[low_pos]] <- gtd_new
    }
  }

  t <- bracket[[low_pos]]
  f_new <- torch_tensor(bracket_f[[low_pos]])
  g_new <- bracket_g[[low_pos]]
  list(f_new, g_new, t, ls_func_evals)
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
#' @includeRmd man/rmd/optim-note.Rmd note
#' @export
optim_lbfgs <- optimizer(
  "optim_lbfgs",
  initialize = function(params,
                        lr = 1,
                        max_iter = 20,
                        max_eval = NULL,
                        tolerance_grad = 1e-7,
                        tolerance_change = 1e-9,
                        history_size = 100,
                        line_search_fn = NULL) {
    if (is.null(max_eval)) {
      max_eval <- as.integer(max_iter * 5 / 4)
    }

    defaults <- list(
      lr = lr,
      max_iter = max_iter,
      max_eval = max_eval,
      tolerance_grad = tolerance_grad,
      tolerance_change = tolerance_change,
      history_size = history_size,
      line_search_fn = line_search_fn
    )

    super$initialize(params, defaults)

    if (length(self$param_groups) != 1) {
      value_error(
        "LBFGS doesn't support per-parameter options ",
        "(parameter groups)"
      )
    }

    private$.params <- self$param_groups[[1]][["params"]]
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
      lr <- group[["lr"]]
      max_iter <- group[["max_iter"]]
      max_eval <- group[["max_eval"]]
      tolerance_grad <- group[["tolerance_grad"]]
      tolerance_change <- group[["tolerance_change"]]
      line_search_fn <- group[["line_search_fn"]]
      history_size <- group[["history_size"]]

      # NOTE: LBFGS has only global state, but we register it as state for
      # the first param, because this helps with casting in load_state_dict
      if (is.null(state(private$.params[[1]]))) {
        state(private$.params[[1]]) <- new.env(parent = emptyenv())
        state(private$.params[[1]])[["func_evals"]] <- 0
        state(private$.params[[1]])[["n_iter"]] <- 0
      }
      state <- state(private$.params[[1]])

      # evaluate initial f(x) and df/dx
      orig_loss <- closure_()
      loss <- orig_loss$item()
      current_evals <- 1
      state[["func_evals"]] <- state[["func_evals"]] + 1

      flat_grad <- private$.gather_flat_grad()
      opt_cond <- (flat_grad$abs()$max() <= tolerance_grad)$item()

      if (opt_cond) {
        return(orig_loss)
      }

      # tensors cached in state (for tracing)
      d <- state[["d"]]
      t <- state[["t"]]
      old_dirs <- state[["old_dirs"]]
      old_stps <- state[["old_stps"]]
      ro <- state[["ro"]]
      H_diag <- state[["H_diag"]]
      prev_flat_grad <- state[["prev_flat_grad"]]
      prev_loss <- state[["prev_loss"]]

      n_iter <- 0
      # optimize for a max of max_iter iterations
      while (n_iter < max_iter) {
        # keep track of nb of iterations
        n_iter <- n_iter + 1
        state[["n_iter"]] <- state[["n_iter"]] + 1

        ############################################################
        # compute gradient descent direction
        ############################################################

        if (state[["n_iter"]] == 1) {
          d <- flat_grad$neg()
          old_dirs <- list()
          old_stps <- list()
          ro <- list()
          H_diag <- 1
        } else {
          # do lbfgs update (update memory)
          y <- flat_grad$sub(prev_flat_grad)
          s <- d$mul(t)
          ys <- y$dot(s) # y*s

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
            H_diag <- ys / y$dot(y) # (y*y)
          }

          # compute the approximate (L-BFGS) inverse Hessian
          # multiplied by the gradient
          num_old <- length(old_dirs)

          if (is.null(state[["al"]])) {
            state[["al"]] <- vector(mode = "list", length = history_size)
          }

          al <- state[["al"]]

          # iteration in L-BFGS loop collapsed to use just one buffer
          q <- flat_grad$neg()
          if (num_old >= 1) {
            for (i in seq(num_old, 1, by = -1)) {
              al[[i]] <- old_stps[[i]]$dot(q) * ro[[i]]
              q$add_(old_dirs[[i]], alpha = -al[[i]])
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

        if (is.null(prev_flat_grad) || is_undefined_tensor(prev_flat_grad)) {
          prev_flat_grad <- flat_grad$clone(memory_format = torch_contiguous_format())
        } else {
          prev_flat_grad$copy_(flat_grad)
        }
        prev_loss <- loss

        ############################################################
        # compute step length
        ############################################################
        # reset initial guess for step size
        if (state[["n_iter"]] == 1) {
          t <- min(1., 1. / flat_grad$abs()$sum()$item()) * lr
        } else {
          t <- lr
        }

        # directional derivative
        gtd <- flat_grad$dot(d) # g * d

        # directional derivative is below tolerance
        if (!is.na(gtd$item()) && gtd$item() > (-tolerance_change)) {
          break
        }

        # optional line search: user function
        ls_func_evals <- 0

        if (!is.null(line_search_fn)) {
          if (line_search_fn != "strong_wolfe") {
            value_error("only strong_wolfe is supported")
          } else {
            x_init <- private$.clone_param()

            obj_func <- function(x, t, d) {
              private$.directional_evaluate(closure_, x, t, d)
            }

            ret <- .strong_wolfe(obj_func, x_init, t, d, loss, flat_grad, gtd)

            loss <- ret[[1]]$item()
            flat_grad <- ret[[2]]
            t <- ret[[3]]
            ls_func_evals <- ret[[4]]

            private$.add_grad(t, d)
            opt_cond <- flat_grad$abs()$max()$item() <= tolerance_grad
          }
        } else {
          # no line search, simply move with fixed-step
          private$.add_grad(t, d)
          if (n_iter != max_iter) {
            # re-evaluate function only if not in last iteration
            # the reason we do this: in a stochastic setting,
            # no use to re-evaluate that function here
            loss <- closure_()$item()
            flat_grad <- private$.gather_flat_grad()
            opt_cond <- flat_grad$abs()$max()$item() <= tolerance_grad
            ls_func_evals <- 1
          }
        }

        # update func eval
        current_evals <- current_evals + ls_func_evals
        state[["func_evals"]] <- state[["func_evals"]] + ls_func_evals

        ############################################################
        # check conditions
        ############################################################
        if (n_iter == max_iter) {
          break
        }

        if (current_evals >= max_eval) {
          break
        }

        # optimal condition
        if (!is.na(opt_cond) && opt_cond) {
          break
        }

        # lack of progress
        d_ml <- d$mul(t)$abs()$max()$item()
        if (!is.na(d_ml) && d_ml <= tolerance_change) {
          break
        }

        if (!is.na(loss) && abs(loss - prev_loss) < tolerance_change) {
          break
        }
      }

      state[["d"]] <- d
      state[["t"]] <- t
      state[["old_dirs"]] <- old_dirs
      state[["old_stps"]] <- old_stps
      state[["ro"]] <- ro
      state[["H_diag"]] <- H_diag
      state[["prev_flat_grad"]] <- prev_flat_grad
      state[["prev_loss"]] <- prev_loss
    })

    orig_loss
  },
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
      for (i in seq_along(private$.params)) {
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
