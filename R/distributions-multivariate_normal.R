.batch_mv <- function(bmat, bvec) {
  torch_matmul(bmat, bvec$unsqueeze(-1))$squeeze(-1)
}

.batch_mahalanobis <- function(bL, bx) {
  n <- tail(bx$shape, 1)
  bx_batch_shape <- bx$shape[-length(bx$shape)]

  # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
  # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve

  bx_batch_dims <- length(bx_batch_shape)
  bL_batch_dims <- bL$dim() - 2
  outer_batch_dims <- bx_batch_dims - bL_batch_dims + 1
  old_batch_dims <- outer_batch_dims + bL_batch_dims
  new_batch_dims <- outer_batch_dims + 2 * bL_batch_dims

  # Reshape bx with the shape (..., 1, i, j, 1, n)
  bx_new_shape <- head(bx$shape, outer_batch_dims - 1)
  l <- list(
    sL = head2(bL$shape, -2),
    sx = bx$shape[seq2(outer_batch_dims, length(bx$shape) - 1)]
  )
  for (x in transpose_list(l)) {
    bx_new_shape <- c(bx_new_shape, c(trunc(x$sx / x$sL), x$sL))
  }
  bx_new_shape <- c(bx_new_shape, n)
  bx$reshape(bx_new_shape)

  permute_dims <- c(
    seq2(1, outer_batch_dims - 1),
    seq2(outer_batch_dims - 1, new_batch_dims, by = 2),
    seq2(outer_batch_dims - 1, new_batch_dims, by = 2),
    new_batch_dims
  )

  bx <- bx$permute(permute_dims)

  flat_L <- bL$reshape(c(-1, n, n)) # shape = b x n x n
  flat_x <- bx$reshape(c(-1, flat_L$size(1), n)) # shape = c x b x n
  flat_x_swap <- flat_x$permute(c(2, 3, 1)) # shape = b x n x c
  M_swap <- torch_triangular_solve(flat_x_swap, flat_L, upper = FALSE)[[1]]$pow(2)$sum(-2) # shape = b x c
  M <- M_swap$t()

  # Now we revert the above reshape and permute operators.
  permuted_M <- M$reshape(head(bx$shape, length(bx$shape) - 1)) # shape = (..., 1, j, i, 1)
  permute_inv_dims <- seq_len(outer_batch_dims - 1)

  for (i in seq_len(bL_batch_dims)) {
    permute_inv_dims <- c(permute_inv_dims, c(outer_batch_dims + i - 1, old_batch_dims + i))
  }
  reshaped_M <- permuted_M$permute(permute_inv_dims)
  reshaped_M$reshape(bx_batch_shape)
}

.precision_to_scale_tril <- function(P) {
  # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
  Lf <- linalg_cholesky(torch_flip(P, c(-2, -1)))
  L_inv <- torch_transpose(torch_flip(Lf, c(-2, -1)), -2, -1)
  torch_triangular_solve(torch_eye(head2(P$shape, -1), dtype = P$dtype, device = P$device),
    L_inv,
    upper = FALSE
  )[[1]]
}

MultivariateNormal <- R6::R6Class(
  "torch_MultivariateNormal",
  lock_objects = FALSE,
  inherit = Distribution,
  public = list(
    .arg_constraints = list(
      loc = constraint_real_vector,
      covariance_matrix = constraint_positive_definite,
      precision_matrix = constraint_positive_definite,
      scale = constraint_lower_cholesky
    ),
    .support = constraint_real_vector,
    has_rsample = TRUE,
    ._mean_carrier_measure = 0,
    initialize = function(loc, covariance_matrix = NULL, precision_matrix = NULL,
                          scale_tril = NULL, validate_args = NULL) {
      if (loc$dim() < 1) {
        value_error("loc must be at least one-dimensional.")
      }

      if ((!is.null(covariance_matrix) + !is.null(precision_matrix) +
        !is.null(scale_tril)) != 1) {
        value_error("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")
      }


      if (!is.null(scale_tril)) {
        if (scale_tril$dim() < 2) {
          value_error(paste0(
            "scale_tril matrix must be at least two-dimensional ",
            "with optional leading batch dimensions"
          ))
        }

        batch_shape <- torch_broadcast_shapes(head2(scale_tril$shape, -2), head2(loc$shape, -1))
        self$scale_tril <- scale_tril$expand(c(batch_shape, c(-1, -1)))
      } else if (!is.null(covariance_matrix)) {
        if (covariance_matrix$dim() < 2) {
          value_error(paste0(
            "covariance_matrix matrix must be at least two-dimensional ",
            "with optional leading batch dimensions"
          ))
        }
        batch_shape <- torch_broadcast_shapes(
          head2(covariance_matrix$shape, -2),
          head2(loc$shape, -1)
        )
        self$covariance_matrix <- covariance_matrix$expand(c(batch_shape, c(-1, -1)))
      } else {
        if (precision_matrix$dim() < 2) {
          value_error(paste0(
            "precision_matrix matrix must be at least two-dimensional ",
            "with optional leading batch dimensions"
          ))
        }
        batch_shape <- torch_broadcast_shapes(
          head2(precision_matrix$shape, -2),
          head2(loc$shape, -1)
        )
        self$precision_matrix <- precision_matrix$expand(c(batch_shape, c(-1, -1)))
      }

      self$loc <- loc$expand(c(batch_shape, -1))
      event_shape <- tail(self$loc$shape, 1)
      super$initialize(batch_shape, event_shape, validate_args = validate_args)

      if (!is.null(scale_tril)) {
        self$.unbroadcasted_scale_tril <- scale_tril
      } else if (!is.null(covariance_matrix)) {
        self$.unbroadcasted_scale_tril <- linalg_cholesky(covariance_matrix)
      } else {
        self$.unbroadcasted_scale_tril <- .precision_to_scale_tril(precision_matrix)
      }
    },
    expand = function(batch_shape, .instance = NULL) {

      # new <- private$.get_checked_instance(self, .instance)
      new <- list()

      loc_shape <- c(batch_shape, self$event_shape)
      cov_shape <- c(batch_shape, self$event_shape, self$event_shape)

      new$loc <- self$loc$expand(loc_shape)
      # new$.unbroadcasted_scale_tril <- self$.unbroadcasted_scale_tril

      if (!is.null(self$covariance_matrix)) {
        new$covariance_matrix <- self$convariance_matrix$expand(cov_shape)
      }

      if (!is.null(self$scale_tril)) {
        new$scale_tril <- self$scale_tril$expand(cov_shape)
      }

      if (!is.null(self$scale_tril)) {
        new$precision_matrix <- self$precision_matrix$expand(cov_shape)
      }

      new <- do.call(MultivariateNormal$new, new)
      new$.unbroadcasted_scale_tril <- self$.unbroadcasted_scale_tril
      new$.validate_args <- self$.validate_args
      new
    },
    rsample = function(sample_shape = NULL) {
      shape <- self$.extended_shape(sample_shape)
      eps <- .standard_normal(shape, dtype = self$loc$dtype, device = self$loc$device)
      self$loc + .batch_mv(self$.unbroadcasted_scale_tril, eps)
    },
    log_prob = function(value) {
      if (self$.validate_args) {
        self$.validate_sample(value)
      }

      diff <- value - self$loc
      M <- .batch_mahalanobis(self$.unbroadcasted_scale_tril, diff)
      half_log_det <- self$.unbroadcasted_scale_tril$diagonal(dim1 = -2, dim2 = -1)$log()$sum(-1)
      -0.5 * (self$event_shape[1] * log(2 * pi) + M) - half_log_det
    },
    entropy = function() {
      half_log_det <- self$.unbroadcasted_scale_tril$diagonal(dim1 = -2, dim2 = -1)$log()$sum(-1)
      H <- 0.5 * self$event_shape[1] * (1.0 + log(2 * pi)) + half_log_det
      if (length(self$batch_shape) == 0) {
        H
      } else {
        H$expand(self$batch_shape)
      }
    }
  ),
  active = list(
    scale_tril = function(x) {
      if (!missing(x)) {
        private$scale_tril <- x
      }

      if (!is.null(private$scale_tril)) {
        return(private$scale_tril)
      }

      self$.unbroadcasted_scale_tril$expand(
        c(self$batch_shape, self$event_shape, self$event_shape)
      )
    },
    covariance_matrix = function(x) {
      if (!missing(x)) {
        private$covariance_matrix <- x
      }

      if (!is.null(private$covariance_matrix)) {
        return(private$covariance_matrix)
      }

      torch_matmul(
        self$.unbroadcasted_scale_tril,
        self$.unbroadcasted_scale_tril$transpose(c(-1, -2))
      )$
        expand(c(self$batch_shape, self$event_shape, self$event_shape))
    },
    precision_matrix = function(x) {
      if (!missing(x)) {
        private$precision_matrix <- x
      }

      if (!is.null(private$precision_matrix)) {
        return(private$precision_matrix)
      }

      identity <- torch_eye(tail(self$loc$shape, 1),
        device = self$loc$device,
        dtype = self$loc$dtype
      )
      torch_cholesky_solve(identity, self$.unbroadcasted_scale_tril)$expand(
        c(self$batch_shape, self$event_shape, self$event_shape)
      )
    },
    mean = function() {
      self$loc
    },
    variance = function() {
      self$.unbroadcasted_scale_tril$pow(2)$sum(-1)$expand(
        c(self$batch_shape, self$event_shape)
      )
    }
  )
)

MultivariateNormal <- add_class_definition(MultivariateNormal)

#' Gaussian distribution
#'
#' Creates a multivariate normal (also called Gaussian) distribution
#' parameterized by a mean vector and a covariance matrix.
#'
#' The multivariate normal distribution can be parameterized either
#' in terms of a positive definite covariance matrix \eqn{\mathbf{\Sigma}}
#' or a positive definite precision matrix \eqn{\mathbf{\Sigma}^{-1}}
#' or a lower-triangular matrix \eqn{\mathbf{L}} with positive-valued
#' diagonal entries, such that
#' \eqn{\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top}. This triangular matrix
#' can be obtained via e.g. Cholesky decomposition of the covariance.
#'
#' @examples
#' m <- distr_multivariate_normal(torch_zeros(2), torch_eye(2))
#' m$sample() # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
#' @param loc (Tensor): mean of the distribution
#' @param covariance_matrix (Tensor): positive-definite covariance matrix
#' @param precision_matrix (Tensor): positive-definite precision matrix
#' @param scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
#' @param validate_args Bool wether to validate the arguments or not.
#'
#' @note
#' Only one of `covariance_matrix` or `precision_matrix` or
#' `scale_tril` can be specified.
#' Using `scale_tril` will be more efficient: all computations internally
#' are based on `scale_tril`. If `covariance_matrix` or
#' `precision_matrix` is passed instead, it is only used to compute
#' the corresponding lower triangular matrices using a Cholesky decomposition.
#'
#' @seealso [Distribution] for details on the available methods.
#' @family distributions
#' @export
distr_multivariate_normal <- function(loc, covariance_matrix = NULL, precision_matrix = NULL,
                                      scale_tril = NULL, validate_args = NULL) {
  MultivariateNormal$new(
    loc, covariance_matrix, precision_matrix,
    scale_tril, validate_args
  )
}
