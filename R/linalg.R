

#' Computes a vector or matrix norm.
#' 
#' If `A` is complex valued, it computes the norm of `A$abs()`
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Whether this function computes a vector or matrix norm is determined as follows:
#'   
#' * If `dim` is an int, the vector norm will be computed.
#' * If `dim` is a 2-tuple, the matrix norm will be computed.
#' * If `dim=NULL` and `ord=NULL`, A will be flattened to 1D and the 2-norm of the resulting vector will be computed.
#' * If `dim=NULL` and `ord!=NULL`, A must be 1D or 2D.
#' 
#' @includeRmd man/rmd/linalg-norm.Rmd details
#' 
#' @param A (Tensor): tensor of shape `(*, n)` or `(*, m, n)` where `*` is zero or more batch dimensions
#' @param ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `NULL`
#' @param dim (int, Tuple[int], optional): dimensions over which to compute
#'   the vector or matrix norm. See above for the behavior when `dim=NULL`.
#'   Default: `NULL`
#' @param keepdim (bool, optional): If set to `TRUE`, the reduced dimensions are retained
#'   in the result as dimensions with size one. Default: `FALSE`
#' @param dtype dtype (`torch_dtype`, optional): If specified, the input tensor is cast to
#'  `dtype` before performing the operation, and the returned tensor's type
#'  will be `dtype`. Default: `NULL`
#'  
#' @family linalg
#'  
#' @examples 
#' a <- torch_arange(0, 8, dtype=torch_float()) - 4
#' a
#' b <- a$reshape(c(3, 3))
#' b
#' 
#' linalg_norm(a)
#' linalg_norm(b)
#' 
#' @export
linalg_norm <- function(A, ord = NULL, dim = NULL, keepdim = FALSE, dtype = NULL) {
  torch_linalg_norm(self = A, ord = ord, dim = dim, keepdim = keepdim, dtype = dtype)
}

#' Computes a vector norm.
#' 
#' If `A` is complex valued, it computes the norm of `A$abs()`
#' Supports input of float, double, cfloat and cdouble dtypes.
#' This function does not necessarily treat multidimensonal `A` as a batch of
#' vectors, instead:
#' 
#' - If `dim=NULL`, `A` will be flattened before the norm is computed.
#' - If `dim` is an `int` or a `tuple`, the norm will be computed over these dimensions
#'   and the other dimensions will be treated as batch dimensions.
#' 
#' This behavior is for consistency with [linalg_norm()].
#' 
#' @includeRmd man/rmd/linalg-norm.Rmd details
#' @family linalg
#' 
#' @param A (Tensor): tensor, flattened by default, but this behavior can be
#'   controlled using `dim`.
#' @param ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `2`
#' 
#' @inheritParams linalg_norm
#' 
#' @examples
#' a <- torch_arange(0, 8, dtype=torch_float()) - 4
#' a
#' b <- a$reshape(c(3, 3))
#' b
#' 
#' linalg_vector_norm(a, ord = 3.5)
#' linalg_vector_norm(b, ord = 3.5)
#' 
#' @export
linalg_vector_norm <- function(A, ord=2, dim=NULL, keepdim=FALSE, dtype=NULL) {
  torch_linalg_vector_norm(
    self = A,
    ord = ord,
    dim = dim,
    keepdim = keepdim,
    dtype = dtype
  )
}

#' Computes a matrix norm.
#' 
#' If `A` is complex valued, it computes the norm of `A$abs()`
#' Support input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices: the norm will be computed over the
#' dimensions specified by the 2-tuple `dim` and the other dimensions will
#' be treated as batch dimensions. The output will have the same batch dimensions.
#' 
#' @includeRmd man/rmd/linalg-norm.Rmd details
#' @family linalg
#' 
#' @param A (Tensor): tensor with two or more dimensions. By default its
#'   shape is interpreted as `(*, m, n)` where `*` is zero or more
#'   batch dimensions, but this behavior can be controlled using `dim`.
#' @param ord (int, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `'fro'`
#' @inheritParams linalg_norm
#' 
#' @examples 
#' a <- torch_arange(0, 8, dtype=torch_float())$reshape(c(3,3))
#' linalg_matrix_norm(a)
#' linalg_matrix_norm(a, ord = -1)
#' b <- a$expand(c(2, -1, -1))
#' linalg_matrix_norm(b)
#' linalg_matrix_norm(b, dim = c(1, 3))
#' 
#' @export
linalg_matrix_norm <- function(A, ord='fro', dim=c(-2, -1), keepdim=FALSE, dtype=NULL) {
  torch_linalg_matrix_norm(
    self = A,
    ord = ord,
    dim = dim,
    keepdim = keepdim,
    dtype = dtype
  )
}

#' Computes the determinant of a square matrix.
#' 
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' 
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.
#' 
#' @examples
#' a <- torch_randn(3,3)
#' linalg_det(a)
#' 
#' a <- torch_randn(3,3,3)
#' linalg_det(a)
#' 
#' @family linalg
#' @export
linalg_det <- function(A) {
  torch_linalg_det(A)
}