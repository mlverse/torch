

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

#' Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.
#' 
#' For complex `A`, it returns the angle and the natural logarithm of the modulus of the
#' determinant, that is, a logarithmic polar decomposition of the determinant.
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' 
#' @section Notes:
#' - The determinant can be recovered as `sign * exp(logabsdet)`.
#' - When a matrix has a determinant of zero, it returns `(0, -Inf)`.
#' 
#' @inheritParams linalg_det
#' 
#' @returns
#' 
#' A list `(sign, logabsdet)`.
#' `logabsdet` will always be real-valued, even when `A` is complex.
#' `sign` will have the same dtype as `A`.
#' 
#' @examples
#' a <- torch_randn(3,3)
#' linalg_slogdet(a)
#' 
#' @family linalg
#' @export
linalg_slogdet <- function(A) {
  torch_linalg_slogdet(A)
}


#' Computes the condition number of a matrix with respect to a matrix norm.
#' 
#' Letting \eqn{\mathbb{K}} be \eqn{\mathbb{R}} or \eqn{\mathbb{C}},
#' the **condition number** \eqn{\kappa} of a matrix
#' \eqn{A \in \mathbb{K}^{n \times n}} is defined as
#' 
#' \deqn{\kappa(A) = \|A\|_p\|A^{-1}\|_p}
#'   
#' The condition number of `A` measures the numerical stability of the linear system `AX = B`
#' with respect to a matrix norm.
#' 
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' 
#' `p` defines the matrix norm that is computed. See the table in 'Details' to 
#' find the supported norms.
#' 
#' For `p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`, this function uses
#' [linalg_norm()] and [linalg_inv()].
#' 
#' As such, in this case, the matrix (or every matrix in the batch) `A` has to be square
#' and invertible.
#' 
#' For `p` in `(2, -2)`, this function can be computed in terms of the singular values
#' \eqn{\sigma_1 \geq \ldots \geq \sigma_n}
#' 
#' \deqn{\kappa_2(A) = \frac{\sigma_1}{\sigma_n}\qquad \kappa_{-2}(A) = \frac{\sigma_n}{\sigma_1}}
#'     
#' In these cases, it is computed using [`torch_linalg.svd`]. For these norms, the matrix
#' (or every matrix in the batch) `A` may have any shape.
#' 
#' @note When inputs are on a CUDA device, this function synchronizes that device with the CPU if
#'  if `p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`.
#'  
#' @includeRmd man/rmd/linalg-cond.Rmd details
#' 
#' @param A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions
#' for `p` in `(2, -2)`, and of shape `(*, n, n)` where every matrix
#' is invertible for `p` in `('fro', 'nuc', inf, -inf, 1, -1)`.
#' @param p (int, inf, -inf, 'fro', 'nuc', optional):
#'   the type of the matrix norm to use in the computations (see above). Default: `NULL`
#'   
#' @returns 
#' A real-valued tensor, even when `A` is complex.
#' 
#' @examples 
#' a <- torch_tensor(rbind(c(1., 0, -1), c(0, 1, 0), c(1, 0, 1)))
#' linalg_cond(a)
#' linalg_cond(a, "fro")
#'  
#' @export
linalg_cond <- function(A, p=NULL) {
  torch_linalg_cond(a, p = p)
}

#' Computes the numerical rank of a matrix.
#' 
#' The matrix rank is computed as the number of singular values
#' (or eigenvalues in absolute value when `hermitian = TRUE`)
#' that are greater than the specified `tol` threshold.
#' 
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' 
#' If `hermitian = TRUE`, `A` is assumed to be Hermitian if complex or
#' symmetric if real, but this is not checked internally. Instead, just the lower
#' triangular part of the matrix is used in the computations.
#' 
#' If `tol` is not specified and `A` is a matrix of dimensions `(m, n)`,
#' the tolerance is set to be
#' 
#' \deqn{
#' \mbox{tol} = \sigma_1 \max(m, n) \varepsilon
#' }
#'   
#' where \eqn{\sigma_1} is the largest singular value
#' (or eigenvalue in absolute value when `hermitian = TRUE`), and
#' \eqn{\varepsilon} is the epsilon value for the dtype of `A` (see [torch_finfo()]).
#' 
#' If `A` is a batch of matrices, `tol` is computed this way for every element of
#' the batch.
#' 
#' @param A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more 
#'   batch dimensions.
#' @param tol (float, Tensor, optional): the tolerance value. See above for 
#' the value it takes when `NULL`. Default: `NULL`.
#' @param hermitian(bool, optional): indicates whether `A` is Hermitian if complex
#' or symmetric if real. Default: `FALSE`.
#' 
#' @examples 
#' a <- torch_eye(10)
#' linalg_matrix_rank(a)
#' 
#' @family linalg
#' @export
linalg_matrix_rank <- function(A, tol=NULL, hermitian=FALSE) {
  
  if (is.null(tol))
    torch_linalg_matrix_rank(self = A, tol = tol, hermitian = hermitian)
  else {
    if (!is_torch_tensor(tol))
      tol <- torch_scalar_tensor(tol)
    torch_linalg_matrix_rank(input = A, tol = tol, hermitian = hermitian)
  }
}