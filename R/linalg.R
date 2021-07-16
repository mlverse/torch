

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
  torch_linalg_cond(A, p = p)
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
#' @param hermitian (bool, optional): indicates whether `A` is Hermitian if complex
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


#' Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.
#' 
#' Letting \eqn{\mathbb{K}} be \eqn{\mathbb{R}} or \eqn{\mathbb{C}},
#' the **Cholesky decomposition** of a complex Hermitian or real symmetric positive-definite matrix
#' \eqn{A \in \mathbb{K}^{n \times n}} is defined as
#' 
#' \deqn{
#' A = LL^{\mbox{H}}\mathrlap{\qquad L \in \mathbb{K}^{n \times n}}
#' }
#' where \eqn{L} is a lower triangular matrix and
#' \eqn{L^{\mbox{H}}} is the conjugate transpose when \eqn{L} is complex, and the
#' transpose when \eqn{L} is real-valued.
#' 
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' 
#' @seealso 
#' - [linalg_cholesky_ex()] for a version of this operation that
#' skips the (slow) error checking by default and instead returns the debug
#' information. This makes it a faster way to check if a matrix is
#' positive-definite.
#' [linalg_eigh()] for a different decomposition of a Hermitian matrix.
#' The eigenvalue decomposition gives more information about the matrix but it
#' slower to compute than the Cholesky decomposition.
#' 
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
#' consisting of symmetric or Hermitian positive-definite matrices.
#' 
#' @examples
#' a <- torch_eye(10)
#' linalg_cholesky(a)
#' 
#' @family linalg
#' @export
linalg_cholesky <- function(A) {
  torch_linalg_cholesky(A)
}

#' Computes the QR decomposition of a matrix.
#' 
#' Letting \eqn{\mathbb{K}} be \eqn{\mathbb{R}} or \eqn{\mathbb{C}},
#' the **full QR decomposition** of a matrix
#' \eqn{A \in \mathbb{K}^{m \times n}} is defined as
#' 
#' \deqn{
#'   A = QR\mathrlap{\qquad Q \in \mathbb{K}^{m \times m}, R \in \mathbb{K}^{m \times n}}
#' }
#' 
#' where \eqn{Q} is orthogonal in the real case and unitary in the complex case, and \eqn{R} is upper triangular.
#' When `m > n` (tall matrix), as `R` is upper triangular, its last `m - n` rows are zero.
#' In this case, we can drop the last `m - n` columns of `Q` to form the
#' **reduced QR decomposition**:
#'
#' \deqn{
#'   A = QR\mathrlap{\qquad Q \in \mathbb{K}^{m \times n}, R \in \mathbb{K}^{n \times n}}
#' }
#' 
#' The reduced QR decomposition agrees with the full QR decomposition when `n >= m` (wide matrix).
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' The parameter `mode` chooses between the full and reduced QR decomposition.
#' 
#' If `A` has shape `(*, m, n)`, denoting `k = min(m, n)`
#' - `mode = 'reduced'` (default): Returns `(Q, R)` of shapes `(*, m, k)`, `(*, k, n)` respectively.
#' - `mode = 'complete'`: Returns `(Q, R)` of shapes `(*, m, m)`, `(*, m, n)` respectively.
#' - `mode = 'r'`: Computes only the reduced `R`. Returns `(Q, R)` with `Q` empty and `R` of shape `(*, k, n)`.
#' 
#' 
#' @param A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
#' @param mode (str, optional): one of `'reduced'`, `'complete'`, `'r'`.
#'   Controls the shape of the returned tensors. Default: `'reduced'`.
#' 
#' @returns A list `(Q, R)`.
#' 
#' @examples
#' a <- torch_tensor(rbind(c(12., -51, 4), c(6, 167, -68), c(-4, 24, -41)))
#' qr <- linalg_qr(a)
#' 
#' torch_mm(qr[[1]], qr[[2]])$round()
#' torch_mm(qr[[1]]$t(), qr[[1]])$round()
#' 
#' @family linalg
#' @export
linalg_qr <- function(A, mode='reduced') {
  torch_linalg_qr(A, mode = mode)
}

#' Computes the eigenvalue decomposition of a square matrix if it exists.
#' 
#' Letting \eqn{\mathbb{K}} be \eqn{\mathbb{R}} or \eqn{\mathbb{C}},
#' the **eigenvalue decomposition** of a square matrix
#' \eqn{A \in \mathbb{K}^{n \times n}} (if it exists) is defined as
#' 
#' \deqn{
#'   A = V \operatorname{diag}(\Lambda) V^{-1}\mathrlap{\qquad V \in \mathbb{C}^{n \times n}, \Lambda \in \mathbb{C}^n}
#' }
#' 
#' This decomposition exists if and only if \eqn{A} is `diagonalizable`_.
#' This is the case when all its eigenvalues are different.
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' 
#' @note The eigenvalues and eigenvectors of a real matrix may be complex.
#' 
#' @warning 
#' - This function assumes that `A` is `diagonalizable`_ (for example, when all the
#'   eigenvalues are different). If it is not diagonalizable, the returned
#'   eigenvalues will be correct but \eqn{A \neq V \operatorname{diag}(\Lambda)V^{-1}}.
#'  
#' - The eigenvectors of a matrix are not unique, nor are they continuous with respect to
#'   `A`. Due to this lack of uniqueness, different hardware and software may compute
#'   different eigenvectors.
#'   This non-uniqueness is caused by the fact that multiplying an eigenvector by a
#'   non-zero number produces another set of valid eigenvectors of the matrix.
#'   In this implmentation, the returned eigenvectors are normalized to have norm
#'   `1` and largest real component.
#' 
#' - Gradients computed using `V` will only be finite when `A` does not have repeated eigenvalues.
#'   Furthermore, if the distance between any two eigenvalues is close to zero,
#'   the gradient will be numerically unstable, as it depends on the eigenvalues
#'   \eqn{\lambda_i} through the computation of
#'   \eqn{\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}}.
#'   
#' @seealso 
#' - [linalg_eigvals()] computes only the eigenvalues. Unlike [linalg_eig()], the gradients of 
#'   [linalg_eigvals()] are always numerically stable.
#' - [linalg_eigh()] for a (faster) function that computes the eigenvalue decomposition
#'   for Hermitian and symmetric matrices.
#' - [linalg_svd()] for a function that computes another type of spectral
#'   decomposition that works on matrices of any shape.
#' - [linalg_qr()] for another (much faster) decomposition that works on matrices of
#'   any shape.
#'   
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
#'   consisting of diagonalizable matrices.
#' 
#' @returns
#' A list `(eigenvalues, eigenvectors)` which corresponds to \eqn{\Lambda} and \eqn{V} above.
#' `eigenvalues` and `eigenvectors` will always be complex-valued, even when `A` is real. The eigenvectors
#' will be given by the columns of `eigenvectors`.
#' 
#' @examples
#' a <- torch_randn(2, 2)
#' wv = linalg_eig(a)
#' 
#' @family linalg
#' @export
linalg_eig <- function(A) {
  torch_linalg_eig(A)
}