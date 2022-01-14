

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
#' a <- torch_arange(0, 8, dtype = torch_float()) - 4
#' a
#' b <- a$reshape(c(3, 3))
#' b
#'
#' linalg_norm(a)
#' linalg_norm(b)
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
#' a <- torch_arange(0, 8, dtype = torch_float()) - 4
#' a
#' b <- a$reshape(c(3, 3))
#' b
#'
#' linalg_vector_norm(a, ord = 3.5)
#' linalg_vector_norm(b, ord = 3.5)
#' @export
linalg_vector_norm <- function(A, ord = 2, dim = NULL, keepdim = FALSE, dtype = NULL) {
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
#' a <- torch_arange(0, 8, dtype = torch_float())$reshape(c(3, 3))
#' linalg_matrix_norm(a)
#' linalg_matrix_norm(a, ord = -1)
#' b <- a$expand(c(2, -1, -1))
#' linalg_matrix_norm(b)
#' linalg_matrix_norm(b, dim = c(1, 3))
#' @export
linalg_matrix_norm <- function(A, ord = "fro", dim = c(-2, -1), keepdim = FALSE, dtype = NULL) {
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
#' a <- torch_randn(3, 3)
#' linalg_det(a)
#'
#' a <- torch_randn(3, 3, 3)
#' linalg_det(a)
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
#' a <- torch_randn(3, 3)
#' linalg_slogdet(a)
#' @family linalg
#' @export
linalg_slogdet <- function(A) {
  torch_linalg_slogdet(A)
}


#' Computes the condition number of a matrix with respect to a matrix norm.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **condition number** \teqn{\kappa} of a matrix
#' \teqn{A \in \mathbb{K}^{n \times n}} is defined as
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("\\\\kappa(A) = \\\\|A\\\\|_p\\\\|A^{-1}\\\\|_p")}
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
#' \teqn{\sigma_1 \geq \ldots \geq \sigma_n}
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("\\\\kappa_2(A) = \\\\frac{\\\\sigma_1}{\\\\sigma_n}\\\\qquad \\\\kappa_{-2}(A) = \\\\frac{\\\\sigma_n}{\\\\sigma_1}")}
#'
#' In these cases, it is computed using [linalg_svd()]. For these norms, the matrix
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
#' @export
linalg_cond <- function(A, p = NULL) {
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
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#' tol = \\\\sigma_1 \\\\max(m, n) \\\\varepsilon
#' ")}
#'
#' where \teqn{\sigma_1} is the largest singular value
#' (or eigenvalue in absolute value when `hermitian = TRUE`), and
#' \teqn{\varepsilon} is the epsilon value for the dtype of `A` (see [torch_finfo()]).
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
#' @family linalg
#' @export
linalg_matrix_rank <- function(A, tol = NULL, hermitian = FALSE) {
  if (is.null(tol)) {
    torch_linalg_matrix_rank(self = A, tol = tol, hermitian = hermitian)
  } else {
    if (!is_torch_tensor(tol)) {
      tol <- torch_scalar_tensor(tol)
    }
    torch_linalg_matrix_rank(input = A, tol = tol, hermitian = hermitian)
  }
}


#' Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **Cholesky decomposition** of a complex Hermitian or real symmetric positive-definite matrix
#' \teqn{A \in \mathbb{K}^{n \times n}} is defined as
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#' A = LL^{H}\\\\mathrlap{\\\\qquad L \\\\in \\\\mathbb{K}^{n \\\\times n}}
#' ")}
#'
#' where \teqn{L} is a lower triangular matrix and
#' \teqn{L^{H}} is the conjugate transpose when \teqn{L} is complex, and the
#' transpose when \teqn{L} is real-valued.
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
#' @family linalg
#' @export
linalg_cholesky <- function(A) {
  torch_linalg_cholesky(A)
}

#' Computes the QR decomposition of a matrix.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **full QR decomposition** of a matrix
#' \teqn{A \in \mathbb{K}^{m \times n}} is defined as
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   A = QR\\\\mathrlap{\\\\qquad Q \\\\in \\\\mathbb{K}^{m \\\\times m}, R \\\\in \\\\mathbb{K}^{m \\\\times n}}
#' ")}
#'
#' where \teqn{Q} is orthogonal in the real case and unitary in the complex case, and \teqn{R} is upper triangular.
#' When `m > n` (tall matrix), as `R` is upper triangular, its last `m - n` rows are zero.
#' In this case, we can drop the last `m - n` columns of `Q` to form the
#' **reduced QR decomposition**:
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   A = QR\\\\mathrlap{\\\\qquad Q \\\\in \\\\mathbb{K}^{m \\\\times n}, R \\\\in \\\\mathbb{K}^{n \\\\times n}}
#' ")}
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
#' @family linalg
#' @export
linalg_qr <- function(A, mode = "reduced") {
  torch_linalg_qr(A, mode = mode)
}

#' Computes the eigenvalue decomposition of a square matrix if it exists.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **eigenvalue decomposition** of a square matrix
#' \teqn{A \in \mathbb{K}^{n \times n}} (if it exists) is defined as
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   A = V \\\\operatorname{diag}(\\\\Lambda) V^{-1}\\\\mathrlap{\\\\qquad V \\\\in \\\\mathbb{C}^{n \\\\times n}, \\\\Lambda \\\\in \\\\mathbb{C}^n}
#' ")}
#'
#' This decomposition exists if and only if \teqn{A} is `diagonalizable`_.
#' This is the case when all its eigenvalues are different.
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#'
#' @note The eigenvalues and eigenvectors of a real matrix may be complex.
#'
#' @section Warning:
#'
#' - This function assumes that `A` is `diagonalizable`_ (for example, when all the
#'   eigenvalues are different). If it is not diagonalizable, the returned
#'   eigenvalues will be correct but \teqn{A \neq V \operatorname{diag}(\Lambda)V^{-1}}.
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
#'   \teqn{\lambda_i} through the computation of
#'   \teqn{\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}}.
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
#' A list `(eigenvalues, eigenvectors)` which corresponds to \teqn{\Lambda} and \teqn{V} above.
#' `eigenvalues` and `eigenvectors` will always be complex-valued, even when `A` is real. The eigenvectors
#' will be given by the columns of `eigenvectors`.
#'
#' @examples
#' a <- torch_randn(2, 2)
#' wv <- linalg_eig(a)
#' @family linalg
#' @export
linalg_eig <- function(A) {
  torch_linalg_eig(A)
}

#' Computes the eigenvalues of a square matrix.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **eigenvalues** of a square matrix \teqn{A \in \mathbb{K}^{n \times n}} are defined
#' as the roots (counted with multiplicity) of the polynomial `p` of degree `n` given by
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   p(\\\\lambda) = \\\\operatorname{det}(A - \\\\lambda \\\\mathrm{I}_n)\\\\mathrlap{\\\\qquad \\\\lambda \\\\in \\\\mathbb{C}}
#' ")}
#'
#' where \teqn{\mathrm{I}_n} is the `n`-dimensional identity matrix.
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#'
#' @note The eigenvalues of a real matrix may be complex, as the roots of a real polynomial may be complex.
#'       The eigenvalues of a matrix are always well-defined, even when the matrix is not diagonalizable.
#'
#' @seealso [linalg_eig()] computes the full eigenvalue decomposition.
#'
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.
#'
#' @examples
#' a <- torch_randn(2, 2)
#' w <- linalg_eigvals(a)
#' @family linalg
#' @export
linalg_eigvals <- function(A) {
  torch_linalg_eigvals(A)
}

#' Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **eigenvalue decomposition** of a complex Hermitian or real symmetric matrix
#' \teqn{A \in \mathbb{K}^{n \times n}} is defined as
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   A = Q \\\\operatorname{diag}(\\\\Lambda) Q^{H}\\\\mathrlap{\\\\qquad Q \\\\in \\\\mathbb{K}^{n \\\\times n}, \\\\Lambda \\\\in \\\\mathbb{R}^n}
#' ")}
#'
#' where \teqn{Q^{H}} is the conjugate transpose when \teqn{Q} is complex, and the transpose when \teqn{Q} is real-valued.
#' \teqn{Q} is orthogonal in the real case and unitary in the complex case.
#'
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#'
#' `A` is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead:
#' - If `UPLO`\ `= 'L'` (default), only the lower triangular part of the matrix is used in the computation.
#' - If `UPLO`\ `= 'U'`, only the upper triangular part of the matrix is used.
#' The eigenvalues are returned in ascending order.
#'
#' @note The eigenvalues of real symmetric or complex Hermitian matrices are always real.
#'
#' @section Warning:
#' - The eigenvectors of a symmetric matrix are not unique, nor are they continuous with
#'   respect to `A`. Due to this lack of uniqueness, different hardware and
#'   software may compute different eigenvectors.
#'   This non-uniqueness is caused by the fact that multiplying an eigenvector by
#'   `-1` in the real case or by \teqn{e^{i \phi}, \phi \in \mathbb{R}} in the complex
#'   case produces another set of valid eigenvectors of the matrix.
#'   This non-uniqueness problem is even worse when the matrix has repeated eigenvalues.
#'   In this case, one may multiply the associated eigenvectors spanning
#'   the subspace by a rotation matrix and the resulting eigenvectors will be valid
#'   eigenvectors.
#' - Gradients computed using the `eigenvectors` tensor will only be finite when
#'   `A` has unique eigenvalues.
#'   Furthermore, if the distance between any two eigvalues is close to zero,
#'   the gradient will be numerically unstable, as it depends on the eigenvalues
#'   \teqn{\lambda_i} through the computation of
#'   \teqn{\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}}.
#'
#' @seealso
#' - [linalg_eigvalsh()] computes only the eigenvalues values of a Hermitian matrix.
#'   Unlike [linalg_eigh()], the gradients of [linalg_eigvalsh()] are always
#'   numerically stable.
#' - [linalg_cholesky()] for a different decomposition of a Hermitian matrix.
#'   The Cholesky decomposition gives less information about the matrix but is much faster
#'   to compute than the eigenvalue decomposition.
#' - [linalg_eig()] for a (slower) function that computes the eigenvalue decomposition
#'   of a not necessarily Hermitian square matrix.
#' - [linalg_svd()] for a (slower) function that computes the more general SVD
#'   decomposition of matrices of any shape.
#' - [linalg_qr()] for another (much faster) decomposition that works on general
#'   matrices.
#'
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
#'   consisting of symmetric or Hermitian matrices.
#' @param UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
#'   of `A` in the computations. Default: `'L'`.
#'
#' @returns
#' A list `(eigenvalues, eigenvectors)` which corresponds to \teqn{\Lambda} and \teqn{Q} above.
#' `eigenvalues` will always be real-valued, even when `A` is complex.
#'
#' It will also be ordered in ascending order.
#' `eigenvectors` will have the same dtype as `A` and will contain the eigenvectors as its columns.
#'
#' @examples
#' a <- torch_randn(2, 2)
#' linalg_eigh(a)
#' @family linalg
#' @export
linalg_eigh <- function(A, UPLO = "L") {
  torch_linalg_eigh(A, UPLO)
}

#' Computes the eigenvalues of a complex Hermitian or real symmetric matrix.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **eigenvalues** of a complex Hermitian or real symmetric  matrix \teqn{A \in \mathbb{K}^{n \times n}}
#' are defined as the roots (counted with multiplicity) of the polynomial `p` of degree `n` given by
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   p(\\\\lambda) = \\\\operatorname{det}(A - \\\\lambda \\\\mathrm{I}_n)\\\\mathrlap{\\\\qquad \\\\lambda \\\\in \\\\mathbb{R}}
#' ")}
#'
#' where \teqn{\mathrm{I}_n} is the `n`-dimensional identity matrix.
#'
#' The eigenvalues of a real symmetric or complex Hermitian matrix are always real.
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' The eigenvalues are returned in ascending order.
#'
#' `A` is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead:
#' - If `UPLO`\ `= 'L'` (default), only the lower triangular part of the matrix is used in the computation.
#' - If `UPLO`\ `= 'U'`, only the upper triangular part of the matrix is used.
#'
#'
#' @seealso
#' - [linalg_eigh()] computes the full eigenvalue decomposition.
#'
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
#'   consisting of symmetric or Hermitian matrices.
#' @param UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
#'   of `A` in the computations. Default: `'L'`.
#'
#' @returns
#' A real-valued tensor cointaining the eigenvalues even when `A` is complex.
#' The eigenvalues are returned in ascending order.
#'
#' @examples
#' a <- torch_randn(2, 2)
#' linalg_eigvalsh(a)
#' @family linalg
#' @export
linalg_eigvalsh <- function(A, UPLO = "L") {
  torch_linalg_eigvalsh(A, UPLO = UPLO)
}

#' Computes the singular value decomposition (SVD) of a matrix.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **full SVD** of a matrix
#' \teqn{A \in \mathbb{K}^{m \times n}}, if `k = min(m,n)`, is defined as
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   A = U \\\\operatorname{diag}(S) V^{H} \\\\mathrlap{\\\\qquad U \\\\in \\\\mathbb{K}^{m \\\\times m}, S \\\\in \\\\mathbb{R}^k, V \\\\in \\\\mathbb{K}^{n \\\\times n}}
#' ")}
#'
#' where \teqn{\operatorname{diag}(S) \in \mathbb{K}^{m \times n}},
#' \teqn{V^{H}} is the conjugate transpose when \teqn{V} is complex, and the transpose when \teqn{V} is real-valued.
#'
#' The matrices  \teqn{U}, \teqn{V} (and thus \teqn{V^{H}}) are orthogonal in the real case, and unitary in the complex case.
#' When `m > n` (resp. `m < n`) we can drop the last `m - n` (resp. `n - m`) columns of `U` (resp. `V`) to form the **reduced SVD**:
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   A = U \\\\operatorname{diag}(S) V^{H} \\\\mathrlap{\\\\qquad U \\\\in \\\\mathbb{K}^{m \\\\times k}, S \\\\in \\\\mathbb{R}^k, V \\\\in \\\\mathbb{K}^{k \\\\times n}}
#' ")}
#'
#' where \teqn{\operatorname{diag}(S) \in \mathbb{K}^{k \times k}}.
#'
#' In this case, \teqn{U} and \teqn{V} also have orthonormal columns.
#' Supports input of float, double, cfloat and cdouble dtypes.
#'
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#'
#' The returned decomposition is a named tuple `(U, S, V)`
#' which corresponds to \teqn{U}, \teqn{S}, \teqn{V^{H}} above.
#'
#' The singular values are returned in descending order.
#' The parameter `full_matrices` chooses between the full (default) and reduced SVD.
#'
#' @note
#' When `full_matrices=TRUE`, the gradients with respect to `U[..., :, min(m, n):]`
#' and `Vh[..., min(m, n):, :]` will be ignored, as those vectors can be arbitrary bases
#' of the corresponding subspaces.
#'
#' @section Warnings:
#' The returned tensors `U` and `V` are not unique, nor are they continuous with
#' respect to `A`.
#' Due to this lack of uniqueness, different hardware and software may compute
#' different singular vectors.
#' This non-uniqueness is caused by the fact that multiplying any pair of singular
#' vectors \teqn{u_k, v_k} by `-1` in the real case or by
#' \teqn{e^{i \phi}, \phi \in \mathbb{R}} in the complex case produces another two
#' valid singular vectors of the matrix.
#' This non-uniqueness problem is even worse when the matrix has repeated singular values.
#' In this case, one may multiply the associated singular vectors of `U` and `V` spanning
#' the subspace by a rotation matrix and the resulting vectors will span the same subspace.
#'
#' Gradients computed using `U` or `V` will only be finite when
#' `A` does not have zero as a singular value or repeated singular values.
#' Furthermore, if the distance between any two singular values is close to zero,
#' the gradient will be numerically unstable, as it depends on the singular values
#' \teqn{\sigma_i} through the computation of
#' \teqn{\frac{1}{\min_{i \neq j} \sigma_i^2 - \sigma_j^2}}.
#' The gradient will also be numerically unstable when `A` has small singular
#' values, as it also depends on the computaiton of \teqn{\frac{1}{\sigma_i}}.
#'
#' @seealso
#' - [linalg_svdvals()] computes only the singular values.
#'   Unlike [linalg_svd()], the gradients of [linalg_svdvals()] are always
#'   numerically stable.
#' - [linalg_eig()] for a function that computes another type of spectral
#'   decomposition of a matrix. The eigendecomposition works just on on square matrices.
#' - [linalg_eigh()] for a (faster) function that computes the eigenvalue decomposition
#'   for Hermitian and symmetric matrices.
#' - [linalg_qr()] for another (much faster) decomposition that works on general
#'   matrices.
#'
#' @param A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
#' @param full_matrices (bool, optional): controls whether to compute the full or reduced
#'        SVD, and consequently, the shape of the returned tensors `U` and `V`. Default: `TRUE`.
#'
#' @returns
#' A list `(U, S, V)` which corresponds to \teqn{U}, \teqn{S}, \teqn{V^{H}} above.
#' `S` will always be real-valued, even when `A` is complex.
#' It will also be ordered in descending order.
#' `U` and `V` will have the same dtype as `A`. The left / right singular vectors will be given by
#' the columns of `U` and the rows of `V` respectively.
#'
#' @examples
#'
#' a <- torch_randn(5, 3)
#' linalg_svd(a, full_matrices = FALSE)
#' @family linalg
#' @export
linalg_svd <- function(A, full_matrices = TRUE) {
  torch_linalg_svd(A, full_matrices = full_matrices)
}

#' Computes the singular values of a matrix.
#'
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' The singular values are returned in descending order.
#'
#' @seealso
#' [linalg_svd()] computes the full singular value decomposition.
#'
#' @param A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
#'
#' @returns
#' A real-valued tensor, even when `A` is complex.
#'
#' @examples
#' A <- torch_randn(5, 3)
#' S <- linalg_svdvals(A)
#' S
#' @family linalg
#' @export
linalg_svdvals <- function(A) {
  torch_linalg_svdvals(A)
}

#' Computes the solution of a square system of linear equations with a unique solution.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' this function computes the solution \teqn{X \in \mathbb{K}^{n \times k}} of the **linear system** associated to
#' \teqn{A \in \mathbb{K}^{n \times n}, B \in \mathbb{K}^{m \times k}}, which is defined as
#'
#' \deqn{
#'   AX = B
#' }
#'
#' This system of linear equations has one solution if and only if \teqn{A} is `invertible`_.
#' This function assumes that \teqn{A} is invertible.
#' Supports inputs of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if the inputs are batches of matrices then
#' the output has the same batch dimensions.
#'
#' Letting `*` be zero or more batch dimensions,
#'
#' - If `A` has shape `(*, n, n)` and `B` has shape `(*, n)` (a batch of vectors) or shape
#'   `(*, n, k)` (a batch of matrices or "multiple right-hand sides"), this function returns `X` of shape
#'   `(*, n)` or `(*, n, k)` respectively.
#' - Otherwise, if `A` has shape `(*, n, n)` and  `B` has shape `(n,)`  or `(n, k)`, `B`
#'   is broadcasted to have shape `(*, n)` or `(*, n, k)` respectively.
#'
#' This function then returns the solution of the resulting batch of systems of linear equations.
#'
#' @note
#' This function computes `X = A$inverse() @ B` in a faster and
#' more numerically stable way than performing the computations separately.
#'
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.
#' @param B (Tensor): right-hand side tensor of shape `(*, n)` or  `(*, n, k)` or `(n,)` or `(n, k)`
#'   according to the rules described above
#'
#' @examples
#' A <- torch_randn(3, 3)
#' b <- torch_randn(3)
#' x <- linalg_solve(A, b)
#' torch_allclose(torch_matmul(A, x), b)
#' @family linalg
#' @export
linalg_solve <- function(A, B) {
  torch_linalg_solve(A, B)
}


#' Computes a solution to the least squares problem of a system of linear equations.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' the **least squares problem** for a linear system \teqn{AX = B} with
#' \teqn{A \in \mathbb{K}^{m \times n}, B \in \mathbb{K}^{m \times k}} is defined as
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   \\\\min_{X \\\\in \\\\mathbb{K}^{n \\\\times k}} \\\\|AX - B\\\\|_F
#' ")}
#'
#' where \teqn{\|-\|_F} denotes the Frobenius norm.
#' Supports inputs of float, double, cfloat and cdouble dtypes.
#'
#' Also supports batches of matrices, and if the inputs are batches of matrices then
#' the output has the same batch dimensions.
#' `driver` chooses the LAPACK/MAGMA function that will be used.
#'
#' For CPU inputs the valid values are `'gels'`, `'gelsy'`, `'gelsd`, `'gelss'`.
#' For CUDA input, the only valid driver is `'gels'`, which assumes that `A` is full-rank.
#'
#' To choose the best driver on CPU consider:
#' - If `A` is well-conditioned (its [condition number](https://pytorch.org/docs/master/linalg.html#torch.linalg.cond) is not too large), or you do not mind some precision loss.
#' - For a general matrix: `'gelsy'` (QR with pivoting) (default)
#' - If `A` is full-rank: `'gels'` (QR)
#' - If `A` is not well-conditioned.
#' - `'gelsd'` (tridiagonal reduction and SVD)
#' - But if you run into memory issues: `'gelss'` (full SVD).
#'
#' See also the [full description of these drivers](https://www.netlib.org/lapack/lug/node27.html)
#'
#' `rcond` is used to determine the effective rank of the matrices in `A`
#' when `driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`).
#' In this case, if \teqn{\sigma_i} are the singular values of `A` in decreasing order,
#' \teqn{\sigma_i} will be rounded down to zero if \teqn{\sigma_i \leq rcond \cdot \sigma_1}.
#' If `rcond = NULL` (default), `rcond` is set to the machine precision of the dtype of `A`.
#'
#' This function returns the solution to the problem and some extra information in a list of
#' four tensors `(solution, residuals, rank, singular_values)`. For inputs `A`, `B`
#' of shape `(*, m, n)`, `(*, m, k)` respectively, it cointains
#' - `solution`: the least squares solution. It has shape `(*, n, k)`.
#' - `residuals`: the squared residuals of the solutions, that is, \teqn{\|AX - B\|_F^2}.
#' It has shape equal to the batch dimensions of `A`.
#' It is computed when `m > n` and every matrix in `A` is full-rank,
#' otherwise, it is an empty tensor.
#' If `A` is a batch of matrices and any matrix in the batch is not full rank,
#' then an empty tensor is returned. This behavior may change in a future PyTorch release.
#' - `rank`: tensor of ranks of the matrices in `A`.
#' It has shape equal to the batch dimensions of `A`.
#' It is computed when `driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`),
#' otherwise it is an empty tensor.
#' - `singular_values`: tensor of singular values of the matrices in `A`.
#' It has shape `(*, min(m, n))`.
#' It is computed when `driver` is one of (`'gelsd'`, `'gelss'`),
#' otherwise it is an empty tensor.
#'
#' @note
#' This function computes `X = A$pinverse() %*% B` in a faster and
#' more numerically stable way than performing the computations separately.
#'
#' @section Warning:
#' The default value of `rcond` may change in a future PyTorch release.
#' It is therefore recommended to use a fixed value to avoid potential
#' breaking changes.
#'
#' @param A (Tensor): lhs tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
#' @param B (Tensor): rhs tensor of shape `(*, m, k)` where `*` is zero or more batch dimensions.
#' @param rcond (float, optional): used to determine the effective rank of `A`.
#'   If `rcond = NULL`, `rcond` is set to the machine
#'   precision of the dtype of `A` times `max(m, n)`. Default: `NULL`.
#' @param ... currently unused.
#' @param driver (str, optional): name of the LAPACK/MAGMA method to be used.
#'   If `NULL`, `'gelsy'` is used for CPU inputs and `'gels'` for CUDA inputs.
#'   Default: `NULL`.
#'
#' @returns
#' A list `(solution, residuals, rank, singular_values)`.
#'
#' @examples
#' A <- torch_tensor(rbind(c(10, 2, 3), c(3, 10, 5), c(5, 6, 12)))$unsqueeze(1) # shape (1, 3, 3)
#' B <- torch_stack(list(
#'   rbind(c(2, 5, 1), c(3, 2, 1), c(5, 1, 9)),
#'   rbind(c(4, 2, 9), c(2, 0, 3), c(2, 5, 3))
#' ), dim = 1) # shape (2, 3, 3)
#' X <- linalg_lstsq(A, B)$solution # A is broadcasted to shape (2, 3, 3)
#' @family linalg
#' @export
linalg_lstsq <- function(A, B, rcond = NULL, ..., driver = NULL) {
  ellipsis::check_dots_empty()

  args <- list(
    self = A,
    b = B
  )

  if (is.null(driver)) {
    if (!is_torch_tensor(A) || is_cpu_device(A$device)) {
      driver <- "gelsy"
    } else {
      driver <- "gels"
    }
  }

  args$driver <- driver

  if (!is.null(rcond)) {
    args$rcond <- rcond
  }

  res <- do.call(torch_linalg_lstsq, args)
  res <- setNames(res, c("solution", "residuals", "rank", "singular_values"))
  res
}

#' Computes the inverse of a square matrix if it exists.
#'
#' Throws a `runtime_error` if the matrix is not invertible.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' for a matrix \teqn{A \in \mathbb{K}^{n \times n}},
#' its **inverse matrix** \teqn{A^{-1} \in \mathbb{K}^{n \times n}} (if it exists) is defined as
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#'   A^{-1}A = AA^{-1} = \\\\mathrm{I}_n
#' ")}
#' where \teqn{\mathrm{I}_n} is the `n`-dimensional identity matrix.
#'
#' The inverse matrix exists if and only if \teqn{A} is invertible. In this case,
#' the inverse is unique.
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices
#' then the output has the same batch dimensions.
#'
#' Consider using [linalg_solve()] if possible for multiplying a matrix on the left by
#' the inverse, as `linalg_solve(A, B) == A$inv() %*% B`
#' It is always prefered to use [linalg_solve()] when possible, as it is faster and more
#' numerically stable than computing the inverse explicitly.
#'
#' @seealso
#' [linalg_pinv()] computes the pseudoinverse (Moore-Penrose inverse) of matrices
#' of any shape.
#' [linalg_solve()] computes `A$inv() %*% B` with a
#' numerically stable algorithm.
#'
#'
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
#'   consisting of invertible matrices.
#'
#' @examples
#' A <- torch_randn(4, 4)
#' linalg_inv(A)
#' @family linalg
#' @export
linalg_inv <- function(A) {
  torch_linalg_inv(self = A)
}

#' Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.
#'
#' The pseudoinverse may be `defined algebraically`_
#' but it is more computationally convenient to understand it `through the SVD`_
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#'
#' If `hermitian= TRUE`, `A` is assumed to be Hermitian if complex or
#' symmetric if real, but this is not checked internally. Instead, just the lower
#' triangular part of the matrix is used in the computations.
#' The singular values (or the norm of the eigenvalues when `hermitian= TRUE`)
#' that are below the specified `rcond` threshold are treated as zero and discarded
#' in the computation.
#'
#' @note This function uses [linalg_svd()] if `hermitian= FALSE` and
#' [linalg_eigh()] if `hermitian= TRUE`.
#' For CUDA inputs, this function synchronizes that device with the CPU.
#'
#' @note
#' Consider using [linalg_lstsq()] if possible for multiplying a matrix on the left by
#' the pseudoinverse, as `linalg_lstsq(A, B)$solution == A$pinv() %*% B`
#'
#' It is always prefered to use [linalg_lstsq()] when possible, as it is faster and more
#' numerically stable than computing the pseudoinverse explicitly.
#'
#' @seealso
#' - [linalg_inv()] computes the inverse of a square matrix.
#' - [linalg_lstsq()] computes `A$pinv() %*% B` with a
#'   numerically stable algorithm.
#'
#' @param A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
#' @param rcond (float or Tensor, optional): the tolerance value to determine when is a singular value zero
#'   If it is a `torch_Tensor`, its shape must be
#'   broadcastable to that of the singular values of
#'   `A` as returned by [linalg_svd()].
#'   Default: `1e-15`.
#' @param hermitian (bool, optional): indicates whether `A` is Hermitian if complex
#'   or symmetric if real. Default: `FALSE`.
#'
#' @examples
#' A <- torch_randn(3, 5)
#' linalg_pinv(A)
#' @family linalg
#' @export
linalg_pinv <- function(A, rcond = 1e-15, hermitian = FALSE) {
  out <- torch_linalg_pinv(A, rcond = rcond, hermitian = hermitian)
  if (length(dim(out)) != length(dim(A))) {
    out <- out$squeeze(1)
  }
  out
}

#' Computes the `n`-th power of a square matrix for an integer `n`.
#'
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#'
#' If `n=0`, it returns the identity matrix (or batch) of the same shape
#' as `A`. If `n` is negative, it returns the inverse of each matrix
#' (if invertible) raised to the power of `abs(n)`.
#'
#' @seealso
#' [linalg_solve()] computes `A$inverse() %*% B` with a
#' numerically stable algorithm.
#'
#'
#' @param A (Tensor): tensor of shape `(*, m, m)` where `*` is zero or more batch dimensions.
#' @param n (int): the exponent.
#'
#' @examples
#' A <- torch_randn(3, 3)
#' linalg_matrix_power(A, 0)
#' @family linalg
#' @export
linalg_matrix_power <- function(A, n) {
  torch_linalg_matrix_power(A, n = n)
}

#' Efficiently multiplies two or more matrices
#'
#' Efficiently multiplies two or more matrices by reordering the multiplications so that
#' the fewest arithmetic operations are performed.
#'
#' Supports inputs of `float`, `double`, `cfloat` and `cdouble` dtypes.
#' This function does not support batched inputs.
#'
#' Every tensor in `tensors` must be 2D, except for the first and last which
#' may be 1D. If the first tensor is a 1D vector of shape `(n,)` it is treated as a row vector
#' of shape `(1, n)`, similarly if the last tensor is a 1D vector of shape `(n,)` it is treated
#' as a column vector of shape `(n, 1)`.
#'
#' If the first and last tensors are matrices, the output will be a matrix.
#' However, if either is a 1D vector, then the output will be a 1D vector.
#' @note This function is implemented by chaining [torch_mm()] calls after
#' computing the optimal matrix multiplication order.
#'
#' @note The cost of multiplying two matrices with shapes `(a, b)` and `(b, c)` is
#' `a * b * c`. Given matrices `A`, `B`, `C` with shapes `(10, 100)`,
#' `(100, 5)`, `(5, 50)` respectively, we can calculate the cost of different
#' multiplication orders as follows:
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#' \\\\begin{align*}
#' \\\\operatorname{cost}((AB)C) &= 10 \\\\times 100 \\\\times 5 + 10 \\\\times 5 \\\\times 50 = 7500 \\\\
#' \\\\operatorname{cost}(A(BC)) &= 10 \\\\times 100 \\\\times 50 + 100 \\\\times 5 \\\\times 50 = 75000
#' \\\\end{align*}
#' ")}
#'
#' In this case, multiplying `A` and `B` first followed by `C` is 10 times faster.
#'
#'
#' @param tensors (`Sequence[Tensor]`): two or more tensors to multiply. The first and last
#' tensors may be 1D or 2D. Every other tensor must be 2D.
#'
#' @examples
#'
#' linalg_multi_dot(list(torch_tensor(c(1, 2)), torch_tensor(c(2, 3))))
#' @family linalg
#' @export
linalg_multi_dot <- function(tensors) {
  torch_linalg_multi_dot(tensors)
}

#' Computes the first `n` columns of a product of Householder matrices.
#'
#' Letting \teqn{\mathbb{K}} be \teqn{\mathbb{R}} or \teqn{\mathbb{C}},
#' for a matrix \teqn{V \in \mathbb{K}^{m \times n}} with columns \teqn{v_i \in \mathbb{K}^m}
#' with \teqn{m \geq n} and a vector \teqn{\tau \in \mathbb{K}^k} with \teqn{k \leq n},
#' this function computes the first \teqn{n} columns of the matrix
#'
#' \Sexpr[results=rd, stage=build]{torch:::math_to_rd("
#' H_1H_2 ... H_k \\\\qquad with \\\\qquad H_i = \\\\mathrm{I}_m - \\\\tau_i v_i v_i^{H}
#' ")}
#'
#' where \teqn{\mathrm{I}_m} is the `m`-dimensional identity matrix and
#' \teqn{v^{H}} is the conjugate transpose when \teqn{v} is complex, and the transpose when \teqn{v} is real-valued.
#' See [Representation of Orthogonal or Unitary Matrices](https://www.netlib.org/lapack/lug/node128.html) for
#' further details.
#'
#' Supports inputs of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if the inputs are batches of matrices then
#' the output has the same batch dimensions.
#' @note This function only uses the values strictly below the main diagonal of `A`.
#' The other values are ignored.
#'
#' @seealso
#' - [torch_geqrf()] can be used together with this function to form the `Q` from the
#' [linalg_qr()] decomposition.
#'
#' - [torch_ormqr()] is a related function that computes the matrix multiplication
#' of a product of Householder matrices with another matrix.
#' However, that function is not supported by autograd.
#'
#' @param A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
#' @param tau (Tensor): tensor of shape `(*, k)` where `*` is zero or more batch dimensions.
#'
#' @examples
#' A <- torch_randn(2, 2)
#' h_tau <- torch_geqrf(A)
#' Q <- linalg_householder_product(h_tau[[1]], h_tau[[2]])
#' torch_allclose(Q, linalg_qr(A)[[1]])
#' @family linalg
#' @export
linalg_householder_product <- function(A, tau) {
  torch_linalg_householder_product(A, tau)
}

#' Computes the multiplicative inverse of [torch_tensordot()]
#'
#' If `m` is the product of the first `ind` dimensions of `A` and `n` is the product of
#' the rest of the dimensions, this function expects `m` and `n` to be equal.
#' If this is the case, it computes a tensor `X` such that
#' `tensordot(A, X, ind)` is the identity matrix in dimension `m`.
#'
#' Supports input of float, double, cfloat and cdouble dtypes.
#'
#' @note Consider using [linalg_tensorsolve()] if possible for multiplying a tensor on the left
#' by the tensor inverse as `linalg_tensorsolve(A, B) == torch_tensordot(linalg_tensorinv(A), B))`
#'
#' It is always prefered to use [linalg_tensorsolve()] when possible, as it is faster and more
#' numerically stable than computing the pseudoinverse explicitly.
#'
#' @seealso
#' - [linalg_tensorsolve()] computes `torch_tensordot(linalg_tensorinv(A), B))`.
#'
#' @param A (Tensor): tensor to invert.
#' @param ind (int): index at which to compute the inverse of [torch_tensordot()]. Default: `3`.
#'
#' @examples
#' A <- torch_eye(4 * 6)$reshape(c(4, 6, 8, 3))
#' Ainv <- linalg_tensorinv(A, ind = 3)
#' Ainv$shape
#' B <- torch_randn(4, 6)
#' torch_allclose(torch_tensordot(Ainv, B), linalg_tensorsolve(A, B))
#'
#' A <- torch_randn(4, 4)
#' Atensorinv <- linalg_tensorinv(A, 2)
#' Ainv <- linalg_inv(A)
#' torch_allclose(Atensorinv, Ainv)
#' @family linalg
#' @export
linalg_tensorinv <- function(A, ind = 3L) {
  torch_linalg_tensorinv(A, ind = ind - 1L)
}

#' Computes the solution `X` to the system `torch_tensordot(A, X) = B`.
#'
#' If `m` is the product of the first `B`\ `.ndim`  dimensions of `A` and
#' `n` is the product of the rest of the dimensions, this function expects `m` and `n` to be equal.
#' The returned tensor `x` satisfies
#' `tensordot(A, x, dims=x$ndim) == B`.
#'
#' If `dims` is specified, `A` will be reshaped as
#' `A = movedim(A, dims, seq(len(dims) - A$ndim + 1, 0))`
#'
#' Supports inputs of float, double, cfloat and cdouble dtypes.
#'
#' @seealso
#' - [linalg_tensorinv()] computes the multiplicative inverse of
#' [torch_tensordot()].
#'
#' @param A (Tensor): tensor to solve for.
#' @param B (Tensor): the solution
#' @param dims (Tuple[int], optional): dimensions of `A` to be moved.
#' If `NULL`, no dimensions are moved. Default: `NULL`.
#'
#' @examples
#' A <- torch_eye(2 * 3 * 4)$reshape(c(2 * 3, 4, 2, 3, 4))
#' B <- torch_randn(2 * 3, 4)
#' X <- linalg_tensorsolve(A, B)
#' X$shape
#' torch_allclose(torch_tensordot(A, X, dims = X$ndim), B)
#'
#' A <- torch_randn(6, 4, 4, 3, 2)
#' B <- torch_randn(4, 3, 2)
#' X <- linalg_tensorsolve(A, B, dims = c(1, 3))
#' A <- A$permute(c(2, 4, 5, 1, 3))
#' torch_allclose(torch_tensordot(A, X, dims = X$ndim), B, atol = 1e-6)
#' @family linalg
#' @export
linalg_tensorsolve <- function(A, B, dims = NULL) {
  torch_linalg_tensorsolve(A, B, dims)
}

#' Computes the Cholesky decomposition of a complex Hermitian or real
#' symmetric positive-definite matrix.
#'
#' This function skips the (slow) error checking and error message construction
#' of [linalg_cholesky()], instead directly returning the LAPACK
#' error codes as part of a named tuple `(L, info)`. This makes this function
#' a faster way to check if a matrix is positive-definite, and it provides an
#' opportunity to handle decomposition errors more gracefully or performantly
#' than [linalg_cholesky()] does.
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' If `A` is not a Hermitian positive-definite matrix, or if it's a batch of matrices
#' and one or more of them is not a Hermitian positive-definite matrix,
#' then `info` stores a positive integer for the corresponding matrix.
#' The positive integer indicates the order of the leading minor that is not positive-definite,
#' and the decomposition could not be completed.
#' `info` filled with zeros indicates that the decomposition was successful.
#' If `check_errors=TRUE` and `info` contains positive integers, then a RuntimeError is thrown.
#' @note If `A` is on a CUDA device, this function may synchronize that device with the CPU.
#' @note This function is "experimental" and it may change in a future PyTorch release.
#' @seealso
#' [linalg_cholesky()] is a NumPy compatible variant that always checks for errors.
#'
#' @param A (Tensor): the Hermitian `n \times n` matrix or the batch of such matrices of size
#'                     `(*, n, n)` where `*` is one or more batch dimensions.
#' @param check_errors (bool, optional): controls whether to check the content of `infos`. Default: `FALSE`.
#'
#' @examples
#' A <- torch_randn(2, 2)
#' out <- linalg_cholesky_ex(A)
#' out
#' @family linalg
#' @export
linalg_cholesky_ex <- function(A, check_errors = FALSE) {
  setNames(
    torch_linalg_cholesky_ex(A, check_errors = check_errors),
    c("L", "info")
  )
}

#' Computes the inverse of a square matrix if it is invertible.
#'
#' Returns a namedtuple `(inverse, info)`. `inverse` contains the result of
#' inverting `A` and `info` stores the LAPACK error codes.
#' If `A` is not an invertible matrix, or if it's a batch of matrices
#' and one or more of them is not an invertible matrix,
#' then `info` stores a positive integer for the corresponding matrix.
#' The positive integer indicates the diagonal element of the LU decomposition of
#' the input matrix that is exactly zero.
#' `info` filled with zeros indicates that the inversion was successful.
#' If `check_errors=TRUE` and `info` contains positive integers, then a RuntimeError is thrown.
#' Supports input of float, double, cfloat and cdouble dtypes.
#' Also supports batches of matrices, and if `A` is a batch of matrices then
#' the output has the same batch dimensions.
#' @note
#' If `A` is on a CUDA device then this function may synchronize
#' that device with the CPU.
#' @note This function is "experimental" and it may change in a future PyTorch release.
#'
#' @seealso
#' [linalg_inv()] is a NumPy compatible variant that always checks for errors.
#'
#' @param A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
#'                     consisting of square matrices.
#' @param check_errors (bool, optional): controls whether to check the content of `info`. Default: `FALSE`.
#'
#' @examples
#' A <- torch_randn(3, 3)
#' out <- linalg_inv_ex(A)
#' @family linalg
#' @importFrom stats setNames
#' @export
linalg_inv_ex <- function(A, check_errors = FALSE) {
  setNames(
    torch_linalg_inv_ex(A, check_errors = check_errors),
    c("inverse", "info")
  )
}
