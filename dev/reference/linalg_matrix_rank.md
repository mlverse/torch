# Computes the numerical rank of a matrix.

The matrix rank is computed as the number of singular values (or
eigenvalues in absolute value when `hermitian = TRUE`) that are greater
than the specified `tol` threshold.

## Usage

``` r
linalg_matrix_rank(
  A,
  ...,
  atol = NULL,
  rtol = NULL,
  tol = NULL,
  hermitian = FALSE
)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch
  dimensions.

- ...:

  Not currently used.

- atol:

  the absolute tolerance value. When `NULL` it’s considered to be zero.

- rtol:

  the relative tolerance value. See above for the value it takes when
  `NULL`.

- tol:

  (float, Tensor, optional): the tolerance value. See above for the
  value it takes when `NULL`. Default: `NULL`.

- hermitian:

  (bool, optional): indicates whether `A` is Hermitian if complex or
  symmetric if real. Default: `FALSE`.

## Details

Supports input of float, double, cfloat and cdouble dtypes. Also
supports batches of matrices, and if `A` is a batch of matrices then the
output has the same batch dimensions.

If `hermitian = TRUE`, `A` is assumed to be Hermitian if complex or
symmetric if real, but this is not checked internally. Instead, just the
lower triangular part of the matrix is used in the computations.

If `tol` is not specified and `A` is a matrix of dimensions `(m, n)`,
the tolerance is set to be

$$tol = \sigma_{1}\max(m,n)\varepsilon$$tol=σ1​max(m,n)ε

where is the largest singular value (or eigenvalue in absolute value
when `hermitian = TRUE`), and is the epsilon value for the dtype of `A`
(see
[`torch_finfo()`](https://torch.mlverse.org/docs/dev/reference/torch_finfo.md)).

If `A` is a batch of matrices, `tol` is computed this way for every
element of the batch.

## See also

Other linalg:
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
[`linalg_cholesky_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky_ex.md),
[`linalg_det()`](https://torch.mlverse.org/docs/dev/reference/linalg_det.md),
[`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md),
[`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md),
[`linalg_eigvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvals.md),
[`linalg_eigvalsh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvalsh.md),
[`linalg_householder_product()`](https://torch.mlverse.org/docs/dev/reference/linalg_householder_product.md),
[`linalg_inv()`](https://torch.mlverse.org/docs/dev/reference/linalg_inv.md),
[`linalg_inv_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_inv_ex.md),
[`linalg_lstsq()`](https://torch.mlverse.org/docs/dev/reference/linalg_lstsq.md),
[`linalg_matrix_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_norm.md),
[`linalg_matrix_power()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_power.md),
[`linalg_multi_dot()`](https://torch.mlverse.org/docs/dev/reference/linalg_multi_dot.md),
[`linalg_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_norm.md),
[`linalg_pinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_pinv.md),
[`linalg_qr()`](https://torch.mlverse.org/docs/dev/reference/linalg_qr.md),
[`linalg_slogdet()`](https://torch.mlverse.org/docs/dev/reference/linalg_slogdet.md),
[`linalg_solve()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve.md),
[`linalg_solve_triangular()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve_triangular.md),
[`linalg_svd()`](https://torch.mlverse.org/docs/dev/reference/linalg_svd.md),
[`linalg_svdvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_svdvals.md),
[`linalg_tensorinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorinv.md),
[`linalg_tensorsolve()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorsolve.md),
[`linalg_vector_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_vector_norm.md)

## Examples

``` r
if (torch_is_installed()) {
a <- torch_eye(10)
linalg_matrix_rank(a)
}
#> torch_tensor
#> 10
#> [ CPULongType{} ]
```
