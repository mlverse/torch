# Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.

The pseudoinverse may be `defined algebraically`\_ but it is more
computationally convenient to understand it `through the SVD`\_ Supports
input of float, double, cfloat and cdouble dtypes. Also supports batches
of matrices, and if `A` is a batch of matrices then the output has the
same batch dimensions.

## Usage

``` r
linalg_pinv(A, rcond = NULL, hermitian = FALSE, atol = NULL, rtol = NULL)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch
  dimensions.

- rcond:

  (float or Tensor, optional): the tolerance value to determine when is
  a singular value zero If it is a `torch_Tensor`, its shape must be
  broadcastable to that of the singular values of `A` as returned by
  [`linalg_svd()`](https://torch.mlverse.org/docs/dev/reference/linalg_svd.md).
  Alias for `rtol`. Default: `0`.

- hermitian:

  (bool, optional): indicates whether `A` is Hermitian if complex or
  symmetric if real. Default: `FALSE`.

- atol:

  the absolute tolerance value. When `NULL` it’s considered to be zero.

- rtol:

  the relative tolerance value. See above for the value it takes when
  `NULL`.

## Details

If `hermitian= TRUE`, `A` is assumed to be Hermitian if complex or
symmetric if real, but this is not checked internally. Instead, just the
lower triangular part of the matrix is used in the computations. The
singular values (or the norm of the eigenvalues when `hermitian= TRUE`)
that are below the specified `rcond` threshold are treated as zero and
discarded in the computation.

## Note

This function uses
[`linalg_svd()`](https://torch.mlverse.org/docs/dev/reference/linalg_svd.md)
if `hermitian= FALSE` and
[`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md)
if `hermitian= TRUE`. For CUDA inputs, this function synchronizes that
device with the CPU.

Consider using
[`linalg_lstsq()`](https://torch.mlverse.org/docs/dev/reference/linalg_lstsq.md)
if possible for multiplying a matrix on the left by the pseudoinverse,
as `linalg_lstsq(A, B)$solution == A$pinv() %*% B`

It is always prefered to use
[`linalg_lstsq()`](https://torch.mlverse.org/docs/dev/reference/linalg_lstsq.md)
when possible, as it is faster and more numerically stable than
computing the pseudoinverse explicitly.

## See also

- [`linalg_inv()`](https://torch.mlverse.org/docs/dev/reference/linalg_inv.md)
  computes the inverse of a square matrix.

- [`linalg_lstsq()`](https://torch.mlverse.org/docs/dev/reference/linalg_lstsq.md)
  computes `A$pinv() %*% B` with a numerically stable algorithm.

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
[`linalg_matrix_rank()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_rank.md),
[`linalg_multi_dot()`](https://torch.mlverse.org/docs/dev/reference/linalg_multi_dot.md),
[`linalg_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_norm.md),
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
A <- torch_randn(3, 5)
linalg_pinv(A)
}
#> torch_tensor
#>  0.0517 -0.4091 -0.1494
#> -0.5276  0.3803 -0.2363
#> -0.0258  0.2677  0.2618
#>  0.5259  0.3422 -0.0605
#>  0.2356 -0.3627  0.1077
#> [ CPUFloatType{5,3} ]
```
