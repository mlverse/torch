# Computes the `n`-th power of a square matrix for an integer `n`.

Supports input of float, double, cfloat and cdouble dtypes. Also
supports batches of matrices, and if `A` is a batch of matrices then the
output has the same batch dimensions.

## Usage

``` r
linalg_matrix_power(A, n)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, m, m)` where `*` is zero or more batch
  dimensions.

- n:

  (int): the exponent.

## Details

If `n=0`, it returns the identity matrix (or batch) of the same shape as
`A`. If `n` is negative, it returns the inverse of each matrix (if
invertible) raised to the power of `abs(n)`.

## See also

[`linalg_solve()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve.md)
computes `A$inverse() %*% B` with a numerically stable algorithm.

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
[`linalg_matrix_rank()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_rank.md),
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
A <- torch_randn(3, 3)
linalg_matrix_power(A, 0)
}
#> torch_tensor
#>  1  0  0
#>  0  1  0
#>  0  0  1
#> [ CPUFloatType{3,3} ]
```
