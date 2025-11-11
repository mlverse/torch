# Computes the multiplicative inverse of [`torch_tensordot()`](https://torch.mlverse.org/docs/dev/reference/torch_tensordot.md)

If `m` is the product of the first `ind` dimensions of `A` and `n` is
the product of the rest of the dimensions, this function expects `m` and
`n` to be equal. If this is the case, it computes a tensor `X` such that
`tensordot(A, X, ind)` is the identity matrix in dimension `m`.

## Usage

``` r
linalg_tensorinv(A, ind = 3L)
```

## Arguments

- A:

  (Tensor): tensor to invert.

- ind:

  (int): index at which to compute the inverse of
  [`torch_tensordot()`](https://torch.mlverse.org/docs/dev/reference/torch_tensordot.md).
  Default: `3`.

## Details

Supports input of float, double, cfloat and cdouble dtypes.

## Note

Consider using
[`linalg_tensorsolve()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorsolve.md)
if possible for multiplying a tensor on the left by the tensor inverse
as
`linalg_tensorsolve(A, B) == torch_tensordot(linalg_tensorinv(A), B))`

It is always prefered to use
[`linalg_tensorsolve()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorsolve.md)
when possible, as it is faster and more numerically stable than
computing the pseudoinverse explicitly.

## See also

- [`linalg_tensorsolve()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorsolve.md)
  computes `torch_tensordot(linalg_tensorinv(A), B))`.

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
[`linalg_pinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_pinv.md),
[`linalg_qr()`](https://torch.mlverse.org/docs/dev/reference/linalg_qr.md),
[`linalg_slogdet()`](https://torch.mlverse.org/docs/dev/reference/linalg_slogdet.md),
[`linalg_solve()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve.md),
[`linalg_solve_triangular()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve_triangular.md),
[`linalg_svd()`](https://torch.mlverse.org/docs/dev/reference/linalg_svd.md),
[`linalg_svdvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_svdvals.md),
[`linalg_tensorsolve()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorsolve.md),
[`linalg_vector_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_vector_norm.md)

## Examples

``` r
if (torch_is_installed()) {
A <- torch_eye(4 * 6)$reshape(c(4, 6, 8, 3))
Ainv <- linalg_tensorinv(A, ind = 3)
Ainv$shape
B <- torch_randn(4, 6)
torch_allclose(torch_tensordot(Ainv, B), linalg_tensorsolve(A, B))

A <- torch_randn(4, 4)
Atensorinv <- linalg_tensorinv(A, 2)
Ainv <- linalg_inv(A)
torch_allclose(Atensorinv, Ainv)
}
#> [1] TRUE
```
