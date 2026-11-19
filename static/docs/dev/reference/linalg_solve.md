# Computes the solution of a square system of linear equations with a unique solution.

Letting be or , this function computes the solution of the **linear
system** associated to , which is defined as

## Usage

``` r
linalg_solve(A, B)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch
  dimensions.

- B:

  (Tensor): right-hand side tensor of shape `(*, n)` or `(*, n, k)` or
  `(n,)` or `(n, k)` according to the rules described above

## Details

\$\$ AX = B \$\$

This system of linear equations has one solution if and only if is
`invertible`\_. This function assumes that is invertible. Supports
inputs of float, double, cfloat and cdouble dtypes. Also supports
batches of matrices, and if the inputs are batches of matrices then the
output has the same batch dimensions.

Letting `*` be zero or more batch dimensions,

- If `A` has shape `(*, n, n)` and `B` has shape `(*, n)` (a batch of
  vectors) or shape `(*, n, k)` (a batch of matrices or "multiple
  right-hand sides"), this function returns `X` of shape `(*, n)` or
  `(*, n, k)` respectively.

- Otherwise, if `A` has shape `(*, n, n)` and `B` has shape `(n,)` or
  `(n, k)`, `B` is broadcasted to have shape `(*, n)` or `(*, n, k)`
  respectively.

This function then returns the solution of the resulting batch of
systems of linear equations.

## Note

This function computes `X = A$inverse() @ B` in a faster and more
numerically stable way than performing the computations separately.

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
[`linalg_matrix_rank()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_rank.md),
[`linalg_multi_dot()`](https://torch.mlverse.org/docs/dev/reference/linalg_multi_dot.md),
[`linalg_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_norm.md),
[`linalg_pinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_pinv.md),
[`linalg_qr()`](https://torch.mlverse.org/docs/dev/reference/linalg_qr.md),
[`linalg_slogdet()`](https://torch.mlverse.org/docs/dev/reference/linalg_slogdet.md),
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
b <- torch_randn(3)
x <- linalg_solve(A, b)
torch_allclose(torch_matmul(A, x), b)
}
#> [1] TRUE
```
