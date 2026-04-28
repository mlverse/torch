# Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

Letting be or , the **Cholesky decomposition** of a complex Hermitian or
real symmetric positive-definite matrix is defined as

## Usage

``` r
linalg_cholesky(A)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch
  dimensions consisting of symmetric or Hermitian positive-definite
  matrices.

## Details

$$A = LL^{H}{\qquad L \in \mathbb{K}^{n \times n}}$$A=LLHL∈Kn×n

where is a lower triangular matrix and is the conjugate transpose when
is complex, and the transpose when is real-valued.

Supports input of float, double, cfloat and cdouble dtypes. Also
supports batches of matrices, and if `A` is a batch of matrices then the
output has the same batch dimensions.

## See also

- [`linalg_cholesky_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky_ex.md)
  for a version of this operation that skips the (slow) error checking
  by default and instead returns the debug information. This makes it a
  faster way to check if a matrix is positive-definite.
  [`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md)
  for a different decomposition of a Hermitian matrix. The eigenvalue
  decomposition gives more information about the matrix but it slower to
  compute than the Cholesky decomposition.

Other linalg:
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
[`linalg_tensorinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorinv.md),
[`linalg_tensorsolve()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorsolve.md),
[`linalg_vector_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_vector_norm.md)

## Examples

``` r
if (torch_is_installed()) {
a <- torch_eye(10)
linalg_cholesky(a)
}
#> torch_tensor
#>  1  0  0  0  0  0  0  0  0  0
#>  0  1  0  0  0  0  0  0  0  0
#>  0  0  1  0  0  0  0  0  0  0
#>  0  0  0  1  0  0  0  0  0  0
#>  0  0  0  0  1  0  0  0  0  0
#>  0  0  0  0  0  1  0  0  0  0
#>  0  0  0  0  0  0  1  0  0  0
#>  0  0  0  0  0  0  0  1  0  0
#>  0  0  0  0  0  0  0  0  1  0
#>  0  0  0  0  0  0  0  0  0  1
#> [ CPUFloatType{10,10} ]
```
