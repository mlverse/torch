# Computes the first `n` columns of a product of Householder matrices.

Letting be or , for a matrix with columns with and a vector with , this
function computes the first columns of the matrix

## Usage

``` r
linalg_householder_product(A, tau)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch
  dimensions.

- tau:

  (Tensor): tensor of shape `(*, k)` where `*` is zero or more batch
  dimensions.

## Details

$$H_{1}H_{2}...H_{k}\qquad with\qquad H_{i} = I_{m} - \tau_{i}v_{i}v_{i}^{H}$$H1​H2​...Hk​withHi​=Im​−τi​vi​viH​

where is the `m`-dimensional identity matrix and is the conjugate
transpose when is complex, and the transpose when is real-valued. See
[Representation of Orthogonal or Unitary
Matrices](https://netlib.org/lapack/lug/node128.html) for further
details.

Supports inputs of float, double, cfloat and cdouble dtypes. Also
supports batches of matrices, and if the inputs are batches of matrices
then the output has the same batch dimensions.

## Note

This function only uses the values strictly below the main diagonal of
`A`. The other values are ignored.

## See also

- [`torch_geqrf()`](https://torch.mlverse.org/docs/dev/reference/torch_geqrf.md)
  can be used together with this function to form the `Q` from the
  [`linalg_qr()`](https://torch.mlverse.org/docs/dev/reference/linalg_qr.md)
  decomposition.

- [`torch_ormqr()`](https://torch.mlverse.org/docs/dev/reference/torch_ormqr.md)
  is a related function that computes the matrix multiplication of a
  product of Householder matrices with another matrix. However, that
  function is not supported by autograd.

Other linalg:
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
[`linalg_cholesky_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky_ex.md),
[`linalg_det()`](https://torch.mlverse.org/docs/dev/reference/linalg_det.md),
[`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md),
[`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md),
[`linalg_eigvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvals.md),
[`linalg_eigvalsh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvalsh.md),
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
A <- torch_randn(2, 2)
h_tau <- torch_geqrf(A)
Q <- linalg_householder_product(h_tau[[1]], h_tau[[2]])
torch_allclose(Q, linalg_qr(A)[[1]])
}
#> [1] TRUE
```
