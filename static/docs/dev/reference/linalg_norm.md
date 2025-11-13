# Computes a vector or matrix norm.

If `A` is complex valued, it computes the norm of `A$abs()` Supports
input of float, double, cfloat and cdouble dtypes. Whether this function
computes a vector or matrix norm is determined as follows:

## Usage

``` r
linalg_norm(A, ord = NULL, dim = NULL, keepdim = FALSE, dtype = NULL)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, n)` or `(*, m, n)` where `*` is zero or
  more batch dimensions

- ord:

  (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm.
  Default: `NULL`

- dim:

  (int, `Tuple[int]`, optional): dimensions over which to compute the
  vector or matrix norm. See above for the behavior when `dim=NULL`.
  Default: `NULL`

- keepdim:

  (bool, optional): If set to `TRUE`, the reduced dimensions are
  retained in the result as dimensions with size one. Default: `FALSE`

- dtype:

  dtype (`torch_dtype`, optional): If specified, the input tensor is
  cast to `dtype` before performing the operation, and the returned
  tensor's type will be `dtype`. Default: `NULL`

## Details

- If `dim` is an int, the vector norm will be computed.

- If `dim` is a 2-tuple, the matrix norm will be computed.

- If `dim=NULL` and `ord=NULL`, A will be flattened to 1D and the 2-norm
  of the resulting vector will be computed.

- If `dim=NULL` and `ord!=NULL`, A must be 1D or 2D.

`ord` defines the norm that is computed. The following norms are
supported:

|                        |                           |                                 |
|------------------------|---------------------------|---------------------------------|
| `ord`                  | norm for matrices         | norm for vectors                |
| `NULL` (default)       | Frobenius norm            | `2`-norm (see below)            |
| `"fro"`                | Frobenius norm            | – not supported –               |
| `"nuc"`                | nuclear norm              | – not supported –               |
| `Inf`                  | `max(sum(abs(x), dim=2))` | `max(abs(x))`                   |
| `-Inf`                 | `min(sum(abs(x), dim=2))` | `min(abs(x))`                   |
| `0`                    | – not supported –         | `sum(x != 0)`                   |
| `1`                    | `max(sum(abs(x), dim=1))` | as below                        |
| `-1`                   | `min(sum(abs(x), dim=1))` | as below                        |
| `2`                    | largest singular value    | as below                        |
| `-2`                   | smallest singular value   | as below                        |
| other `int` or `float` | – not supported –         | `sum(abs(x)^{ord})^{(1 / ord)}` |

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
a <- torch_arange(0, 8, dtype = torch_float()) - 4
a
b <- a$reshape(c(3, 3))
b

linalg_norm(a)
linalg_norm(b)
}
#> torch_tensor
#> 7.74597
#> [ CPUFloatType{} ]
```
