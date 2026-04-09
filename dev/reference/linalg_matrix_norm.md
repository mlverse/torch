# Computes a matrix norm.

If `A` is complex valued, it computes the norm of `A$abs()` Support
input of float, double, cfloat and cdouble dtypes. Also supports batches
of matrices: the norm will be computed over the dimensions specified by
the 2-tuple `dim` and the other dimensions will be treated as batch
dimensions. The output will have the same batch dimensions.

## Usage

``` r
linalg_matrix_norm(
  A,
  ord = "fro",
  dim = c(-2, -1),
  keepdim = FALSE,
  dtype = NULL
)
```

## Arguments

- A:

  (Tensor): tensor with two or more dimensions. By default its shape is
  interpreted as `(*, m, n)` where `*` is zero or more batch dimensions,
  but this behavior can be controlled using `dim`.

- ord:

  (int, inf, -inf, 'fro', 'nuc', optional): order of norm. Default:
  `'fro'`

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
a <- torch_arange(0, 8, dtype = torch_float())$reshape(c(3, 3))
linalg_matrix_norm(a)
linalg_matrix_norm(a, ord = -1)
b <- a$expand(c(2, -1, -1))
linalg_matrix_norm(b)
linalg_matrix_norm(b, dim = c(1, 3))
}
#> torch_tensor
#>   3.1623
#>  10.0000
#>  17.2627
#> [ CPUFloatType{3} ]
```
