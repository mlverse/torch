# Computes the condition number of a matrix with respect to a matrix norm.

Letting be or , the **condition number** of a matrix is defined as

## Usage

``` r
linalg_cond(A, p = NULL)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch
  dimensions for `p` in `(2, -2)`, and of shape `(*, n, n)` where every
  matrix is invertible for `p` in `('fro', 'nuc', inf, -inf, 1, -1)`.

- p:

  (int, inf, -inf, 'fro', 'nuc', optional): the type of the matrix norm
  to use in the computations (see above). Default: `NULL`

## Value

A real-valued tensor, even when `A` is complex.

## Details

$$\kappa(A) = \parallel A\parallel_{p}\parallel A^{- 1}\parallel_{p}$$κ(A)=∥A∥p​∥A−1∥p​

The condition number of `A` measures the numerical stability of the
linear system `AX = B` with respect to a matrix norm.

Supports input of float, double, cfloat and cdouble dtypes. Also
supports batches of matrices, and if `A` is a batch of matrices then the
output has the same batch dimensions.

`p` defines the matrix norm that is computed. See the table in 'Details'
to find the supported norms.

For `p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`, this function uses
[`linalg_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_norm.md)
and
[`linalg_inv()`](https://torch.mlverse.org/docs/dev/reference/linalg_inv.md).

As such, in this case, the matrix (or every matrix in the batch) `A` has
to be square and invertible.

For `p` in `(2, -2)`, this function can be computed in terms of the
singular values

$$\kappa_{2}(A) = \frac{\sigma_{1}}{\sigma_{n}}\qquad\kappa_{- 2}(A) = \frac{\sigma_{n}}{\sigma_{1}}$$κ2​(A)=σn​σ1​​κ−2​(A)=σ1​σn​​

In these cases, it is computed using
[`linalg_svd()`](https://torch.mlverse.org/docs/dev/reference/linalg_svd.md).
For these norms, the matrix (or every matrix in the batch) `A` may have
any shape.

|         |                                   |
|---------|-----------------------------------|
| `p`     | matrix norm                       |
| `NULL`  | `2`-norm (largest singular value) |
| `'fro'` | Frobenius norm                    |
| `'nuc'` | nuclear norm                      |
| `Inf`   | `max(sum(abs(x), dim=2))`         |
| `-Inf`  | `min(sum(abs(x), dim=2))`         |
| `1`     | `max(sum(abs(x), dim=1))`         |
| `-1`    | `min(sum(abs(x), dim=1))`         |
| `2`     | largest singular value            |
| `-2`    | smallest singular value           |

## Note

When inputs are on a CUDA device, this function synchronizes that device
with the CPU if if `p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`.

## Examples

``` r
if (torch_is_installed()) {
a <- torch_tensor(rbind(c(1., 0, -1), c(0, 1, 0), c(1, 0, 1)))
linalg_cond(a)
linalg_cond(a, "fro")
}
#> torch_tensor
#> 3.16228
#> [ CPUFloatType{} ]
```
