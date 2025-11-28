# Svd

Svd

## Usage

``` r
torch_svd(self, some = TRUE, compute_uv = TRUE)
```

## Arguments

- self:

  (Tensor) the input tensor of size \\(\*, m, n)\\ where `*` is zero or
  more batch dimensions consisting of \\m \times n\\ matrices.

- some:

  (bool, optional) controls the shape of returned `U` and `V`

- compute_uv:

  (bool, optional) option whether to compute `U` and `V` or not

## Note

The singular values are returned in descending order. If `input` is a
batch of matrices, then the singular values of each matrix in the batch
is returned in descending order.

The implementation of SVD on CPU uses the LAPACK routine `?gesdd` (a
divide-and-conquer algorithm) instead of `?gesvd` for speed.
Analogously, the SVD on GPU uses the MAGMA routine `gesdd` as well.

Irrespective of the original strides, the returned matrix `U` will be
transposed, i.e. with strides
`U.contiguous().transpose(-2, -1).stride()`

Extra care needs to be taken when backward through `U` and `V` outputs.
Such operation is really only stable when `input` is full rank with all
distinct singular values. Otherwise, `NaN` can appear as the gradients
are not properly defined. Also, notice that double backward will usually
do an additional backward through `U` and `V` even if the original
backward is only on `S`.

When `some` = `FALSE`, the gradients on `U[..., :, min(m, n):]` and
`V[..., :, min(m, n):]` will be ignored in backward as those vectors can
be arbitrary bases of the subspaces.

When `compute_uv` = `FALSE`, backward cannot be performed since `U` and
`V` from the forward pass is required for the backward operation.

## svd(input, some=TRUE, compute_uv=TRUE) -\> (Tensor, Tensor, Tensor)

This function returns a namedtuple `(U, S, V)` which is the singular
value decomposition of a input real matrix or batches of real matrices
`input` such that \\input = U \times diag(S) \times V^T\\.

If `some` is `TRUE` (default), the method returns the reduced singular
value decomposition i.e., if the last two dimensions of `input` are `m`
and `n`, then the returned `U` and `V` matrices will contain only
\\min(n, m)\\ orthonormal columns.

If `compute_uv` is `FALSE`, the returned `U` and `V` matrices will be
zero matrices of shape \\(m \times m)\\ and \\(n \times n)\\
respectively. `some` will be ignored here.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(5, 3))
a
out = torch_svd(a)
u = out[[1]]
s = out[[2]]
v = out[[3]]
torch_dist(a, torch_mm(torch_mm(u, torch_diag(s)), v$t()))
a_big = torch_randn(c(7, 5, 3))
out = torch_svd(a_big)
u = out[[1]]
s = out[[2]]
v = out[[3]]
torch_dist(a_big, torch_matmul(torch_matmul(u, torch_diag_embed(s)), v$transpose(-2, -1)))
}
#> torch_tensor
#> 2.35912e-06
#> [ CPUFloatType{} ]
```
