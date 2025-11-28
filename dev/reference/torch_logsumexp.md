# Logsumexp

Logsumexp

## Usage

``` r
torch_logsumexp(self, dim, keepdim = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int or tuple of ints) the dimension or dimensions to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

## logsumexp(input, dim, keepdim=False, out=NULL)

Returns the log of summed exponentials of each row of the `input` tensor
in the given dimension `dim`. The computation is numerically stabilized.

For summation index \\j\\ given by `dim` and other indices \\i\\, the
result is

\$\$ \mbox{logsumexp}(x)\_{i} = \log \sum_j \exp(x\_{ij}) \$\$

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim`
is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensor having 1 (or `len(dim)`) fewer
dimension(s).

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
torch_logsumexp(a, 1)
}
#> torch_tensor
#>  1.0213
#>  0.9566
#>  1.3648
#> [ CPUFloatType{3} ]
```
