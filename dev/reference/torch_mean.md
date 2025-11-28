# Mean

Mean

## Usage

``` r
torch_mean(self, dim, keepdim = FALSE, dtype = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int or tuple of ints) the dimension or dimensions to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

- dtype:

  the resulting data type.

## mean(input) -\> Tensor

Returns the mean value of all elements in the `input` tensor.

## mean(input, dim, keepdim=False, out=NULL) -\> Tensor

Returns the mean value of each row of the `input` tensor in the given
dimension `dim`. If `dim` is a list of dimensions, reduce over all of
them.

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim`
is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensor having 1 (or `len(dim)`) fewer
dimension(s).

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_mean(a)


a = torch_randn(c(4, 4))
a
torch_mean(a, 1)
torch_mean(a, 1, TRUE)
}
#> torch_tensor
#> -0.2449  0.6263 -0.4619  0.8996
#> [ CPUFloatType{1,4} ]
```
