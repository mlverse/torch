# Median

Median

## Usage

``` r
torch_median(self, dim, keepdim = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

## median(input) -\> Tensor

Returns the median value of all elements in the `input` tensor.

## median(input, dim=-1, keepdim=False, out=NULL) -\> (Tensor, LongTensor)

Returns a namedtuple `(values, indices)` where `values` is the median
value of each row of the `input` tensor in the given dimension `dim`.
And `indices` is the index location of each median value found.

By default, `dim` is the last dimension of the `input` tensor.

If `keepdim` is `TRUE`, the output tensors are of the same size as
`input` except in the dimension `dim` where they are of size 1.
Otherwise, `dim` is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the outputs tensor having 1 fewer dimension than `input`.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_median(a)


a = torch_randn(c(4, 5))
a
torch_median(a, 1)
}
#> [[1]]
#> torch_tensor
#>  0.4414
#> -0.2265
#> -1.8186
#>  0.4766
#> -0.3928
#> [ CPUFloatType{5} ]
#> 
#> [[2]]
#> torch_tensor
#>  1
#>  2
#>  1
#>  2
#>  3
#> [ CPULongType{5} ]
#> 
```
