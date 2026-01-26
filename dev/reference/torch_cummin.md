# Cummin

Cummin

## Usage

``` r
torch_cummin(self, dim)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to do the operation over

## cummin(input, dim) -\> (Tensor, LongTensor)

Returns a namedtuple `(values, indices)` where `values` is the
cumulative minimum of elements of `input` in the dimension `dim`. And
`indices` is the index location of each maximum value found in the
dimension `dim`.

\$\$ y_i = min(x_1, x_2, x_3, \dots, x_i) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(10))
a
torch_cummin(a, dim=1)
}
#> [[1]]
#> torch_tensor
#> 0.01 *
#>  9.2583
#>  9.2583
#> -49.1980
#> -49.1980
#> -110.6780
#> -110.6780
#> -186.5649
#> -186.5649
#> -186.5649
#> -186.5649
#> [ CPUFloatType{10} ]
#> 
#> [[2]]
#> torch_tensor
#>  0
#>  0
#>  2
#>  2
#>  4
#>  4
#>  6
#>  6
#>  6
#>  6
#> [ CPULongType{10} ]
#> 
```
