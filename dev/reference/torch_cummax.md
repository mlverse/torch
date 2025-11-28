# Cummax

Cummax

## Usage

``` r
torch_cummax(self, dim)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to do the operation over

## cummax(input, dim) -\> (Tensor, LongTensor)

Returns a namedtuple `(values, indices)` where `values` is the
cumulative maximum of elements of `input` in the dimension `dim`. And
`indices` is the index location of each maximum value found in the
dimension `dim`.

\$\$ y_i = max(x_1, x_2, x_3, \dots, x_i) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(10))
a
torch_cummax(a, dim=1)
}
#> [[1]]
#> torch_tensor
#>  0.7606
#>  0.7606
#>  0.7606
#>  0.7606
#>  0.7606
#>  1.5072
#>  1.5072
#>  1.5072
#>  1.5072
#>  1.5072
#> [ CPUFloatType{10} ]
#> 
#> [[2]]
#> torch_tensor
#>  0
#>  0
#>  0
#>  0
#>  0
#>  5
#>  5
#>  5
#>  5
#>  5
#> [ CPULongType{10} ]
#> 
```
