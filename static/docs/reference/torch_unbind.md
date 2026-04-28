# Unbind

Unbind

## Usage

``` r
torch_unbind(self, dim = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to unbind

- dim:

  (int) dimension to remove

## unbind(input, dim=0) -\> seq

Removes a tensor dimension.

Returns a tuple of all slices along a given dimension, already without
it.

## Examples

``` r
if (torch_is_installed()) {

torch_unbind(torch_tensor(matrix(1:9, ncol = 3, byrow=TRUE)))
}
#> [[1]]
#> torch_tensor
#>  1
#>  2
#>  3
#> [ CPULongType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  4
#>  5
#>  6
#> [ CPULongType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  7
#>  8
#>  9
#> [ CPULongType{3} ]
#> 
```
