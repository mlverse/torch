# Logical_not

Logical_not

## Arguments

- self:

  (Tensor) the input tensor.

## logical_not(input, out=NULL) -\> Tensor

Computes the element-wise logical NOT of the given input tensor. If not
specified, the output tensor will have the bool dtype. If the input
tensor is not a bool tensor, zeros are treated as `FALSE` and non-zeros
are treated as `TRUE`.

## Examples

``` r
if (torch_is_installed()) {

torch_logical_not(torch_tensor(c(TRUE, FALSE)))
torch_logical_not(torch_tensor(c(0, 1, -10), dtype=torch_int8()))
torch_logical_not(torch_tensor(c(0., 1.5, -10.), dtype=torch_double()))
}
#> torch_tensor
#>  1
#>  0
#>  0
#> [ CPUBoolType{3} ]
```
