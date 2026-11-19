# Eye initialization

Fills the 2-dimensional input `Tensor` with the identity matrix.
Preserves the identity of the inputs in `Linear` layers, where as many
inputs are preserved as possible.

## Usage

``` r
nn_init_eye_(tensor)
```

## Arguments

- tensor:

  a 2-dimensional torch tensor.

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_eye_(w)
}
#> torch_tensor
#>  1  0  0  0  0
#>  0  1  0  0  0
#>  0  0  1  0  0
#> [ CPUFloatType{3,5} ]
```
