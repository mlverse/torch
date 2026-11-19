# Dot

Dot

## Usage

``` r
torch_dot(self, tensor)
```

## Arguments

- self:

  the input tensor

- tensor:

  the other input tensor

## Note

This function does not broadcast .

## dot(input, tensor) -\> Tensor

Computes the dot product (inner product) of two tensors.

## Examples

``` r
if (torch_is_installed()) {

torch_dot(torch_tensor(c(2, 3)), torch_tensor(c(2, 1)))
}
#> torch_tensor
#> 7
#> [ CPUFloatType{} ]
```
