# Reshape

Reshape

## Usage

``` r
torch_reshape(self, shape)
```

## Arguments

- self:

  (Tensor) the tensor to be reshaped

- shape:

  (tuple of ints) the new shape

## reshape(input, shape) -\> Tensor

Returns a tensor with the same data and number of elements as `input`,
but with the specified shape. When possible, the returned tensor will be
a view of `input`. Otherwise, it will be a copy. Contiguous inputs and
inputs with compatible strides can be reshaped without copying, but you
should not depend on the copying vs. viewing behavior.

See `torch_Tensor.view` on when it is possible to return a view.

A single dimension may be -1, in which case it's inferred from the
remaining dimensions and the number of elements in `input`.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_arange(0, 3)
torch_reshape(a, list(2, 2))
b <- torch_tensor(matrix(c(0, 1, 2, 3), ncol = 2, byrow=TRUE))
torch_reshape(b, list(-1))
}
#> torch_tensor
#>  0
#>  1
#>  2
#>  3
#> [ CPUFloatType{4} ]
```
