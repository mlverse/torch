# Eq

Eq

## Usage

``` r
torch_eq(self, other)
```

## Arguments

- self:

  (Tensor) the tensor to compare

- other:

  (Tensor or float) the tensor or value to compare Must be a
  `ByteTensor`

## eq(input, other, out=NULL) -\> Tensor

Computes element-wise equality

The second argument can be a number or a tensor whose shape is
broadcastable with the first argument.

## Examples

``` r
if (torch_is_installed()) {

torch_eq(torch_tensor(c(1,2,3,4)), torch_tensor(c(1, 3, 2, 4)))
}
#> torch_tensor
#>  1
#>  0
#>  0
#>  1
#> [ CPUBoolType{4} ]
```
