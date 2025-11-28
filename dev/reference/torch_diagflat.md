# Diagflat

Diagflat

## Usage

``` r
torch_diagflat(self, offset = 0L)
```

## Arguments

- self:

  (Tensor) the input tensor.

- offset:

  (int, optional) the diagonal to consider. Default: 0 (main diagonal).

## diagflat(input, offset=0) -\> Tensor

- If `input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of `input` as the diagonal.

- If `input` is a tensor with more than one dimension, then returns a
  2-D tensor with diagonal elements equal to a flattened `input`.

The argument `offset` controls which diagonal to consider:

- If `offset` = 0, it is the main diagonal.

- If `offset` \> 0, it is above the main diagonal.

- If `offset` \< 0, it is below the main diagonal.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3))
a
torch_diagflat(a)
torch_diagflat(a, 1)
a = torch_randn(c(2, 2))
a
torch_diagflat(a)
}
#> torch_tensor
#>  0.3606  0.0000  0.0000  0.0000
#>  0.0000  0.7641  0.0000  0.0000
#>  0.0000  0.0000 -0.1729  0.0000
#>  0.0000  0.0000  0.0000  0.3655
#> [ CPUFloatType{4,4} ]
```
