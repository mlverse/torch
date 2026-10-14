# Transpose

Transpose

## Usage

``` r
torch_transpose(self, dim0, dim1)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim0:

  (int) the first dimension to be transposed

- dim1:

  (int) the second dimension to be transposed

## transpose(input, dim0, dim1) -\> Tensor

Returns a tensor that is a transposed version of `input`. The given
dimensions `dim0` and `dim1` are swapped.

The resulting `out` tensor shares it's underlying storage with the
`input` tensor, so changing the content of one would change the content
of the other.

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(2, 3))
x
torch_transpose(x, 1, 2)
}
#> torch_tensor
#>  0.4465 -1.0808
#> -0.9955 -1.3864
#> -0.3668 -0.1908
#> [ CPUFloatType{3,2} ]
```
