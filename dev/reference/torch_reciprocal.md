# Reciprocal

Reciprocal

## Usage

``` r
torch_reciprocal(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## reciprocal(input, out=NULL) -\> Tensor

Returns a new tensor with the reciprocal of the elements of `input`

\$\$ \mbox{out}\_{i} = \frac{1}{\mbox{input}\_{i}} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_reciprocal(a)
}
#> torch_tensor
#>  8.7440
#>  0.7127
#> -6.6193
#>  0.6483
#> [ CPUFloatType{4} ]
```
