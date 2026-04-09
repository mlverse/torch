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
#> -1.6411
#> -0.4249
#> -2.6241
#> -3.4347
#> [ CPUFloatType{4} ]
```
