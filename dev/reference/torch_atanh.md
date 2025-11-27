# Atanh

Atanh

## Usage

``` r
torch_atanh(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## Note

The domain of the inverse hyperbolic tangent is `(-1, 1)` and values
outside this range will be mapped to `NaN`, except for the values `1`
and `-1` for which the output is mapped to `+/-INF` respectively.

\$\$ \mbox{out}\_{i} = \tanh^{-1}(\mbox{input}\_{i}) \$\$

## atanh(input, \*, out=None) -\> Tensor

Returns a new tensor with the inverse hyperbolic tangent of the elements
of `input`.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))$uniform_(-1, 1)
a
torch_atanh(a)
}
#> torch_tensor
#>  0.4305
#>  0.9069
#>  1.0485
#> -0.7020
#> [ CPUFloatType{4} ]
```
