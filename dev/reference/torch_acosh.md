# Acosh

Acosh

## Usage

``` r
torch_acosh(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## Note

The domain of the inverse hyperbolic cosine is `[1, inf)` and values
outside this range will be mapped to `NaN`, except for `+ INF` for which
the output is mapped to `+ INF`.

\$\$ \mbox{out}\_{i} = \cosh^{-1}(\mbox{input}\_{i}) \$\$

## acosh(input, \*, out=None) -\> Tensor

Returns a new tensor with the inverse hyperbolic cosine of the elements
of `input`.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_randn(c(4))$uniform_(1, 2)
a
torch_acosh(a)
}
#> torch_tensor
#>  1.1400
#>  0.7713
#>  1.2416
#>  0.8963
#> [ CPUFloatType{4} ]
```
