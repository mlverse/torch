# Erf

Erf

## Usage

``` r
torch_erf(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## erf(input, out=NULL) -\> Tensor

Computes the error function of each element. The error function is
defined as follows:

\$\$ \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int\_{0}^{x} e^{-t^2} dt
\$\$

## Examples

``` r
if (torch_is_installed()) {

torch_erf(torch_tensor(c(0, -1., 10.)))
}
#> torch_tensor
#>  0.0000
#> -0.8427
#>  1.0000
#> [ CPUFloatType{3} ]
```
