# Erfc

Erfc

## Usage

``` r
torch_erfc(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## erfc(input, out=NULL) -\> Tensor

Computes the complementary error function of each element of `input`.
The complementary error function is defined as follows:

\$\$ \mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int\_{0}^{x} e^{-t^2}
dt \$\$

## Examples

``` r
if (torch_is_installed()) {

torch_erfc(torch_tensor(c(0, -1., 10.)))
}
#> torch_tensor
#>  1.0000e+00
#>  1.8427e+00
#>  2.8026e-45
#> [ CPUFloatType{3} ]
```
