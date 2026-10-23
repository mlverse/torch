# Erfinv

Erfinv

## Usage

``` r
torch_erfinv(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## erfinv(input, out=NULL) -\> Tensor

Computes the inverse error function of each element of `input`. The
inverse error function is defined in the range \\(-1, 1)\\ as:

\$\$ \mathrm{erfinv}(\mathrm{erf}(x)) = x \$\$

## Examples

``` r
if (torch_is_installed()) {

torch_erfinv(torch_tensor(c(0, 0.5, -1.)))
}
#> torch_tensor
#>  0.0000
#>  0.4769
#>    -inf
#> [ CPUFloatType{3} ]
```
