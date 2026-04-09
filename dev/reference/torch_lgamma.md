# Lgamma

Lgamma

## Usage

``` r
torch_lgamma(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## lgamma(input, out=NULL) -\> Tensor

Computes the logarithm of the gamma function on `input`.

\$\$ \mbox{out}\_{i} = \log \Gamma(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_arange(0.5, 2, 0.5)
torch_lgamma(a)
}
#> torch_tensor
#>  0.5724
#>  0.0000
#> -0.1208
#>  0.0000
#> [ CPUFloatType{4} ]
```
