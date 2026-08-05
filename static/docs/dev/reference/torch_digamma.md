# Digamma

Digamma

## Usage

``` r
torch_digamma(self)
```

## Arguments

- self:

  (Tensor) the tensor to compute the digamma function on

## digamma(input, out=NULL) -\> Tensor

Computes the logarithmic derivative of the gamma function on `input`.

\$\$ \psi(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) =
\frac{\Gamma'(x)}{\Gamma(x)} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_tensor(c(1, 0.5))
torch_digamma(a)
}
#> torch_tensor
#> -0.5772
#> -1.9635
#> [ CPUFloatType{2} ]
```
