# Mvlgamma

Mvlgamma

## Usage

``` r
torch_mvlgamma(self, p)
```

## Arguments

- self:

  (Tensor) the tensor to compute the multivariate log-gamma function

- p:

  (int) the number of dimensions

## mvlgamma(input, p) -\> Tensor

Computes the
`multivariate log-gamma function <https://en.wikipedia.org/wiki/Multivariate_gamma_function>`\_)
with dimension \\p\\ element-wise, given by

\$\$ \log(\Gamma\_{p}(a)) = C + \displaystyle \sum\_{i=1}^{p}
\log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right) \$\$ where \\C =
\log(\pi) \times \frac{p (p - 1)}{4}\\ and \\\Gamma(\cdot)\\ is the
Gamma function.

All elements must be greater than \\\frac{p - 1}{2}\\, otherwise an
error would be thrown.

## Examples

``` r
if (torch_is_installed()) {

a = torch_empty(c(2, 3))$uniform_(1, 2)
a
torch_mvlgamma(a, 2)
}
#> torch_tensor
#>  1.0558  0.3998  0.3905
#>  0.4774  0.4018  0.3899
#> [ CPUFloatType{2,3} ]
```
