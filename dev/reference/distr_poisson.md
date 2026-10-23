# Creates a Poisson distribution parameterized by `rate`, the rate parameter.

Samples are nonnegative integers, with a pmf given by \$\$
\mbox{rate}^{k} \frac{e^{-\mbox{rate}}}{k!} \$\$

## Usage

``` r
distr_poisson(rate, validate_args = NULL)
```

## Arguments

- rate:

  (numeric, torch_tensor): the rate parameter

- validate_args:

  whether to validate arguments or not.

## See also

[Distribution](https://torch.mlverse.org/docs/dev/reference/Distribution.md)
for details on the available methods.

Other distributions:
[`distr_bernoulli()`](https://torch.mlverse.org/docs/dev/reference/distr_bernoulli.md),
[`distr_chi2()`](https://torch.mlverse.org/docs/dev/reference/distr_chi2.md),
[`distr_gamma()`](https://torch.mlverse.org/docs/dev/reference/distr_gamma.md),
[`distr_multivariate_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_multivariate_normal.md),
[`distr_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_normal.md)

## Examples

``` r
if (torch_is_installed()) {
m <- distr_poisson(torch_tensor(4))
m$sample()
}
#> torch_tensor
#>  5
#> [ CPUFloatType{1} ]
```
