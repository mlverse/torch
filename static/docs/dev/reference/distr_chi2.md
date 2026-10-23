# Creates a Chi2 distribution parameterized by shape parameter `df`. This is exactly equivalent to `distr_gamma(alpha=0.5*df, beta=0.5)`

Creates a Chi2 distribution parameterized by shape parameter `df`. This
is exactly equivalent to `distr_gamma(alpha=0.5*df, beta=0.5)`

## Usage

``` r
distr_chi2(df, validate_args = NULL)
```

## Arguments

- df:

  (float or torch_tensor): shape parameter of the distribution

- validate_args:

  whether to validate arguments or not.

## See also

[Distribution](https://torch.mlverse.org/docs/dev/reference/Distribution.md)
for details on the available methods.

Other distributions:
[`distr_bernoulli()`](https://torch.mlverse.org/docs/dev/reference/distr_bernoulli.md),
[`distr_gamma()`](https://torch.mlverse.org/docs/dev/reference/distr_gamma.md),
[`distr_multivariate_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_multivariate_normal.md),
[`distr_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_normal.md),
[`distr_poisson()`](https://torch.mlverse.org/docs/dev/reference/distr_poisson.md)

## Examples

``` r
if (torch_is_installed()) {
m <- distr_chi2(torch_tensor(1.0))
m$sample() # Chi2 distributed with shape df=1
torch_tensor(0.1046)
}
#> torch_tensor
#>  0.1046
#> [ CPUFloatType{1} ]
```
