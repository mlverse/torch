# Creates a Gamma distribution parameterized by shape `concentration` and `rate`.

Creates a Gamma distribution parameterized by shape `concentration` and
`rate`.

## Usage

``` r
distr_gamma(concentration, rate, validate_args = NULL)
```

## Arguments

- concentration:

  (float or Tensor): shape parameter of the distribution (often referred
  to as alpha)

- rate:

  (float or Tensor): rate = 1 / scale of the distribution (often
  referred to as beta)

- validate_args:

  whether to validate arguments or not.

## See also

[Distribution](https://torch.mlverse.org/docs/dev/reference/Distribution.md)
for details on the available methods.

Other distributions:
[`distr_bernoulli()`](https://torch.mlverse.org/docs/dev/reference/distr_bernoulli.md),
[`distr_chi2()`](https://torch.mlverse.org/docs/dev/reference/distr_chi2.md),
[`distr_multivariate_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_multivariate_normal.md),
[`distr_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_normal.md),
[`distr_poisson()`](https://torch.mlverse.org/docs/dev/reference/distr_poisson.md)

## Examples

``` r
if (torch_is_installed()) {
m <- distr_gamma(torch_tensor(1.0), torch_tensor(1.0))
m$sample() # Gamma distributed with concentration=1 and rate=1
}
#> torch_tensor
#>  1.3633
#> [ CPUFloatType{1} ]
```
