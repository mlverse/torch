# Creates a normal (also called Gaussian) distribution parameterized by `loc` and `scale`.

Creates a normal (also called Gaussian) distribution parameterized by
`loc` and `scale`.

## Usage

``` r
distr_normal(loc, scale, validate_args = NULL)
```

## Arguments

- loc:

  (float or Tensor): mean of the distribution (often referred to as mu)

- scale:

  (float or Tensor): standard deviation of the distribution (often
  referred to as sigma)

- validate_args:

  Additional arguments

## Value

Object of `torch_Normal` class

## See also

[Distribution](https://torch.mlverse.org/docs/dev/reference/Distribution.md)
for details on the available methods.

Other distributions:
[`distr_bernoulli()`](https://torch.mlverse.org/docs/dev/reference/distr_bernoulli.md),
[`distr_chi2()`](https://torch.mlverse.org/docs/dev/reference/distr_chi2.md),
[`distr_gamma()`](https://torch.mlverse.org/docs/dev/reference/distr_gamma.md),
[`distr_multivariate_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_multivariate_normal.md),
[`distr_poisson()`](https://torch.mlverse.org/docs/dev/reference/distr_poisson.md)

## Examples

``` r
if (torch_is_installed()) {
m <- distr_normal(loc = 0, scale = 1)
m$sample() # normally distributed with loc=0 and scale=1
}
#> torch_tensor
#> -0.6511
#> [ CPUFloatType{1} ]
```
