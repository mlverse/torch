# Creates a Bernoulli distribution parameterized by `probs` or `logits` (but not both). Samples are binary (0 or 1). They take the value `1` with probability `p` and `0` with probability `1 - p`.

Creates a Bernoulli distribution parameterized by `probs` or `logits`
(but not both). Samples are binary (0 or 1). They take the value `1`
with probability `p` and `0` with probability `1 - p`.

## Usage

``` r
distr_bernoulli(probs = NULL, logits = NULL, validate_args = NULL)
```

## Arguments

- probs:

  (numeric or torch_tensor): the probability of sampling `1`

- logits:

  (numeric or torch_tensor): the log-odds of sampling `1`

- validate_args:

  whether to validate arguments or not.

## See also

[Distribution](https://torch.mlverse.org/docs/dev/reference/Distribution.md)
for details on the available methods.

Other distributions:
[`distr_chi2()`](https://torch.mlverse.org/docs/dev/reference/distr_chi2.md),
[`distr_gamma()`](https://torch.mlverse.org/docs/dev/reference/distr_gamma.md),
[`distr_multivariate_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_multivariate_normal.md),
[`distr_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_normal.md),
[`distr_poisson()`](https://torch.mlverse.org/docs/dev/reference/distr_poisson.md)

## Examples

``` r
if (torch_is_installed()) {
m <- distr_bernoulli(0.3)
m$sample() # 30% chance 1; 70% chance 0
}
#> torch_tensor
#>  0
#> [ CPUFloatType{1} ]
```
