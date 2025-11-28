# Gaussian distribution

Creates a multivariate normal (also called Gaussian) distribution
parameterized by a mean vector and a covariance matrix.

## Usage

``` r
distr_multivariate_normal(
  loc,
  covariance_matrix = NULL,
  precision_matrix = NULL,
  scale_tril = NULL,
  validate_args = NULL
)
```

## Arguments

- loc:

  (Tensor): mean of the distribution

- covariance_matrix:

  (Tensor): positive-definite covariance matrix

- precision_matrix:

  (Tensor): positive-definite precision matrix

- scale_tril:

  (Tensor): lower-triangular factor of covariance, with positive-valued
  diagonal

- validate_args:

  Bool wether to validate the arguments or not.

## Details

The multivariate normal distribution can be parameterized either in
terms of a positive definite covariance matrix \\\mathbf{\Sigma}\\ or a
positive definite precision matrix \\\mathbf{\Sigma}^{-1}\\ or a
lower-triangular matrix \\\mathbf{L}\\ with positive-valued diagonal
entries, such that \\\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top\\. This
triangular matrix can be obtained via e.g. Cholesky decomposition of the
covariance.

## Note

Only one of `covariance_matrix` or `precision_matrix` or `scale_tril`
can be specified. Using `scale_tril` will be more efficient: all
computations internally are based on `scale_tril`. If
`covariance_matrix` or `precision_matrix` is passed instead, it is only
used to compute the corresponding lower triangular matrices using a
Cholesky decomposition.

## See also

[Distribution](https://torch.mlverse.org/docs/dev/reference/Distribution.md)
for details on the available methods.

Other distributions:
[`distr_bernoulli()`](https://torch.mlverse.org/docs/dev/reference/distr_bernoulli.md),
[`distr_chi2()`](https://torch.mlverse.org/docs/dev/reference/distr_chi2.md),
[`distr_gamma()`](https://torch.mlverse.org/docs/dev/reference/distr_gamma.md),
[`distr_normal()`](https://torch.mlverse.org/docs/dev/reference/distr_normal.md),
[`distr_poisson()`](https://torch.mlverse.org/docs/dev/reference/distr_poisson.md)

## Examples

``` r
if (torch_is_installed()) {
m <- distr_multivariate_normal(torch_zeros(2), torch_eye(2))
m$sample() # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
}
#> torch_tensor
#>  0.2241
#> -0.3210
#> [ CPUFloatType{2} ]
```
