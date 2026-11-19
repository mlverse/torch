# Mixture of components in the same family

The `MixtureSameFamily` distribution implements a (batch of) mixture
distribution where all component are from different parameterizations of
the same distribution type. It is parameterized by a `Categorical`
selecting distribution" (over `k` component) and a component
distribution, i.e., a `Distribution` with a rightmost batch shape (equal
to `[k]`) which indexes each (batch of) component.

## Usage

``` r
distr_mixture_same_family(
  mixture_distribution,
  component_distribution,
  validate_args = NULL
)
```

## Arguments

- mixture_distribution:

  `torch_distributions.Categorical`-like instance. Manages the
  probability of selecting component. The number of categories must
  match the rightmost batch dimension of the `component_distribution`.
  Must have either scalar `batch_shape` or `batch_shape` matching
  `component_distribution.batch_shape[:-1]`

- component_distribution:

  `torch_distributions.Distribution`-like instance. Right-most batch
  dimension indexes component.

- validate_args:

  Additional arguments

## Examples

``` r
if (torch_is_installed()) {
# Construct Gaussian Mixture Model in 1D consisting of 5 equally
# weighted normal distributions
mix <- distr_categorical(torch_ones(5))
comp <- distr_normal(torch_randn(5), torch_rand(5))
gmm <- distr_mixture_same_family(mix, comp)
}
```
