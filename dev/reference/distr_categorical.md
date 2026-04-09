# Creates a categorical distribution parameterized by either `probs` or `logits` (but not both).

Creates a categorical distribution parameterized by either `probs` or
`logits` (but not both).

## Usage

``` r
distr_categorical(probs = NULL, logits = NULL, validate_args = NULL)
```

## Arguments

- probs:

  (Tensor): event probabilities

- logits:

  (Tensor): event log probabilities (unnormalized)

- validate_args:

  Additional arguments

## Note

It is equivalent to the distribution that
[`torch_multinomial()`](https://torch.mlverse.org/docs/dev/reference/torch_multinomial.md)
samples from.

Samples are integers from \\\\0, \ldots, K-1\\\\ where `K` is
`probs$size(-1)`.

If `probs` is 1-dimensional with length-`K`, each element is the
relative probability of sampling the class at that index.

If `probs` is N-dimensional, the first N-1 dimensions are treated as a
batch of relative probability vectors.

The `probs` argument must be non-negative, finite and have a non-zero
sum, and it will be normalized to sum to 1 along the last dimension.
attr:`probs` will return this normalized value. The `logits` argument
will be interpreted as unnormalized log probabilities and can therefore
be any real number. It will likewise be normalized so that the resulting
probabilities sum to 1 along the last dimension. attr:`logits` will
return this normalized value.

See also:
[`torch_multinomial()`](https://torch.mlverse.org/docs/dev/reference/torch_multinomial.md)

## Examples

``` r
if (torch_is_installed()) {
m <- distr_categorical(torch_tensor(c(0.25, 0.25, 0.25, 0.25)))
m$sample() # equal probability of 1,2,3,4
}
#> torch_tensor
#> 2
#> [ CPULongType{} ]
```
