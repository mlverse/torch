# Gumbel_softmax

Samples from the Gumbel-Softmax distribution and optionally discretizes.

## Usage

``` r
nnf_gumbel_softmax(logits, tau = 1, hard = FALSE, dim = -1)
```

## Arguments

- logits:

  `[..., num_features]` unnormalized log probabilities

- tau:

  non-negative scalar temperature

- hard:

  if `True`, the returned samples will be discretized as one-hot
  vectors, but will be differentiated as if it is the soft sample in
  autograd

- dim:

  (int) A dimension along which softmax will be computed. Default: -1.
