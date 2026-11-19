# Triplet margin with distance loss

See
[`nn_triplet_margin_with_distance_loss()`](https://torch.mlverse.org/docs/dev/reference/nn_triplet_margin_with_distance_loss.md)

## Usage

``` r
nnf_triplet_margin_with_distance_loss(
  anchor,
  positive,
  negative,
  distance_function = NULL,
  margin = 1,
  swap = FALSE,
  reduction = "mean"
)
```

## Arguments

- anchor:

  the anchor input tensor

- positive:

  the positive input tensor

- negative:

  the negative input tensor

- distance_function:

  (callable, optional): A nonnegative, real-valued function that
  quantifies the closeness of two tensors. If not specified,
  [`nn_pairwise_distance()`](https://torch.mlverse.org/docs/dev/reference/nn_pairwise_distance.md)
  will be used. Default: `None`

- margin:

  Default: 1.

- swap:

  The distance swap is described in detail in the paper Learning shallow
  convolutional feature descriptors with triplet losses by V.
  Balntas, E. Riba et al. Default: `FALSE`.

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'
