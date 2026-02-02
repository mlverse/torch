# Triplet_margin_loss

Creates a criterion that measures the triplet loss given an input
tensors x1 , x2 , x3 and a margin with a value greater than 0 . This is
used for measuring a relative similarity between samples. A triplet is
composed by a, p and n (i.e., anchor, positive examples and negative
examples respectively). The shapes of all input tensors should be (N,
D).

## Usage

``` r
nnf_triplet_margin_loss(
  anchor,
  positive,
  negative,
  margin = 1,
  p = 2,
  eps = 1e-06,
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

- margin:

  Default: 1.

- p:

  The norm degree for pairwise distance. Default: 2.

- eps:

  (float, optional) Small value to avoid division by zero.

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
