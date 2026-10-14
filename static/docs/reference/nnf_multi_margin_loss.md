# Multi_margin_loss

Creates a criterion that optimizes a multi-class classification hinge
loss (margin-based loss) between input x (a 2D mini-batch Tensor) and
output y (which is a 1D tensor of target class indices,
`0 <= y <= x$size(2) - 1` ).

## Usage

``` r
nnf_multi_margin_loss(
  input,
  target,
  p = 1,
  margin = 1,
  weight = NULL,
  reduction = "mean"
)
```

## Arguments

- input:

  tensor (N,\*) where \*\* means, any number of additional dimensions

- target:

  tensor (N,\*) , same shape as the input

- p:

  Has a default value of 1. 1 and 2 are the only supported values.

- margin:

  Has a default value of 1.

- weight:

  a manual rescaling weight given to each class. If given, it has to be
  a Tensor of size C. Otherwise, it is treated as if having all ones.

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'
