# Binary_cross_entropy_with_logits

Function that measures Binary Cross Entropy between target and output
logits.

## Usage

``` r
nnf_binary_cross_entropy_with_logits(
  input,
  target,
  weight = NULL,
  reduction = c("mean", "sum", "none"),
  pos_weight = NULL
)
```

## Arguments

- input:

  Tensor of arbitrary shape

- target:

  Tensor of the same shape as input

- weight:

  (Tensor, optional) a manual rescaling weight if provided it's repeated
  to match input tensor shape.

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'

- pos_weight:

  (Tensor, optional) a weight of positive examples. Must be a vector
  with length equal to the number of classes.
