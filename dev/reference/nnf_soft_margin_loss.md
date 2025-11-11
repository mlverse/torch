# Soft_margin_loss

Creates a criterion that optimizes a two-class classification logistic
loss between input tensor x and target tensor y (containing 1 or -1).

## Usage

``` r
nnf_soft_margin_loss(input, target, reduction = "mean")
```

## Arguments

- input:

  tensor (N,\*) where \*\* means, any number of additional dimensions

- target:

  tensor (N,\*) , same shape as the input

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'
