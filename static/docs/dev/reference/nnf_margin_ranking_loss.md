# Margin_ranking_loss

Creates a criterion that measures the loss given inputs x1 , x2 , two 1D
mini-batch Tensors, and a label 1D mini-batch tensor y (containing 1 or
-1).

## Usage

``` r
nnf_margin_ranking_loss(input1, input2, target, margin = 0, reduction = "mean")
```

## Arguments

- input1:

  the first tensor

- input2:

  the second input tensor

- target:

  the target tensor

- margin:

  Has a default value of 00 .

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'
