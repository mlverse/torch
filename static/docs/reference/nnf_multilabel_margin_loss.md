# Multilabel_margin_loss

Creates a criterion that optimizes a multi-class multi-classification
hinge loss (margin-based loss) between input x (a 2D mini-batch Tensor)
and output y (which is a 2D Tensor of target class indices).

## Usage

``` r
nnf_multilabel_margin_loss(input, target, reduction = "mean")
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
