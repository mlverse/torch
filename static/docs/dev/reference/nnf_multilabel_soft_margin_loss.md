# Multilabel_soft_margin_loss

Creates a criterion that optimizes a multi-label one-versus-all loss
based on max-entropy, between input x and target y of size (N, C).

## Usage

``` r
nnf_multilabel_soft_margin_loss(
  input,
  target,
  weight = NULL,
  reduction = "mean"
)
```

## Arguments

- input:

  tensor (N,\*) where \*\* means, any number of additional dimensions

- target:

  tensor (N,\*) , same shape as the input

- weight:

  weight tensor to apply on the loss.

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'

## Note

It takes a one hot encoded target vector as input.
