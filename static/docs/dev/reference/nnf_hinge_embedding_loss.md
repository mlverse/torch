# Hinge_embedding_loss

Measures the loss given an input tensor xx and a labels tensor yy
(containing 1 or -1). This is usually used for measuring whether two
inputs are similar or dissimilar, e.g. using the L1 pairwise distance as
xx , and is typically used for learning nonlinear embeddings or
semi-supervised learning.

## Usage

``` r
nnf_hinge_embedding_loss(input, target, margin = 1, reduction = "mean")
```

## Arguments

- input:

  tensor (N,\*) where \*\* means, any number of additional dimensions

- target:

  tensor (N,\*) , same shape as the input

- margin:

  Has a default value of 1.

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'
