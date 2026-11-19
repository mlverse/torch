# Cosine_embedding_loss

Creates a criterion that measures the loss given input tensors x_1, x_2
and a Tensor label y with values 1 or -1. This is used for measuring
whether two inputs are similar or dissimilar, using the cosine distance,
and is typically used for learning nonlinear embeddings or
semi-supervised learning.

## Usage

``` r
nnf_cosine_embedding_loss(
  input1,
  input2,
  target,
  margin = 0,
  reduction = c("mean", "sum", "none")
)
```

## Arguments

- input1:

  the input x_1 tensor

- input2:

  the input x_2 tensor

- target:

  the target tensor

- margin:

  Should be a number from -1 to 1 , 0 to 0.5 is suggested. If margin is
  missing, the default value is 0.

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'
