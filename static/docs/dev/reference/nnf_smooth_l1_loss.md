# Smooth_l1_loss

Function that uses a squared term if the absolute element-wise error
falls below 1 and an L1 term otherwise.

## Usage

``` r
nnf_smooth_l1_loss(input, target, reduction = "mean")
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
