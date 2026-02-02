# Cross_entropy

This criterion combines `log_softmax` and `nll_loss` in a single
function.

## Usage

``` r
nnf_cross_entropy(
  input,
  target,
  weight = NULL,
  ignore_index = -100,
  reduction = c("mean", "sum", "none")
)
```

## Arguments

- input:

  (Tensor) \\(N, C)\\ where `C = number of classes` or \\(N, C, H, W)\\
  in case of 2D Loss, or \\(N, C, d_1, d_2, ..., d_K)\\ where \\K \geq
  1\\ in the case of K-dimensional loss.

- target:

  (Tensor) \\(N)\\ where each value is \\0 \leq \mbox{targets}\[i\] \leq
  C-1\\, or \\(N, d_1, d_2, ..., d_K)\\ where \\K \geq 1\\ for
  K-dimensional loss.

- weight:

  (Tensor, optional) a manual rescaling weight given to each class. If
  given, has to be a Tensor of size `C`

- ignore_index:

  (int, optional) Specifies a target value that is ignored and does not
  contribute to the input gradient.

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'
