# Margin ranking loss

Creates a criterion that measures the loss given inputs \\x1\\, \\x2\\,
two 1D mini-batch `Tensors`, and a label 1D mini-batch tensor \\y\\
(containing 1 or -1). If \\y = 1\\ then it assumed the first input
should be ranked higher (have a larger value) than the second input, and
vice-versa for \\y = -1\\.

## Usage

``` r
nn_margin_ranking_loss(margin = 0, reduction = "mean")
```

## Arguments

- margin:

  (float, optional): Has a default value of \\0\\.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

The loss function for each pair of samples in the mini-batch is:

\$\$ \mbox{loss}(x1, x2, y) = \max(0, -y \* (x1 - x2) + \mbox{margin})
\$\$

## Shape

- Input1: \\(N)\\ where `N` is the batch size.

- Input2: \\(N)\\, same shape as the Input1.

- Target: \\(N)\\, same shape as the inputs.

- Output: scalar. If `reduction` is `'none'`, then \\(N)\\.

## Examples

``` r
if (torch_is_installed()) {
loss <- nn_margin_ranking_loss()
input1 <- torch_randn(3, requires_grad = TRUE)
input2 <- torch_randn(3, requires_grad = TRUE)
target <- torch_randn(3)$sign()
output <- loss(input1, input2, target)
output$backward()
}
```
