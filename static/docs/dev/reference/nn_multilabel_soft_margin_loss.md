# Multi label soft margin loss

Creates a criterion that optimizes a multi-label one-versus-all loss
based on max-entropy, between input \\x\\ and target \\y\\ of size \\(N,
C)\\.

## Usage

``` r
nn_multilabel_soft_margin_loss(weight = NULL, reduction = "mean")
```

## Arguments

- weight:

  (Tensor, optional): a manual rescaling weight given to each class. If
  given, it has to be a Tensor of size `C`. Otherwise, it is treated as
  if having all ones.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

For each sample in the minibatch:

\$\$ loss(x, y) = - \frac{1}{C} \* \sum_i y\[i\] \* \log((1 +
\exp(-x\[i\]))^{-1}) + (1-y\[i\]) \* \log\left(\frac{\exp(-x\[i\])}{(1 +
\exp(-x\[i\]))}\right) \$\$

where \\i \in \left\\0, \\ \cdots , \\ \mbox{x.nElement}() -
1\right\\\\, \\y\[i\] \in \left\\0, \\ 1\right\\\\.

## Shape

- Input: \\(N, C)\\ where `N` is the batch size and `C` is the number of
  classes.

- Target: \\(N, C)\\, label targets padded by -1 ensuring same shape as
  the input.

- Output: scalar. If `reduction` is `'none'`, then \\(N)\\.
