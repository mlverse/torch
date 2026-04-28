# Soft margin loss

Creates a criterion that optimizes a two-class classification logistic
loss between input tensor \\x\\ and target tensor \\y\\ (containing 1 or
-1).

## Usage

``` r
nn_soft_margin_loss(reduction = "mean")
```

## Arguments

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

\$\$ \mbox{loss}(x, y) = \sum_i \frac{\log(1 +
\exp(-y\[i\]\*x\[i\]))}{\mbox{x.nelement}()} \$\$

## Shape

- Input: \\(\*)\\ where \\\*\\ means, any number of additional
  dimensions

- Target: \\(\*)\\, same shape as the input

- Output: scalar. If `reduction` is `'none'`, then same shape as the
  input
