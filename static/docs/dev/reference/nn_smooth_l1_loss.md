# Smooth L1 loss

Creates a criterion that uses a squared term if the absolute
element-wise error falls below 1 and an L1 term otherwise. It is less
sensitive to outliers than the `MSELoss` and in some cases prevents
exploding gradients (e.g. see `Fast R-CNN` paper by Ross Girshick). Also
known as the Huber loss:

## Usage

``` r
nn_smooth_l1_loss(reduction = "mean")
```

## Arguments

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

\$\$ \mbox{loss}(x, y) = \frac{1}{n} \sum\_{i} z\_{i} \$\$

where \\z\_{i}\\ is given by:

\$\$ z\_{i} = \begin{array}{ll} 0.5 (x_i - y_i)^2, & \mbox{if } \|x_i -
y_i\| \< 1 \\ \|x_i - y_i\| - 0.5, & \mbox{otherwise } \end{array} \$\$

\\x\\ and \\y\\ arbitrary shapes with a total of \\n\\ elements each the
sum operation still operates over all the elements, and divides by
\\n\\. The division by \\n\\ can be avoided if sets `reduction = 'sum'`.

## Shape

- Input: \\(N, \*)\\ where \\\*\\ means, any number of additional
  dimensions

- Target: \\(N, \*)\\, same shape as the input

- Output: scalar. If `reduction` is `'none'`, then \\(N, \*)\\, same
  shape as the input
