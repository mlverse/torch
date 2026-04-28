# Multi margin loss

Creates a criterion that optimizes a multi-class classification hinge
loss (margin-based loss) between input \\x\\ (a 2D mini-batch `Tensor`)
and output \\y\\ (which is a 1D tensor of target class indices, \\0 \leq
y \leq \mbox{x.size}(1)-1\\):

## Usage

``` r
nn_multi_margin_loss(p = 1, margin = 1, weight = NULL, reduction = "mean")
```

## Arguments

- p:

  (int, optional): Has a default value of \\1\\. \\1\\ and \\2\\ are the
  only supported values.

- margin:

  (float, optional): Has a default value of \\1\\.

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

For each mini-batch sample, the loss in terms of the 1D input \\x\\ and
scalar output \\y\\ is: \$\$ \mbox{loss}(x, y) = \frac{\sum_i \max(0,
\mbox{margin} - x\[y\] + x\[i\]))^p}{\mbox{x.size}(0)} \$\$

where \\x \in \left\\0, \\ \cdots , \\ \mbox{x.size}(0) - 1\right\\\\
and \\i \neq y\\.

Optionally, you can give non-equal weighting on the classes by passing a
1D `weight` tensor into the constructor. The loss function then becomes:

\$\$ \mbox{loss}(x, y) = \frac{\sum_i \max(0, w\[y\] \* (\mbox{margin} -
x\[y\] + x\[i\]))^p)}{\mbox{x.size}(0)} \$\$
