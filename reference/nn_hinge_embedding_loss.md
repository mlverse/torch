# Hinge embedding loss

Measures the loss given an input tensor \\x\\ and a labels tensor \\y\\
(containing 1 or -1).

## Usage

``` r
nn_hinge_embedding_loss(margin = 1, reduction = "mean")
```

## Arguments

- margin:

  (float, optional): Has a default value of `1`.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

This is usually used for measuring whether two inputs are similar or
dissimilar, e.g. using the L1 pairwise distance as \\x\\, and is
typically used for learning nonlinear embeddings or semi-supervised
learning. The loss function for \\n\\-th sample in the mini-batch is

\$\$ l_n = \begin{array}{ll} x_n, & \mbox{if}\\ y_n = 1,\\ \max \\0,
\Delta - x_n\\, & \mbox{if}\\ y_n = -1, \end{array} \$\$

and the total loss functions is

\$\$ \ell(x, y) = \begin{array}{ll} \mbox{mean}(L), & \mbox{if
reduction} = \mbox{'mean';}\\ \mbox{sum}(L), & \mbox{if reduction} =
\mbox{'sum'.} \end{array} \$\$

where \\L = \\l_1,\dots,l_N\\^\top\\.

## Shape

- Input: \\(\*)\\ where \\\*\\ means, any number of dimensions. The sum
  operation operates over all the elements.

- Target: \\(\*)\\, same shape as the input

- Output: scalar. If `reduction` is `'none'`, then same shape as the
  input
