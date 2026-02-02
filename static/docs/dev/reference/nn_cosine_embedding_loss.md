# Cosine embedding loss

Creates a criterion that measures the loss given input tensors \\x_1\\,
\\x_2\\ and a `Tensor` label \\y\\ with values 1 or -1. This is used for
measuring whether two inputs are similar or dissimilar, using the cosine
distance, and is typically used for learning nonlinear embeddings or
semi-supervised learning. The loss function for each sample is:

## Usage

``` r
nn_cosine_embedding_loss(margin = 0, reduction = "mean")
```

## Arguments

- margin:

  (float, optional): Should be a number from \\-1\\ to \\1\\, \\0\\ to
  \\0.5\\ is suggested. If `margin` is missing, the default value is
  \\0\\.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

\$\$ \mbox{loss}(x, y) = \begin{array}{ll} 1 - \cos(x_1, x_2), &
\mbox{if } y = 1 \\ \max(0, \cos(x_1, x_2) - \mbox{margin}), & \mbox{if
} y = -1 \end{array} \$\$
