# Triplet margin loss

Creates a criterion that measures the triplet loss given an input
tensors \\x1\\, \\x2\\, \\x3\\ and a margin with a value greater than
\\0\\. This is used for measuring a relative similarity between samples.
A triplet is composed by `a`, `p` and `n` (i.e., `anchor`,
`positive examples` and `negative examples` respectively). The shapes of
all input tensors should be \\(N, D)\\.

## Usage

``` r
nn_triplet_margin_loss(
  margin = 1,
  p = 2,
  eps = 1e-06,
  swap = FALSE,
  reduction = "mean"
)
```

## Arguments

- margin:

  (float, optional): Default: \\1\\.

- p:

  (int, optional): The norm degree for pairwise distance. Default:
  \\2\\.

- eps:

  constant to avoid NaN's

- swap:

  (bool, optional): The distance swap is described in detail in the
  paper Learning shallow convolutional feature descriptors with triplet
  losses by V. Balntas, E. Riba et al. Default: `FALSE`.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

The distance swap is described in detail in the paper Learning shallow
convolutional feature descriptors with triplet losses
[doi:10.5244/C.30.119](https://doi.org/10.5244/C.30.119) by V. Balntas,
E. Riba et al.

The loss function for each sample in the mini-batch is:

\$\$ L(a, p, n) = \max \\d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\\
\$\$

where

\$\$ d(x_i, y_i) = \| {\bf x}\_i - {\bf y}\_i \|\_p \$\$

See also
[`nn_triplet_margin_with_distance_loss()`](https://torch.mlverse.org/docs/dev/reference/nn_triplet_margin_with_distance_loss.md),
which computes the triplet margin loss for input tensors using a custom
distance function.

## Shape

- Input: \\(N, D)\\ where \\D\\ is the vector dimension.

- Output: A Tensor of shape \\(N)\\ if `reduction` is `'none'`, or a
  scalar otherwise.

## Examples

``` r
if (torch_is_installed()) {
triplet_loss <- nn_triplet_margin_loss(margin = 1, p = 2)
anchor <- torch_randn(100, 128, requires_grad = TRUE)
positive <- torch_randn(100, 128, requires_grad = TRUE)
negative <- torch_randn(100, 128, requires_grad = TRUE)
output <- triplet_loss(anchor, positive, negative)
output$backward()
}
```
