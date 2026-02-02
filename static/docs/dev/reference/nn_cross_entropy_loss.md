# CrossEntropyLoss module

This criterion combines
[`nn_log_softmax()`](https://torch.mlverse.org/docs/dev/reference/nn_log_softmax.md)
and
[`nn_nll_loss()`](https://torch.mlverse.org/docs/dev/reference/nn_nll_loss.md)
in one single class. It is useful when training a classification problem
with `C` classes.

## Usage

``` r
nn_cross_entropy_loss(weight = NULL, ignore_index = -100, reduction = "mean")
```

## Arguments

- weight:

  (Tensor, optional): a manual rescaling weight given to each class. If
  given, has to be a Tensor of size `C`

- ignore_index:

  (int, optional): Specifies a target value that is ignored and does not
  contribute to the input gradient. When `size_average` is `TRUE`, the
  loss is averaged over non-ignored targets.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

If provided, the optional argument `weight` should be a 1D `Tensor`
assigning weight to each of the classes.

This is particularly useful when you have an unbalanced training set.
The `input` is expected to contain raw, unnormalized scores for each
class. `input` has to be a Tensor of size either \\(minibatch, C)\\ or
\\(minibatch, C, d_1, d_2, ..., d_K)\\ with \\K \geq 1\\ for the
`K`-dimensional case (described later).

This criterion expects a class index in the range \\\[0, C-1\]\\ as the
`target` for each value of a 1D tensor of size `minibatch`; if
`ignore_index` is specified, this criterion also accepts this class
index (this index may not necessarily be in the class range).

The loss can be described as: \$\$ \mbox{loss}(x, class) =
-\log\left(\frac{\exp(x\[class\])}{\sum_j \exp(x\[j\])}\right) =
-x\[class\] + \log\left(\sum_j \exp(x\[j\])\right) \$\$ or in the case
of the `weight` argument being specified: \$\$ \mbox{loss}(x, class) =
weight\[class\] \left(-x\[class\] + \log\left(\sum_j
\exp(x\[j\])\right)\right) \$\$

The losses are averaged across observations for each minibatch. Can also
be used for higher dimension inputs, such as 2D images, by providing an
input of size \\(minibatch, C, d_1, d_2, ..., d_K)\\ with \\K \geq 1\\,
where \\K\\ is the number of dimensions, and a target of appropriate
shape (see below).

## Shape

- Input: \\(N, C)\\ where `C = number of classes`, or \\(N, C, d_1, d_2,
  ..., d_K)\\ with \\K \geq 1\\ in the case of `K`-dimensional loss.

- Target: \\(N)\\ where each value is \\0 \leq \mbox{targets}\[i\] \leq
  C-1\\, or \\(N, d_1, d_2, ..., d_K)\\ with \\K \geq 1\\ in the case of
  K-dimensional loss.

- Output: scalar. If `reduction` is `'none'`, then the same size as the
  target: \\(N)\\, or \\(N, d_1, d_2, ..., d_K)\\ with \\K \geq 1\\ in
  the case of K-dimensional loss.

## Examples

``` r
if (torch_is_installed()) {
loss <- nn_cross_entropy_loss()
input <- torch_randn(3, 5, requires_grad = TRUE)
target <- torch_randint(low = 1, high = 5, size = 3, dtype = torch_long())
output <- loss(input, target)
output$backward()
}
```
