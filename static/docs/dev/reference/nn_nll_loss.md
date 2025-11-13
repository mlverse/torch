# Nll loss

The negative log likelihood loss. It is useful to train a classification
problem with `C` classes.

## Usage

``` r
nn_nll_loss(weight = NULL, ignore_index = -100, reduction = "mean")
```

## Arguments

- weight:

  (Tensor, optional): a manual rescaling weight given to each class. If
  given, it has to be a Tensor of size `C`. Otherwise, it is treated as
  if having all ones.

- ignore_index:

  (int, optional): Specifies a target value that is ignored and does not
  contribute to the input gradient.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the weighted mean of the output is taken, `'sum'`:
  the output will be summed.

## Details

If provided, the optional argument `weight` should be a 1D Tensor
assigning weight to each of the classes. This is particularly useful
when you have an unbalanced training set.

The `input` given through a forward call is expected to contain
log-probabilities of each class. `input` has to be a Tensor of size
either \\(minibatch, C)\\ or \\(minibatch, C, d_1, d_2, ..., d_K)\\ with
\\K \geq 1\\ for the `K`-dimensional case (described later).

Obtaining log-probabilities in a neural network is easily achieved by
adding a `LogSoftmax` layer in the last layer of your network.

You may use `CrossEntropyLoss` instead, if you prefer not to add an
extra layer.

The `target` that this loss expects should be a class index in the range
\\\[0, C-1\]\\ where `C = number of classes`; if `ignore_index` is
specified, this loss also accepts this class index (this index may not
necessarily be in the class range).

The unreduced (i.e. with `reduction` set to `'none'`) loss can be
described as:

\$\$ \ell(x, y) = L = \\l_1,\dots,l_N\\^\top, \quad l_n = - w\_{y_n}
x\_{n,y_n}, \quad w\_{c} = \mbox{weight}\[c\] \cdot \mbox{1}\\c \not=
\mbox{ignore\\index}\\, \$\$

where \\x\\ is the input, \\y\\ is the target, \\w\\ is the weight, and
\\N\\ is the batch size. If `reduction` is not `'none'` (default
`'mean'`), then

\$\$ \ell(x, y) = \begin{array}{ll} \sum\_{n=1}^N \frac{1}{\sum\_{n=1}^N
w\_{y_n}} l_n, & \mbox{if reduction} = \mbox{'mean';}\\ \sum\_{n=1}^N
l_n, & \mbox{if reduction} = \mbox{'sum'.} \end{array} \$\$

Can also be used for higher dimension inputs, such as 2D images, by
providing an input of size \\(minibatch, C, d_1, d_2, ..., d_K)\\ with
\\K \geq 1\\, where \\K\\ is the number of dimensions, and a target of
appropriate shape (see below). In the case of images, it computes NLL
loss per-pixel.

## Shape

- Input: \\(N, C)\\ where `C = number of classes`, or \\(N, C, d_1, d_2,
  ..., d_K)\\ with \\K \geq 1\\ in the case of `K`-dimensional loss.

- Target: \\(N)\\ where each value is \\0 \leq \mbox{targets}\[i\] \leq
  C-1\\, or \\(N, d_1, d_2, ..., d_K)\\ with \\K \geq 1\\ in the case of
  K-dimensional loss.

- Output: scalar.

If `reduction` is `'none'`, then the same size as the target: \\(N)\\,
or \\(N, d_1, d_2, ..., d_K)\\ with \\K \geq 1\\ in the case of
K-dimensional loss.

## Examples

``` r
if (torch_is_installed()) {
m <- nn_log_softmax(dim = 2)
loss <- nn_nll_loss()
# input is of size N x C = 3 x 5
input <- torch_randn(3, 5, requires_grad = TRUE)
# each element in target has to have 0 <= value < C
target <- torch_tensor(c(2, 1, 5), dtype = torch_long())
output <- loss(m(input), target)
output$backward()

# 2D loss example (used, for example, with image inputs)
N <- 5
C <- 4
loss <- nn_nll_loss()
# input is of size N x C x height x width
data <- torch_randn(N, 16, 10, 10)
conv <- nn_conv2d(16, C, c(3, 3))
m <- nn_log_softmax(dim = 1)
# each element in target has to have 0 <= value < C
target <- torch_empty(N, 8, 8, dtype = torch_long())$random_(1, C)
output <- loss(m(conv(data)), target)
output$backward()
}
```
