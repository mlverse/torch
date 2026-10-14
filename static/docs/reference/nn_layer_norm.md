# Layer normalization

Applies Layer Normalization over a mini-batch of inputs as described in
the paper [Layer Normalization](https://arxiv.org/abs/1607.06450)

## Usage

``` r
nn_layer_norm(normalized_shape, eps = 1e-05, elementwise_affine = TRUE)
```

## Arguments

- normalized_shape:

  (int or list): input shape from an expected input of size \\\[\*
  \times \mbox{normalized\\shape}\[0\] \times
  \mbox{normalized\\shape}\[1\] \times \ldots \times
  \mbox{normalized\\shape}\[-1\]\]\\ If a single integer is used, it is
  treated as a singleton list, and this module will normalize over the
  last dimension which is expected to be of that specific size.

- eps:

  a value added to the denominator for numerical stability. Default:
  1e-5

- elementwise_affine:

  a boolean value that when set to `TRUE`, this module has learnable
  per-element affine parameters initialized to ones (for weights) and
  zeros (for biases). Default: `TRUE`.

## Details

\$\$ y = \frac{x - \mathrm{E}\[x\]}{ \sqrt{\mathrm{Var}\[x\] +
\epsilon}} \* \gamma + \beta \$\$

The mean and standard-deviation are calculated separately over the last
certain number dimensions which have to be of the shape specified by
`normalized_shape`.

\\\gamma\\ and \\\beta\\ are learnable affine transform parameters of
`normalized_shape` if `elementwise_affine` is `TRUE`.

The standard-deviation is calculated via the biased estimator,
equivalent to `torch_var(input, unbiased=FALSE)`.

## Note

Unlike Batch Normalization and Instance Normalization, which applies
scalar scale and bias for each entire channel/plane with the `affine`
option, Layer Normalization applies per-element scale and bias with
`elementwise_affine`.

This layer uses statistics computed from input data in both training and
evaluation modes.

## Shape

- Input: \\(N, \*)\\

- Output: \\(N, \*)\\ (same shape as input)

## Examples

``` r
if (torch_is_installed()) {

input <- torch_randn(20, 5, 10, 10)
# With Learnable Parameters
m <- nn_layer_norm(input$size()[-1])
# Without Learnable Parameters
m <- nn_layer_norm(input$size()[-1], elementwise_affine = FALSE)
# Normalize over last two dimensions
m <- nn_layer_norm(c(10, 10))
# Normalize over last dimension of size 10
m <- nn_layer_norm(10)
# Activating the module
output <- m(input)
}
```
