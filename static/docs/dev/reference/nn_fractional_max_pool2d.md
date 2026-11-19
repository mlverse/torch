# Applies a 2D fractional max pooling over an input signal composed of several input planes.

Fractional MaxPooling is described in detail in the paper [Fractional
MaxPooling](https://arxiv.org/abs/1412.6071) by Ben Graham

## Usage

``` r
nn_fractional_max_pool2d(
  kernel_size,
  output_size = NULL,
  output_ratio = NULL,
  return_indices = FALSE
)
```

## Arguments

- kernel_size:

  the size of the window to take a max over. Can be a single number k
  (for a square kernel of k x k) or a tuple `(kh, kw)`

- output_size:

  the target output size of the image of the form `oH x oW`. Can be a
  tuple `(oH, oW)` or a single number oH for a square image `oH x oH`

- output_ratio:

  If one wants to have an output size as a ratio of the input size, this
  option can be given. This has to be a number or tuple in the range (0,
  1)

- return_indices:

  if `TRUE`, will return the indices along with the outputs. Useful to
  pass to
  [`nn_max_unpool2d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_unpool2d.md).
  Default: `FALSE`

## Details

The max-pooling operation is applied in \\kH \times kW\\ regions by a
stochastic step size determined by the target output size. The number of
output features is equal to the number of input planes.

## Examples

``` r
if (torch_is_installed()) {
# pool of square window of size=3, and target output size 13x12
m <- nn_fractional_max_pool2d(3, output_size = c(13, 12))
# pool of square window and target output size being half of input image size
m <- nn_fractional_max_pool2d(3, output_ratio = c(0.5, 0.5))
input <- torch_randn(20, 16, 50, 32)
output <- m(input)
}
```
