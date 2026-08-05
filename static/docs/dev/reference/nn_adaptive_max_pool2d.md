# Applies a 2D adaptive max pooling over an input signal composed of several input planes.

The output is of size H x W, for any input size. The number of output
features is equal to the number of input planes.

## Usage

``` r
nn_adaptive_max_pool2d(output_size, return_indices = FALSE)
```

## Arguments

- output_size:

  the target output size of the image of the form H x W. Can be a tuple
  `(H, W)` or a single H for a square image H x H. H and W can be either
  a `int`, or `None` which means the size will be the same as that of
  the input.

- return_indices:

  if `TRUE`, will return the indices along with the outputs. Useful to
  pass to
  [`nn_max_unpool2d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_unpool2d.md).
  Default: `FALSE`

## Examples

``` r
if (torch_is_installed()) {
# target output size of 5x7
m <- nn_adaptive_max_pool2d(c(5, 7))
input <- torch_randn(1, 64, 8, 9)
output <- m(input)
# target output size of 7x7 (square)
m <- nn_adaptive_max_pool2d(7)
input <- torch_randn(1, 64, 10, 9)
output <- m(input)
}
```
