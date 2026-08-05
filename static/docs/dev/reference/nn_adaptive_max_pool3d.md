# Applies a 3D adaptive max pooling over an input signal composed of several input planes.

The output is of size D x H x W, for any input size. The number of
output features is equal to the number of input planes.

## Usage

``` r
nn_adaptive_max_pool3d(output_size, return_indices = FALSE)
```

## Arguments

- output_size:

  the target output size of the image of the form D x H x W. Can be a
  tuple (D, H, W) or a single D for a cube D x D x D. D, H and W can be
  either a `int`, or `None` which means the size will be the same as
  that of the input.

- return_indices:

  if `TRUE`, will return the indices along with the outputs. Useful to
  pass to
  [`nn_max_unpool3d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_unpool3d.md).
  Default: `FALSE`

## Examples

``` r
if (torch_is_installed()) {
# target output size of 5x7x9
m <- nn_adaptive_max_pool3d(c(5, 7, 9))
input <- torch_randn(1, 64, 8, 9, 10)
output <- m(input)
# target output size of 7x7x7 (cube)
m <- nn_adaptive_max_pool3d(7)
input <- torch_randn(1, 64, 10, 9, 8)
output <- m(input)
}
```
