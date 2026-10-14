# Applies a 1D adaptive max pooling over an input signal composed of several input planes.

The output size is H, for any input size. The number of output features
is equal to the number of input planes.

## Usage

``` r
nn_adaptive_max_pool1d(output_size, return_indices = FALSE)
```

## Arguments

- output_size:

  the target output size H

- return_indices:

  if `TRUE`, will return the indices along with the outputs. Useful to
  pass to
  [`nn_max_unpool1d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_unpool1d.md).
  Default: `FALSE`

## Examples

``` r
if (torch_is_installed()) {
# target output size of 5
m <- nn_adaptive_max_pool1d(5)
input <- torch_randn(1, 64, 8)
output <- m(input)
}
```
