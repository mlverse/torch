# Applies a 3D adaptive average pooling over an input signal composed of several input planes.

The output is of size D x H x W, for any input size. The number of
output features is equal to the number of input planes.

## Usage

``` r
nn_adaptive_avg_pool3d(output_size)
```

## Arguments

- output_size:

  the target output size of the form D x H x W. Can be a tuple (D, H, W)
  or a single number D for a cube D x D x D. D, H and W can be either a
  `int`, or `None` which means the size will be the same as that of the
  input.

## Examples

``` r
if (torch_is_installed()) {
# target output size of 5x7x9
m <- nn_adaptive_avg_pool3d(c(5, 7, 9))
input <- torch_randn(1, 64, 8, 9, 10)
output <- m(input)
# target output size of 7x7x7 (cube)
m <- nn_adaptive_avg_pool3d(7)
input <- torch_randn(1, 64, 10, 9, 8)
output <- m(input)
}
```
