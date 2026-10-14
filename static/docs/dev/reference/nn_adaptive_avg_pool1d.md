# Applies a 1D adaptive average pooling over an input signal composed of several input planes.

The output size is H, for any input size. The number of output features
is equal to the number of input planes.

## Usage

``` r
nn_adaptive_avg_pool1d(output_size)
```

## Arguments

- output_size:

  the target output size H

## Examples

``` r
if (torch_is_installed()) {
# target output size of 5
m <- nn_adaptive_avg_pool1d(5)
input <- torch_randn(1, 64, 8)
output <- m(input)
}
```
