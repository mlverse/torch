# Softmax2d module

Applies SoftMax over features to each spatial location. When given an
image of `Channels x Height x Width`, it will apply `Softmax` to each
location \\(Channels, h_i, w_j)\\

## Usage

``` r
nn_softmax2d()
```

## Value

a Tensor of the same dimension and shape as the input with values in the
range `[0, 1]`

## Shape

- Input: \\(N, C, H, W)\\

- Output: \\(N, C, H, W)\\ (same shape as input)

## Examples

``` r
if (torch_is_installed()) {
m <- nn_softmax2d()
input <- torch_randn(2, 3, 12, 13)
output <- m(input)
}
```
