# Conv_transpose2d

Conv_transpose2d

## Usage

``` r
torch_conv_transpose2d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  output_padding = 0L,
  groups = 1L,
  dilation = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padH, padW)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padH, out_padW)`.
  Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

## conv_transpose2d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
inputs = torch_randn(c(1, 4, 5, 5))
weights = torch_randn(c(4, 8, 3, 3))
nnf_conv_transpose2d(inputs, weights, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#> -4.2212  5.8413 -1.6348  2.5203  2.4106
#>   2.9831  1.7898 -5.1447  4.4301  0.4185
#>   4.0162 -2.4390 -9.1567 -8.8472  7.9008
#>  -2.1597 -4.3073  9.4875 -5.6061 -4.8604
#>  -2.7048 -1.3778 -4.2260  7.8913 -1.6399
#> 
#> (1,2,.,.) = 
#>  8.4247 -3.2194  4.6318  0.3413  1.6036
#>  -8.2543  1.3958 -1.5592 -8.9083 -0.9262
#>  -5.3314 -2.5565 -3.3299 -2.2390 -1.3404
#>   6.2506 -2.1519 -5.7677  4.3111  4.6089
#>   0.4767  3.5603  3.4368 -2.3805  2.8160
#> 
#> (1,3,.,.) = 
#>  -3.9597  -4.5412  -5.9826   4.0273   0.3648
#>    6.1633 -16.0694  -9.9403   4.6366   5.3262
#>    8.4288   4.5402 -17.6710  -4.5090  13.2438
#>    5.0579   5.2204  -6.9236  -5.1348  -2.7598
#>    1.9817   1.8273   0.2151  -4.1732   0.6670
#> 
#> (1,4,.,.) = 
#>  4.2785 -6.8403 -1.6892 -4.4638 -2.1457
#>   2.3019  1.3732 -6.4554  1.6557  1.6288
#>  -0.2475  5.0021 -3.7653  0.3859  1.7391
#>   1.5504 -4.2791 -2.8019  4.6653  1.4031
#>  -1.5183  8.2701 -1.0011 -6.9393 -3.7580
#> 
#> (1,5,.,.) = 
#>  -4.8024  11.6970   2.7967   3.0085  -8.3566
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
