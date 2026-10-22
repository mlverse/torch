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
#>   7.5492   1.5364  -2.7861   6.9978   0.6142
#>   -0.1406   7.1573   7.1540 -10.4438 -11.1228
#>   -7.6529  -6.0910  -7.5154  -2.9351  -3.1699
#>    2.6955  -0.8462  -9.2796  -9.5660  -3.8699
#>    1.5046  -3.4966   3.0052  -4.2991   5.7284
#> 
#> (1,2,.,.) = 
#>  -0.8193   3.6824   7.8091  -6.3083  -0.4852
#>   -2.7653  -2.1427   3.6406   6.6315  10.4061
#>   -2.2392   6.9159  -1.2693   5.4834  -3.1248
#>    7.6084  -5.9483   4.1148   5.3096  -3.1294
#>   -6.9904   3.9959  -6.1237  -3.3862  -2.7110
#> 
#> (1,3,.,.) = 
#>  -2.1458  -5.5239  -0.3817  -1.7620   3.3352
#>    2.2006  -4.1826   7.7403  10.8004  -1.7885
#>    5.8183   1.0044   4.1442   6.0351   4.5468
#>  -18.5339   3.6110  -1.1461  -2.1043  -0.6847
#>   -0.2007  -1.7555  -3.8056   2.2491   0.7452
#> 
#> (1,4,.,.) = 
#>   2.0558   0.5593   5.2933  -2.8404  -2.5102
#>   -0.9866   4.1012 -22.3640 -12.7877   0.1012
#>  -11.7473  10.6986 -12.3946   3.5141  -0.9056
#>   10.7475  -8.7583  -3.7733   5.4065   2.3142
#>    3.2381  -5.0411   7.8155   4.1406   0.4154
#> 
#> (1,5,.,.) = 
#>  0.1474  2.6455  8.4254  1.4667 -2.1743
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
