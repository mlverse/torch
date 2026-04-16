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
#>  -0.8662  -6.1552  -3.2274   2.1831   4.7981
#>   -2.9379   0.4270   1.0874   3.9260  -1.9148
#>    1.4361   3.7995   1.9862  -1.3077   2.2460
#>   -1.8504  -1.5365  -2.3666  -4.0822  11.0315
#>    4.3983  -2.5553  -1.7901  -2.9753  -2.0264
#> 
#> (1,2,.,.) = 
#>  -4.7226   2.6150  11.4427   6.7458  -0.6201
#>    2.2145   2.5257  -1.2763   1.7593  -3.3740
#>    1.0715   5.1892  -5.1296 -12.8243  -1.4433
#>    1.0539   0.5500   2.2500   1.7594   8.6363
#>    2.3669  -1.0449   2.9189   0.1593  -4.6306
#> 
#> (1,3,.,.) = 
#>  1.5699  0.2021 -0.7585  5.5502  2.4399
#>   5.8467 -4.0177  3.1549  4.7124 -2.9484
#>   1.5327 -4.0435  2.7621 -0.6371 -3.8565
#>   2.3391 -3.3758 -1.4726  1.9412  1.3874
#>   0.0743 -1.3162 -0.5319  1.7018  3.0911
#> 
#> (1,4,.,.) = 
#>  6.7952 -0.7293  5.0920 -3.3582 -4.9383
#>  -2.5088 -4.8477 -8.8907 -1.5769  8.2621
#>   2.0605  0.5661  2.6521  5.6957  5.6726
#>  -2.0429  7.4660  0.3916 -5.8669  1.9843
#>   0.2429  3.4279  4.3999 -8.2631  7.8485
#> 
#> (1,5,.,.) = 
#>  1.1546 -2.1596 -3.7596  0.5946  5.7377
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
