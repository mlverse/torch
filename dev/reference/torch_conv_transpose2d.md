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
#>   0.0428  -1.1502  -1.4725  -5.4103  -2.9640
#>    2.8154  -1.0289  -5.1773  -1.5119   1.6455
#>   -4.8060  -1.3383 -10.5943  -0.4165   0.6038
#>    4.3618   6.8550   7.8568  -0.9156  -2.6889
#>   -2.2344   8.9733   0.8780  -1.6682  -8.9244
#> 
#> (1,2,.,.) = 
#>  -0.2630  -6.8117   2.3262   9.3647  -6.6377
#>   -2.9523   4.2583 -12.8080   5.5663  -1.8744
#>   -5.0584   9.4303  -1.0002  -5.9290  -2.6461
#>    6.4980   7.0796   1.3053  -6.0204  -0.6871
#>   -4.5047   3.3842   5.2351   4.6235   7.3639
#> 
#> (1,3,.,.) = 
#>   3.1065   6.3986  -2.4401  -2.3037  -1.6570
#>   -4.1815  -1.5908  -5.8682 -11.3920  -3.8451
#>    7.8591   4.2043   9.0262  10.6376   2.8128
#>   -6.3690   2.8684   8.1628   0.4858  -2.1321
#>    2.0921  -1.4002  -4.1885  -0.4827   4.3510
#> 
#> (1,4,.,.) = 
#> -0.6336  0.4208  4.0268 -2.1125 -2.2042
#>   7.5682  6.8714  4.4521  3.6394 -0.8871
#>   0.0850 -0.2083 -3.0427  5.1853  2.5702
#>   2.4956 -7.5272 -1.9704  4.0576  2.3596
#>  -3.3450  4.7651 -0.7436  6.0362 -3.5492
#> 
#> (1,5,.,.) = 
#>  -3.1980  -2.9381   0.6456   6.8957  -2.3621
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
