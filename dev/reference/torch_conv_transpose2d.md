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
#>  -5.5823   9.2580  -1.4258   1.4971   1.5421
#>   10.4425  -4.4813   7.5405   6.4934   1.0293
#>   -6.8625   7.9861   1.7477  -4.2151   5.7830
#>   -1.1321   0.2923  -5.5027  -0.7149  -0.6549
#>   -0.3209   1.8950  -2.9971   1.1916   7.2184
#> 
#> (1,2,.,.) = 
#>  -0.2197   4.4034  -0.9235   0.7758   0.1596
#>    1.1444  -9.9886   3.7665  -0.6729  -3.6895
#>   -1.0078   6.5646  -5.7650  -1.3102   3.5140
#>    0.5894   0.0770  -7.1711   5.6144   1.2261
#>    1.5336   1.8755   1.6663  15.1110   7.0821
#> 
#> (1,3,.,.) = 
#>   6.5978   5.0306  -1.9129  -0.3746  -0.8283
#>   -6.5850  -5.4524   4.3713 -10.2589  -2.6545
#>   -0.6482  -0.8752  -5.0336  -1.9508  -0.0299
#>    5.4970   3.2998   0.7064  -5.0501  -3.0330
#>   -1.3835  -1.4945  -2.4936  -1.3149   0.5554
#> 
#> (1,4,.,.) = 
#>  -0.5984  -7.8747   3.5888   1.9737   0.0114
#>    2.2536   9.8432  -0.3910   1.9229  -1.5662
#>    6.0027  -2.5451 -10.7711  -0.3985  -7.9943
#>    0.9682  -2.0505  -1.0461  -1.8087   2.1040
#>   -0.3964  -0.5104   5.7304  -0.6616   3.3541
#> 
#> (1,5,.,.) = 
#> -0.5408  1.7089 -4.5011 -7.3198 -6.3117
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
