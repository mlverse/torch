# Conv2d

Conv2d

## Usage

``` r
torch_conv2d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias tensor of shape \\(\mbox{out\\channels})\\. Default:
  `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a tuple `(padH, padW)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv2d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

See
[`nn_conv2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
filters = torch_randn(c(8,4,3,3))
inputs = torch_randn(c(1,4,5,5))
nnf_conv2d(inputs, filters, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#>  0.4264 -0.9913 -6.5710 -4.9634 -4.5257
#>   4.6579 -6.1985  2.1786  0.3032  5.6471
#>   6.6780  4.6243  0.2879  0.2542  6.1376
#>   5.9210 -1.8147 -2.5894  2.6215  0.2002
#>  -2.9284 -0.4655  5.5235 -1.2176 -1.5908
#> 
#> (1,2,.,.) = 
#>  11.4002  -4.2296   4.6720 -11.3672  -0.6068
#>   -1.7458  -9.3352  -2.5666   1.3572   8.9797
#>   -4.5317   6.9880  -6.7124   0.2220   1.8069
#>   -2.8871   8.7341   4.2452  12.1192   9.4139
#>   -1.6680   0.7633  -2.3325  -5.7903   0.3332
#> 
#> (1,3,.,.) = 
#>  -3.5790  -2.5344   2.1556   4.5384   2.3326
#>   -1.2386   8.2966  -6.5474   0.3691  -1.1368
#>    7.6280   1.5941   2.4443   3.2075   5.2211
#>    5.0012  -1.0140  -0.4268  -9.1560  -3.5791
#>   -0.2845  -6.6975  -1.0818  10.1144   5.0444
#> 
#> (1,4,.,.) = 
#>  -3.0699  -1.1324  -1.9376  -4.8574  -2.0846
#>   -4.4140 -12.7815   3.6441  -0.5620  -4.8132
#>    2.2358   0.7579  10.4239  13.8588  -1.4542
#>    5.1921   0.2090   8.1117  -6.0791  -6.8210
#>   -2.6933  -1.2987   1.0399  -8.2842  -3.2056
#> 
#> (1,5,.,.) = 
#>   1.6303   1.5001  -4.7579  -1.9953   3.8020
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
