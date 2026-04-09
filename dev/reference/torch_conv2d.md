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
#>   2.1222  -2.7653  11.2542  -4.6084   3.9998
#>   -6.3122 -10.6583  -1.7606  -1.8625  -1.7937
#>    2.4961   6.7249  -8.1351  -7.8439   0.3159
#>   -0.2284  -3.4744   6.0355   0.0225  -1.2345
#>   -2.7014  -2.8365   2.3349  -0.8899   2.8595
#> 
#> (1,2,.,.) = 
#>   4.8560  -8.2718   5.0916  -3.9784   7.0149
#>    2.4732   1.0436  -1.4815   1.4581  -0.2114
#>   -1.3924   2.4008  -1.0033 -10.5520   3.3468
#>    5.7581  -1.2003   2.1181   0.4974  -0.1047
#>   -3.2567   3.1700   2.8247   1.6759   0.8648
#> 
#> (1,3,.,.) = 
#> -5.6827 -0.5816 -0.5226 -1.2263  3.0543
#>   1.3905  6.9743  0.1375 -0.5134  1.6099
#>   1.1932  1.5287 -6.2001  3.1556 -0.1611
#>  -2.7367  6.5764  4.2290  0.8360  1.6711
#>   0.3846  1.3589 -3.6392  0.7884 -3.0486
#> 
#> (1,4,.,.) = 
#>  -0.0587  -3.8191  -6.1412   6.4901  -0.0858
#>   -6.6581  -7.7027  -5.3511  -3.2395   0.3631
#>    9.0094  13.6979   1.5288   7.3518   3.4261
#>    0.6184  -6.4517  -0.3202   7.1981  -2.0170
#>   -2.9548  -6.4362  -6.4772   0.8167  -3.2666
#> 
#> (1,5,.,.) = 
#>  -2.3229   7.5384   2.6953   0.8003   2.6828
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
