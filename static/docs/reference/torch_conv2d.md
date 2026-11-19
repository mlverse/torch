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
[`nn_conv2d()`](https://torch.mlverse.org/docs/reference/nn_conv2d.md)
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
#>  -1.3486   0.4085   3.3333   2.0532   0.7807
#>   -1.7282 -14.1785   2.6349  -0.6882  -2.3638
#>   -3.9599   2.7103   0.8450  -2.6567  -5.9306
#>   -1.1641  -4.8645  -4.7126   2.3689   1.4656
#>   -4.1110   3.0686   0.8506  -1.7300  -7.2018
#> 
#> (1,2,.,.) = 
#>  -3.0144   3.3389   1.3247   6.8937  -6.2946
#>    3.4248   8.3310   5.2818 -13.1376  -1.5771
#>    2.2401  -3.4850   0.1974   5.1599   0.3266
#>    2.2300 -10.4043  -6.7522   1.4970   2.9760
#>   -1.0811   5.6528   1.5911  -9.0549  -0.2586
#> 
#> (1,3,.,.) = 
#>  0.2655 -1.2779  7.6327 -5.4033 -1.6440
#>   3.1058 -6.3207 -6.8231 -1.7329 -1.2255
#>  -4.4336  3.4291 -3.2604 -1.5875  0.9993
#>   2.1536 -3.1284  0.4532  7.2594 -5.3957
#>   8.6942  1.1472  0.7643  6.4228  5.8589
#> 
#> (1,4,.,.) = 
#>  4.5213  5.5620 -8.4553 -6.1660 -2.1761
#>   0.6776 -5.8836 -2.6247  0.3690 -0.5036
#>   2.8252 -7.5309 -3.9001 -0.6948 -2.4759
#>  -0.0147  2.2886  8.8800 -7.0130  3.0946
#>   3.1609 -5.6554 -2.7962 -3.8288  4.6225
#> 
#> (1,5,.,.) = 
#>   2.2414  -1.3021   3.1131  -3.6390   6.6006
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
