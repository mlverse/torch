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
#>    3.8359   1.2122   3.3114  -0.5665   1.5156
#>    8.2689  11.9755  -2.6721 -15.6166   0.5523
#>    1.0460   3.1959  -2.0557  -0.3057   3.4973
#>    4.8693  -0.1943 -14.2910   9.3664   2.3221
#>   -3.6883  -7.3369   1.0548   0.0634   1.3225
#> 
#> (1,2,.,.) = 
#>   0.2810 -3.3858 -3.1901  6.4439  6.0047
#>   1.2783 -1.1843 -2.7846  7.6466 -5.0794
#>   7.1127  1.4253  4.5794  6.5209 -5.9649
#>  -13.5861 -2.3432  4.6965  7.4533  0.5725
#>  -0.3464  1.3571  1.1418 -3.8660  2.9801
#> 
#> (1,3,.,.) = 
#>  -1.3737  1.6021 -0.4088 -1.0825 -0.5795
#>   2.1947 -5.8162  3.1493  2.4984 -3.4504
#>   4.9893  4.5474 -0.4007 -4.4411  0.8276
#>  -0.4254  1.8769  4.2033 -5.8921 -0.9608
#>   3.3092  2.0865  0.4868  4.0904 -1.6530
#> 
#> (1,4,.,.) = 
#>   0.7067  4.2834 -0.2758  9.1198  1.4746
#>   4.0473  3.9884 -0.7229  1.8914 -0.8427
#>   4.6823  1.6191 -1.3898 -3.2333 -0.7197
#>  -2.7382  6.1969 -3.3694 -1.5765 -0.9735
#>  -2.6183 -1.5031 -8.7348  6.8357  5.6085
#> 
#> (1,5,.,.) = 
#>  -3.9960 -1.4804 -2.4767  9.0512  1.1885
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
