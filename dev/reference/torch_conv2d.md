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
#>   3.5548  -8.4102  -2.8216  -7.5273   3.2410
#>   -0.6360   0.0797  -5.0383  19.8143   1.7221
#>    0.3325   0.0621  12.6352 -12.2708   6.7849
#>   -1.3505  -1.1081   3.8536   2.2409  -8.0770
#>    4.1682  -3.0444   7.7475  -6.0985   1.2801
#> 
#> (1,2,.,.) = 
#>   7.3346  -3.7403  15.0003  -7.5891   2.4192
#>    3.6353   8.1896   4.6990   0.0321  -3.1074
#>   -3.6866  -8.4336  -8.3313 -12.7215  -4.7092
#>   11.6083  -8.5146  -6.7734   7.1516   4.5164
#>    0.8264   2.9278   0.0267  -1.5433  -3.1445
#> 
#> (1,3,.,.) = 
#> -1.4940  0.8881 -2.1025  3.4320  3.0144
#>   2.3310 -6.5949  2.1715 -1.8614  0.7480
#>  -0.0283 -3.0477  2.3964 -2.2447 -2.0465
#>  -2.4904  0.7088 -4.5892 -8.8169  2.2351
#>   2.2860 -2.1824  3.1799  1.7226 -4.1097
#> 
#> (1,4,.,.) = 
#>  -3.2328   5.9632  -6.4024   6.3963  -5.5028
#>   -2.2127   1.0872  -0.2685  -2.6981   9.7888
#>    0.9126  -5.7230   6.8638  15.3232  -4.5725
#>   -3.9659  -4.9383   2.4420   3.8450   4.2381
#>    5.6672   1.8870   0.4370  -4.9524  -2.2869
#> 
#> (1,5,.,.) = 
#>  -0.0918   7.1954  -6.8529 -11.0311   7.3726
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
