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
#>   2.3074  -1.3843  11.4677  -3.4200   1.1090
#>   -3.1873  -0.2224  -0.2060   2.5007   1.5691
#>  -12.9855   4.5398   1.6852   1.8786  -6.3281
#>    3.7327  -4.4923  -4.1614   0.7875  13.1017
#>    2.6385  11.0907  -3.1093  -1.8359  -1.8311
#> 
#> (1,2,.,.) = 
#>  -3.8802   6.4399  -6.2644   8.6440  -2.2418
#>   -1.1181   2.5953  -5.7413   2.3324   0.9463
#>    4.6583   2.1024  -2.4289 -10.5394  -0.2858
#>    4.9803   0.9206  -1.4650   6.3479  -6.1781
#>    4.7756  -6.2632   7.3449  -1.8496   3.4680
#> 
#> (1,3,.,.) = 
#>  1.9103 -3.0638  0.4474  2.2016 -2.6531
#>  -8.2362 -3.1283 -5.0070 -2.0570  0.2168
#>   1.0958 -6.1049 -2.2297  0.2079  3.1899
#>   2.0700 -1.6610  0.2327 -6.6856  2.8783
#>  -3.0510  2.7751 -2.1340  3.3816  1.1306
#> 
#> (1,4,.,.) = 
#>  -4.7448  -3.5143  -3.7678  -2.2836   2.7021
#>   -0.0760  -7.9468  10.6908  -3.6892   2.3226
#>   -5.6671  -7.6776  -6.6641  -0.2053   4.7083
#>    3.2112   6.5959   1.0057  -7.1523  -0.5698
#>  -10.9347  -2.1493  -2.5684  -1.7632   2.5031
#> 
#> (1,5,.,.) = 
#>  -0.9051   3.9532  -1.0802  -2.9087  -3.4871
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
