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
#>  -0.0118 -12.1627  -4.3912   4.1945  -0.8059
#>   14.9560   6.8666   6.1851  13.1924  11.7078
#>    4.2834  -1.2471  -0.1119   8.0630   5.1299
#>    3.6115   0.2952  -5.0725  -7.9994  -5.8267
#>   -5.8186  -2.4460  -6.2080   0.0707  -4.6674
#> 
#> (1,2,.,.) = 
#>  -6.3220   1.1761   6.6881   4.1185  -4.5394
#>    2.8165  -2.3499   4.4800  -3.3234   0.5893
#>   -7.8190   8.9159  -6.6757  -7.6413   1.9243
#>    7.0934  -2.3163  11.1644  -0.5546   2.4508
#>   -2.2485  -0.1321  -3.5919   5.7284  -1.0996
#> 
#> (1,3,.,.) = 
#>  -5.4035  -5.0141   1.1487  -1.9425   6.0739
#>    7.4063  -4.1227  -2.3941  -4.5168  -0.2121
#>    7.3922  -1.2082  11.4349  -0.5043  -3.5748
#>   -3.9529  -5.9508  -9.8480  -1.4838  -1.2599
#>    2.0999   5.1489   4.1599  -0.6265   1.4702
#> 
#> (1,4,.,.) = 
#>  -5.2050  -4.1797  -5.9926  -4.5270  -1.9651
#>    5.6826   9.2101  11.2357   0.2154   2.0628
#>    7.8477   9.3498   8.9365   2.7714   4.1406
#>    3.4841   6.8148   3.3021  -2.7074  -2.6876
#>   -4.8397   2.7972  -0.8172   0.1030   0.4813
#> 
#> (1,5,.,.) = 
#>   7.7697  -1.8763  -1.3568  -5.5797   2.2351
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
