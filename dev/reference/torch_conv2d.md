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
#>   1.2031  -1.5014  -5.1455   3.4322   4.3129
#>    1.5082  -9.5071   5.6057  -1.9270  -3.1649
#>   -6.1927   1.6675   3.5415  -4.2492  -0.6641
#>    4.8017   2.5589 -13.0111   1.5012   4.8619
#>    2.2197   0.0777  -2.8144  -0.9361   2.0327
#> 
#> (1,2,.,.) = 
#>  1.5813  1.1593 -2.0933 -4.9668  0.2751
#>   4.4542 -9.3832 -9.7499  1.5616 -8.3484
#>   1.8753  6.7435 -0.5895  0.1382  4.4091
#>  -5.7609  0.4283 -3.9771  5.0224  1.5166
#>  -1.8417 -1.2545 -0.6505  2.4134  6.7571
#> 
#> (1,3,.,.) = 
#>   0.3613  -0.8766   1.7205   0.6788  -2.1737
#>   -2.5645   3.1898   2.5295   1.1990   5.2068
#>   -1.5103  -7.2022  12.1876   4.4159   3.5224
#>   -3.1113   0.1753  -0.0661   2.5567  11.3838
#>    2.0249  -5.2384  -5.6264  -6.6734   3.7518
#> 
#> (1,4,.,.) = 
#> -0.9849  0.8477  4.0899 -6.3677  5.7213
#>   2.0448 -6.2019  5.5996  2.4730  0.6235
#>  -7.2579  1.5987  2.2282 -2.5918 -4.9906
#>  -1.6265 -0.3110 -8.7226  8.9274 -5.2061
#>  -3.7868  6.1329 -1.0421 -5.3373  0.4244
#> 
#> (1,5,.,.) = 
#>  -0.8090  -2.7442  -3.4218   4.4911  -3.4284
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
