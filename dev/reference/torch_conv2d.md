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
#>  -1.9748  0.0749 -0.5234  0.7218  0.6989
#>   2.4200 -5.3939 -2.0066 -6.8630 -4.5174
#>  -5.9031 -6.4275  1.2848 -2.2163 -4.3497
#>  -3.2782 -3.7040 -3.4110 -6.3259  5.3205
#>  -3.3415 -1.8898  4.7361 -5.6123  2.0475
#> 
#> (1,2,.,.) = 
#>  -2.3365 -4.9736 -7.2308  5.7577  1.7574
#>   3.4409  2.2554  0.5882 -2.1846  3.3051
#>   0.8736 -1.9606  3.8815  4.0974  5.7779
#>  -0.7087 -3.7679 -3.3017  1.0692  1.2323
#>  -4.5512 -3.9012 -0.5685  2.9639  1.4348
#> 
#> (1,3,.,.) = 
#>   0.9585 -4.1131 -2.4633 -2.8601 -4.3775
#>   1.4149  1.5000 -11.4281 -1.0127 -1.5330
#>   3.9317 -7.5479  4.4520 -10.4420 -1.5716
#>   0.4817 -2.2843 -6.2946 -6.1284 -0.3954
#>  -3.0689 -0.4488  0.5274 -0.1431  2.9993
#> 
#> (1,4,.,.) = 
#>    2.7139  -4.0800  -7.4671  -4.1644  -5.7155
#>    1.9679  -8.6360  -6.6461  -5.4721   4.4588
#>   -2.9161   1.3139  -3.8380  -5.9525  14.9206
#>   -2.1635   3.2388   4.5610  -5.7484   5.8210
#>   -4.4111  -0.8042   6.6141  -0.1406   7.0389
#> 
#> (1,5,.,.) = 
#>  -0.2313  2.1146 -6.0281  1.1759 -2.9854
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
