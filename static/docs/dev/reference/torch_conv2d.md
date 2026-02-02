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
#>   1.2145  3.2266 -0.6520  1.8607  3.1125
#>  -5.5660  4.3597 -4.8524  3.8211  2.3415
#>   3.0680 -7.3036 -9.7745 -8.0851 -1.6391
#>  -2.1491 -4.2902  3.0597 -1.5262  3.5174
#>   1.5301  5.4356  8.2938  9.7848  3.6139
#> 
#> (1,2,.,.) = 
#>   0.4186 -0.8035  1.1692 -2.6034 -0.7030
#>  -1.3910 -3.1711  0.4133 -2.2040  7.0958
#>   3.6643  2.9151 -0.8167 -0.3871 -3.2655
#>   2.6457  3.9005  5.8789  0.5025 -3.2274
#>   1.5036 -0.6156  3.3000  4.2567 -2.0661
#> 
#> (1,3,.,.) = 
#>    5.5225   5.8733  -4.6448  -5.4462  -0.3963
#>   -4.6622  -3.4326  10.5224  -1.8093   8.5294
#>    8.3879  -2.3798  -2.0116   1.3364   7.6309
#>   -4.6118   3.1937  -2.6346 -15.3212  -9.4448
#>    3.7282  -3.7383  -2.6558 -11.3032  -0.0395
#> 
#> (1,4,.,.) = 
#>  -1.9969  0.9805  0.5533 -1.9875  0.7841
#>   5.4976  3.1047 -1.2370 -0.2532  0.3199
#>  -3.5779 -11.7177  5.7205 -0.1997 -0.8511
#>   6.2350  1.3667  1.0726 -6.4547 -1.3319
#>   2.7234  0.0777 -3.6455 -1.8080  0.2745
#> 
#> (1,5,.,.) = 
#>   2.9464 -4.2508  2.2166 -0.5030 -0.5020
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
