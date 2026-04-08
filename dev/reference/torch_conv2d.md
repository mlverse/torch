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
#>   0.5596   2.8604   0.5290 -11.7228  -4.2921
#>   -3.4853   0.8346  12.6855  -3.3340  -7.9319
#>   -0.0215  -6.6424  -3.8414   3.8282  -2.6787
#>    3.2322   6.7627  -3.5089 -16.2409 -11.7177
#>    4.5964   5.2443   1.1225  -6.8813  -4.6037
#> 
#> (1,2,.,.) = 
#>   0.3649  -1.6948  -4.9920   6.8443  -4.9628
#>    2.2729  -7.1587  10.8859  -3.1224   4.6253
#>   14.0555  -0.7514  -6.7290  -7.3565  -4.5011
#>   -4.2865  -6.5859  -2.6125  12.9459   6.5846
#>    0.6598  -3.7671   0.2888  -3.9837  -3.4591
#> 
#> (1,3,.,.) = 
#>  0.9568  6.8694 -0.6974  4.8196  6.7206
#>  -1.3338  4.2417 -0.5096 -6.9522 -2.7300
#>   0.7662 -8.4089  3.0979  5.4421 -1.5877
#>   0.7395  4.0888 -1.7013 -3.4016 -3.9409
#>  -2.4193 -6.3183 -6.5976  5.3272  6.0447
#> 
#> (1,4,.,.) = 
#>  -7.6489   2.7516  -1.5919   8.3837  -1.2134
#>    1.8946  -1.2093  -6.0217  -7.7807  -7.0724
#>    3.6545   7.9567 -13.5195   1.6122   1.7214
#>   -4.1545  -3.0224   9.0008   3.1003   1.2519
#>   -1.4278   1.5977   4.5063  -4.0722   0.3279
#> 
#> (1,5,.,.) = 
#>   1.8640  -7.0419   3.7577  -9.0877  -0.1084
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
