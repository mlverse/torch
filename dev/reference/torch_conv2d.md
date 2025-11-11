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
#>  -4.3736 -5.8401 -9.0715  0.8923 -3.2105
#>  -5.0911  2.6026 -4.5317 -4.5789 -5.2915
#>  -3.5516 -0.4578 -2.4796  2.3332 -4.5054
#>  -1.9728 -8.2213 -4.0019 -8.6201 -5.8161
#>   3.7882 -13.8540  1.4648 -7.9424  1.7526
#> 
#> (1,2,.,.) = 
#>    1.7732   2.8874  -6.2255   7.1877   3.3674
#>  -13.9774   2.5938   1.4898  -3.9382   2.5762
#>   -0.7986   7.1687  -6.1149   1.3374   0.4070
#>    0.2519  -0.2880   3.6029  10.3392   0.3873
#>   -3.5559   3.1294   9.4481   3.0309   0.8277
#> 
#> (1,3,.,.) = 
#>   -5.5538   6.3881   5.1706   5.2467  -1.0742
#>   -2.2380   9.3735  -5.2379   7.1493  -0.5562
#>    5.0256   9.6382   5.8596   7.0712   2.9867
#>   -3.0961  10.9960   8.4250   3.6126  -7.9889
#>   -0.6294   4.5177   1.5222  -0.0762   3.6771
#> 
#> (1,4,.,.) = 
#>   2.6036  1.1168 -3.5730 -4.3493 -8.6624
#>  -3.6132  8.1984  0.0081 -2.5646  4.0507
#>  -0.7796  4.4953 -1.5150  0.7078  1.3731
#>   0.7935  0.7297 -6.3034  0.1625  4.0986
#>  -4.3316  4.5816  3.6835 -3.6908  2.5042
#> 
#> (1,5,.,.) = 
#>    2.1675   7.4384   4.5461   0.9719   0.1552
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
