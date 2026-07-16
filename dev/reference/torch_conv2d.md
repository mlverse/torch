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
#>  12.1949   8.0660  -4.0949   4.1586   3.8330
#>    4.6509  -0.5645  -5.7915   6.2783   3.0576
#>   -7.8635 -11.5405   5.7816   8.5692  10.8278
#>    1.3839  -9.1508  -2.7997   0.9149   5.3933
#>    6.8200  -3.7045  -7.1803  -7.2847  -5.1896
#> 
#> (1,2,.,.) = 
#>  -1.7765  -3.1013  -1.8179  -3.0136  -1.3283
#>   -3.9424  -2.6849  10.2927  -4.7221  -4.3897
#>   -5.0847   5.1449   7.8225   1.5271   2.5579
#>   -2.5542  -9.3846   6.5066  -2.2689   2.4494
#>    4.0935  -7.7236   0.9586  -1.1407  -3.5347
#> 
#> (1,3,.,.) = 
#>   5.6734  -1.7998  14.8897  -0.9402  -0.8966
#>   -5.8591  -6.7184  -2.9751   0.2183  -2.8918
#>   -6.0510  -3.8281   4.5765  -0.3008   1.9805
#>   -2.8105 -10.7446   6.4512  -2.3884   0.2232
#>   -6.0245  -0.3134   5.3145   3.6052   2.1690
#> 
#> (1,4,.,.) = 
#>   4.8159  -2.6019  -0.4895   4.8655  -3.8520
#>   -2.7753  -4.2411   2.3946  12.7730  -7.1885
#>    9.5690  -0.8858  14.1576   2.3874  -1.4187
#>   -1.5629 -14.1560   5.2675   9.8522  -0.8317
#>   -7.4873   2.8202  -0.9306   0.2042   0.8072
#> 
#> (1,5,.,.) = 
#>   1.7801  -1.1097   2.8257  -5.1361   1.3695
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
