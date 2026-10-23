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
#>   3.9658   4.8691   4.5811   2.6811   0.2058
#>   -5.8529  12.9259 -11.1790   0.3694  -7.2111
#>   -1.8980   6.8560  -2.6308   2.6326   1.1960
#>    6.9542  -1.2525   2.4893  -5.8896  -8.1061
#>    4.4524   4.1722   7.7023  12.4507   2.0699
#> 
#> (1,2,.,.) = 
#>   1.6568   3.4172   3.2288  -1.6657  -9.2193
#>   -1.6550 -13.9874  -1.7675  -2.1790  -8.7159
#>    0.2711  -4.4394  -7.0770  12.7142  -3.3479
#>    6.4538  -1.1784   3.9359   0.5611  -8.9572
#>   -3.8767   4.6454  -0.2212   6.6601  -1.4576
#> 
#> (1,3,.,.) = 
#>  -1.8245  -6.8602   0.7763   1.4578   2.3653
#>    8.3881  -1.6278   6.9808  -9.0598  -7.3479
#>   -3.0179  -5.7291   9.8181   3.8777   1.7965
#>    0.9470 -16.1550  -6.7534  -8.6746  -5.0185
#>    6.5880   4.2140   5.8888   9.5147  -6.2290
#> 
#> (1,4,.,.) = 
#>  1.8809  2.3547 -4.7888 -4.6078  0.9118
#>  -9.0237 -5.2838  9.7635 -1.9242 -1.2521
#>   3.1729 -8.4110  3.5025 -7.4121 -0.3216
#>  -0.2074  4.0840  2.4024 -3.1633 -0.6460
#>  -4.6501  1.8981 -0.0674 -6.2687 -1.2247
#> 
#> (1,5,.,.) = 
#> -4.8106  0.2551 -3.6258 -5.1444  3.4947
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
