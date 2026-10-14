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
#>  -3.7739   7.1818   8.3630  -0.8969   1.4034
#>    3.0478   1.4923 -10.3969   1.4913  -4.1711
#>   -5.7749  -5.7054   6.3153   0.2591   0.1905
#>   -1.7667  12.3693   4.2131   7.2588   5.5068
#>   -0.0077   1.1661  -2.2596   4.4521   1.9563
#> 
#> (1,2,.,.) = 
#>  0.5486 -2.1412 -0.2048  1.9250  1.8976
#>  -0.8944 -2.8047 -4.5364  0.4804  7.5114
#>   0.0198  1.6887  3.0651 -1.7589 -2.7570
#>   0.9944  4.0021 -1.5556 -3.8419 -3.9083
#>   0.8799 -2.4085  0.3185  0.6818 -1.3174
#> 
#> (1,3,.,.) = 
#>  -0.6811   3.2634   0.5777  -6.7912  -2.9156
#>   -6.5267  -1.6064   0.9295  -4.7916   1.4719
#>    1.0803  11.8093  -0.9393   7.6847   1.6012
#>    1.5787   2.4917   1.1408  -5.3452 -16.7592
#>   -2.0847   1.7485 -12.6283  -7.7001   0.8071
#> 
#> (1,4,.,.) = 
#>  3.7839  1.7469 -2.3845  0.0977 -2.4887
#>   5.0817  4.5761 -0.3152  9.4550  4.5892
#>   4.6582  0.2926  1.0384 -7.8964  1.6948
#>   4.0869 -2.0545 -9.5299 -5.4699 -1.1150
#>   5.3179 -5.7672  7.2416 -9.4073  1.9370
#> 
#> (1,5,.,.) = 
#>  -6.0451   6.5717   2.8488  -4.1951  -2.9939
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
