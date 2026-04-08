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
#>  -0.2441   5.6277   1.3895   4.6277   0.1536
#>    5.3308  -2.8613   3.9547  -1.4519  14.2636
#>   -0.7168   6.2401   7.9636  -2.7165  -2.8313
#>    8.7927 -11.1124 -17.5763 -13.7344   3.6628
#>   -5.5791  -3.1087   2.8706  -1.5205  -1.7449
#> 
#> (1,2,.,.) = 
#>  -9.1757  -1.6461   0.9509  -3.7492   1.0748
#>   -0.4477  -1.0588  -2.9669  -1.3299   4.7039
#>   -8.4019  -8.1554   2.1192   9.8209  -1.0998
#>    2.8850  14.1558   3.4047   2.6967  -1.4434
#>   -2.1239  12.6478   2.0730  -4.1604  -5.9739
#> 
#> (1,3,.,.) = 
#>  -5.5956   7.1325  10.2515  -0.6700   3.4541
#>   -3.1215   3.9792   5.0009   0.4288   3.0272
#>    2.8228  11.1798   5.5454   3.7051  -0.2378
#>   -2.7543   0.0904   7.1191  12.1934   1.1240
#>    8.8634   7.0123   6.5924   1.3756  -4.9526
#> 
#> (1,4,.,.) = 
#>   8.1043   9.6971   1.9878  -5.2959  -4.9882
#>   -4.5464   0.5942   1.9923   6.4378   3.4133
#>    9.2546   5.6982   5.0927   6.6920  11.5847
#>   10.8607  -8.7167 -13.4346  -1.9491  -2.1579
#>    4.6117   6.4670   0.6936 -12.0250  -9.5194
#> 
#> (1,5,.,.) = 
#>  -0.9589  -9.2431   1.4919  -2.9971  -2.6340
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
