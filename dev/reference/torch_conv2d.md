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
#> -9.2703  0.4822  5.0372 -0.9067  5.6747
#>  -3.6912 -0.0576 -6.6639 -5.3846  4.1407
#>   0.3282 -3.4541 -8.6039  7.0452 -3.4856
#>  -0.0534 -0.3494  3.9150 -4.1120  1.5405
#>  -0.3744 -4.0756  4.0383  5.4671  3.4416
#> 
#> (1,2,.,.) = 
#> -3.8993 -9.7019 -2.3834  0.1090  3.9247
#>   3.3972  0.4011  0.8811  3.1431  1.2112
#>   2.2538 -0.4541 -5.8299  3.8781 -1.0934
#>   1.2227  0.9790 -1.1068 -7.7400 -4.2205
#>  -4.8340  1.2034 -1.7144  3.0508 -0.2146
#> 
#> (1,3,.,.) = 
#>   0.4341   2.9574  -0.9156   3.4973   1.7604
#>    2.1557  -0.4188 -12.1838   5.8285   8.6257
#>    1.8896  -0.5403  -2.0780   0.7664  -7.9669
#>   -4.4059  -4.2888  -2.4671  -1.6244   2.7240
#>   -2.3127  -0.6242  -0.0679   0.4521   4.2966
#> 
#> (1,4,.,.) = 
#>   8.4345  -2.9157  -5.2211   0.9572 -10.1273
#>    5.8342   7.8742  -1.5954   9.7752   8.2999
#>   -5.7944  -1.5789  -0.7950   0.6596  10.9358
#>   -1.9155  -2.1556   0.5022   1.9767  -7.9336
#>    1.2266   4.0835  -2.4726   3.5822   0.9777
#> 
#> (1,5,.,.) = 
#> -0.1675  6.3776 -1.0563  0.2544  5.8013
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
