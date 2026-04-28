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
#>   2.0094   2.8533  -2.0494   1.4007   2.1542
#>   -3.4297  -5.7845  -7.8564   9.3719  -2.6190
#>   -3.9127   0.9959  -9.8412 -11.7308  -6.6251
#>    3.5501  -1.0683  -0.5073   8.0510   4.9753
#>   -7.9575  -3.7071   4.9089   2.4704   3.0363
#> 
#> (1,2,.,.) = 
#>   1.3059 -17.3099   9.7132   0.1655   0.6252
#>    9.1210   4.5371   7.6192 -22.1750  10.2842
#>    6.5076  -4.0999   1.4633   9.5127   7.2622
#>    8.2651  11.9008   2.1590 -16.1687  -6.2833
#>   -1.1522   7.1913   0.6463  -4.7468  -0.8947
#> 
#> (1,3,.,.) = 
#>   5.5478   3.7780  -9.4465  -1.1749  -4.3557
#>   -0.3907  -0.7321  -5.6572   8.5359  -2.9705
#>    3.8857 -13.5505  -4.1293  -7.1545  -6.3911
#>   -3.1960  -1.2129 -19.0033   0.6156  -0.5560
#>   -3.7831  -7.1141  -4.9092   8.9800   8.7415
#> 
#> (1,4,.,.) = 
#>  -5.9278   0.5652  -7.8906  -2.1031   0.4459
#>   -4.3724  -1.2241   3.8079  -8.3864  -4.3934
#>  -19.2379  -9.0775  18.8162  -4.6498   1.0547
#>   -1.2402 -11.2472 -15.6541  -4.7353   4.2268
#>    2.5652  -6.5957  -1.9859  -3.5301  -2.2501
#> 
#> (1,5,.,.) = 
#>  -1.4377   3.9754  -9.8275  -1.5809   0.8237
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
