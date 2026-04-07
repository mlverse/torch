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
#>   3.5759  -7.7916   5.4082   1.1014   4.2936
#>    4.0306   7.8061   2.5526  14.8824   3.8621
#>    4.1731   8.5116   2.7434  11.4281  -5.5303
#>   -3.0863   3.0753  14.2342  -5.0711   1.0721
#>    0.0303  -1.2976   0.8648   6.5164  -2.8988
#> 
#> (1,2,.,.) = 
#>   1.8780   0.4021   2.3351   1.6270  -6.6347
#>   -5.6530  -1.7965  -1.1938   4.1609  -9.0069
#>   -2.8103  -8.5841 -11.5762  -9.6324   8.1574
#>    9.0434  -8.4953   3.3356  -3.1455  -4.4410
#>    0.9770  -1.5391   1.2233   4.7420   1.8243
#> 
#> (1,3,.,.) = 
#>   4.1855  10.4694   4.5540   0.7833   0.8409
#>   -1.3712   0.8977   7.0531  -5.8366  -5.7949
#>   -3.9010   0.4060  -3.1570   6.6372   5.8830
#>    1.7114   6.1563 -15.7400  -6.2971   0.9376
#>   -3.8779  -1.9110  -3.2764  -4.4735  -0.5322
#> 
#> (1,4,.,.) = 
#>   5.0201  -4.8792  -2.6881  -2.5388  -0.3793
#>   -3.6609   6.3556 -10.5513   2.0402 -12.1331
#>   -4.9053  -1.4460   0.6589   8.0207  -6.5822
#>    3.0340   7.6684  11.0325   7.2079   1.6825
#>    3.0399   6.0410   5.6957  12.9183  -0.9641
#> 
#> (1,5,.,.) = 
#>   5.1190 -11.8954   6.2556   8.2441   2.3767
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
