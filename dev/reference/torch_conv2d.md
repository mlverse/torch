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
#>  -3.5011  -1.7023  -4.3256  -0.8177   0.8894
#>   -2.3287  10.1253   2.1124   6.4380  -0.5311
#>   -7.7862   0.8598   0.1259   5.2643  -1.3725
#>   -7.0887   0.5882   5.1209  -1.5488  -3.4447
#>   -0.1954  -3.3248  -2.3056   1.2789   0.4409
#> 
#> (1,2,.,.) = 
#> -3.9419 -8.9354 -0.0517 -3.8203  5.6546
#>  -0.8218  2.8203  2.4546 -2.5197  4.4002
#>  -5.0779  0.5020 -3.0611  3.3222  1.0390
#>  -4.4510  3.6503 -7.3572 -0.8319  1.9290
#>  -0.3172 -1.8853  0.7347  5.1338  1.8313
#> 
#> (1,3,.,.) = 
#> -1.4978 -3.6488  2.8370 -9.9398 -3.8339
#>  -1.2701 -3.5860  4.7914  4.2645  4.7550
#>  -6.7860 -3.3249  4.5060  6.2335  3.9014
#>  -3.7083 -3.5586 -2.5757  6.0963  6.4581
#>  -7.7373 -8.8896 -7.7926 -1.5573  4.3929
#> 
#> (1,4,.,.) = 
#>   1.7759  -6.5634  -1.7111   0.5743   3.0337
#>    0.1608  13.0423  -2.8491  -5.6038  -4.9589
#>    4.5847  -1.7490   6.4339   2.4337   4.0550
#>    4.8144  -2.4940  -2.1953  -5.9287  -0.9334
#>    1.6640   0.7387   7.1802   2.3653   0.0818
#> 
#> (1,5,.,.) = 
#>   1.5286  -4.0623   5.4853  -2.3483   0.8364
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
