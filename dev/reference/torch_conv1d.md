# Conv1d

Conv1d

## Usage

``` r
torch_conv1d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a
  one-element tuple `(sW,)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a one-element tuple `(padW,)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv1d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See
[`nn_conv1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

filters = torch_randn(c(33, 16, 3))
inputs = torch_randn(c(20, 16, 50))
nnf_conv1d(inputs, filters)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 8  -4.6228  15.7809 -15.9251  -5.3153  -0.0863   4.1070  -6.8000   6.4122
#>   10.9606  -4.3079   5.2037 -10.5183  -0.7074   3.9412  -1.1219  -5.8050
#>   -2.2813 -14.5727   0.3282  -7.5982  -5.6861   0.4882  -2.9577 -13.9932
#>   -4.8726   8.2854   1.9919  -9.0227 -10.0740   4.2750  -4.4779  -8.4830
#>   13.7344 -10.4289  13.8733  -9.0457   6.8925  -6.8839  -2.5413  -2.8249
#>    4.6376  -2.9774   7.9928  -1.8510  -0.2627  -3.7934  -9.8209   2.8506
#>   -4.3424   4.7108   6.1420  -7.8608  -0.5656 -10.9586   8.0347   4.8926
#>   -1.8858 -12.4694   5.2032   2.5385  -3.6672  -4.5477   1.4178   2.2699
#>   -2.4128  -0.3223   3.9530   2.4242  -2.7235   4.4876  -1.2774  -4.7794
#>   -7.3203  -8.4798  -3.4821   2.6002  -3.8897  -4.9874  -5.6384   3.7527
#>   -5.2904   8.5070  -9.2510  -7.6178 -10.1729   6.5965  -7.0781  -9.4482
#>   -7.7113  -4.1335  -6.5675   8.8770   0.6349  -3.5348   8.2322  -3.5894
#>   -6.7801   5.2458   2.2042   7.6178   6.1011  17.4437   2.2585  -5.9245
#>   -6.4801   2.7697  12.5791  -5.8959  -6.4215   8.4810   7.2456  10.4130
#>   -0.8914   8.7967  -3.3705  -8.9585  -5.4462   2.0541  -9.8307  -1.1749
#>   -6.4329 -13.7015  -1.9435   7.9106   1.7471   7.5073   2.2788   9.9393
#>    0.7214  10.8880   8.5698  -4.3428  -2.0365  17.0103  -6.1187  12.6525
#>   -2.7777   0.1171  -1.1247  -6.9504  -2.6878   1.2582   3.1024   6.2062
#>   -2.3008   8.3513   8.0923   3.0471   0.7432   8.2327   0.4625   3.2953
#>    4.0762   4.2892  -0.8523  -2.0852  -5.8032   4.3627  -0.4325  -0.8593
#>    9.5319   8.2271  -6.3278  -2.1550   4.0404  -2.2690  -6.8314  -5.9287
#>   -7.4399  -8.3528  -8.8242   1.8421   1.4230  -5.2997   5.5028   5.5262
#>  -12.1824  -1.5476  -8.4200   0.2611  -2.6570  -1.3076  -2.5352  -2.4189
#>    4.8189  -3.6989  15.6136  -9.2577  -2.0670  -9.9229   1.9002  -5.8282
#>    5.8068 -14.5490  -3.3608  -1.3245  -3.8299  -0.3018   4.3946  -5.4867
#>   -4.0648  -6.0784  -8.8835  -5.8604  -2.5772  -1.3209   3.6643  -9.4999
#>    1.6178   0.8005  -0.9280  -7.1360   1.5403   0.7887   5.0483  -5.0528
#>    5.0857  10.0617   1.2503   1.6421  -5.5256  -2.8097  -9.4847   2.9408
#>    7.7962  18.2362   5.9606  -4.2064 -17.2380 -26.4198   1.5490 -13.8061
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
