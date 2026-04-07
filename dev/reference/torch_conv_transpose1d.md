# Conv_transpose1d

Conv_transpose1d

## Usage

``` r
torch_conv_transpose1d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  output_padding = 0L,
  groups = 1L,
  dilation = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sW,)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padW,)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padW)`. Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dW,)`. Default: 1

## conv_transpose1d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

inputs = torch_randn(c(20, 16, 50))
weights = torch_randn(c(16, 33, 5))
nnf_conv_transpose1d(inputs, weights)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 8  -1.4835  -9.5732   5.4111  -7.6651   3.7447  -1.8254  -2.9362  -1.9711
#>    2.8113  -0.1474   9.6034  -2.6226  18.9524 -20.7686   3.0725   4.6660
#>   -1.6819  -8.7901   2.4105  -8.5985  -2.5480 -11.0396 -20.7320  -5.0051
#>    3.3314   8.1727   1.4520  -0.8766   8.8663  -1.1019   6.9300  -2.6664
#>    6.8823   4.6755   0.2048   1.5288  -9.2479   1.1748  -5.6432   7.9523
#>   -6.3780   2.5342   1.1596  -4.6627 -13.3246  -5.6955  -3.4041   1.2249
#>   -2.6542  -1.9142   8.6726 -18.3510 -10.8865  14.3634 -11.8249  -7.0855
#>    1.8831  -8.5795   3.5310  -2.5126  -0.6715   5.4716   7.5573  11.5181
#>    3.5140   3.6360  -0.3457  -8.0989  15.9003  -0.4255  -0.0998   0.6733
#>   -2.2390   5.3469  -5.4041   5.3557  -1.7380  -4.0835   4.4224  -1.9731
#>    1.5798  -5.1308  -2.2532   6.0333   0.1375   3.9748 -11.9339   6.7615
#>   -6.0987  -6.2677  -2.4949  -7.8883   4.8193   2.8822  -3.8580   3.9166
#>    6.1192   0.6129  11.6471   4.8885  12.0397  12.0430  -3.6733  -5.6167
#>    4.1048  -0.4598  -7.0884   4.4350  -4.8674  13.2596  -6.7966  -4.0667
#>   -2.7696   4.5768  -7.8274  -0.0997 -10.7321  -6.0219   8.3296   9.8595
#>    3.2150  -9.7978  -7.4230  -6.8262  -8.9254  13.7519  14.2561 -13.9728
#>    4.8487  -0.4440  -1.0492  -7.5142 -12.9834   2.1019   6.1119   0.7044
#>    2.7294  19.9074  16.3477   7.3904   9.6370  -6.2591  -2.2924  -6.7012
#>   -3.1103  -4.1207  -9.1458  -2.6513   4.0600  -0.4105  -6.5801  -1.5707
#>    2.6917   2.9769   0.1535   4.3622 -11.1299   1.7809 -14.6246  17.0629
#>   -3.8023  -1.0410  -2.6106   1.5570 -12.9946   1.2146   2.4053  -0.2484
#>   -5.7358   1.9958   9.0260  -1.6958   9.1432  -6.5750  -1.7763  -9.3869
#>  -10.1814   6.7733  -9.9978  -0.8625  -4.8017  -0.8699   7.1510  -2.6556
#>    3.0607   9.6657  10.5615   1.4041   4.4323  -4.5691  12.0473  14.0868
#>    3.7454   0.2396   3.5710   3.7168   2.4872  -3.9364  -7.4037  -5.2598
#>    0.9918   0.8027  -7.8894  -9.1171 -19.2436  -1.0931   4.9712   0.0998
#>   -4.0118   3.4701  -4.5152 -13.2623  -4.4876  -2.9645   6.9260 -12.0080
#>   -0.9768  12.9588  -3.5130  -8.4793 -13.4683  -7.8312  -8.2301   7.0026
#>   -5.3626   5.4257  -2.3082 -13.8774  -4.5938 -12.2976  11.0866  14.7851
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
