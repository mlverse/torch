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
#> Columns 1 to 8   3.9526  -0.7846  -4.5646   1.8201   5.2002  -0.9125  -5.7063  -2.6811
#>   -7.0181  -1.3322  -3.7834  -0.3951  13.7224  19.3091  -7.4025  -0.6126
#>   -0.1106  -5.9596   0.4363   9.9238  10.6983   1.5480 -13.0258   5.2199
#>   -5.9060   1.2757   1.2856  -1.6831 -15.7051   6.8492  -7.8865   1.4036
#>    1.2339   0.0166   0.9093   1.9303  -3.8817   6.8935  -9.4423   2.3420
#>  -13.9258  -4.8205  -3.7961   5.8331  -6.5238  -0.9708 -12.8743  -7.7355
#>    5.8862   2.3220  -2.1169  -0.4128  -7.1936  -4.9462  -0.2138   6.3836
#>   -6.0623  -8.9208   0.4704 -11.5116   3.7984  -0.9882  -6.1722  -8.4232
#>    0.6117  -0.4394   5.3232  -4.4840  -3.4889   5.0686   5.9483  -5.3664
#>   -0.1959  -5.6573   5.1060  -4.2040  14.7387   0.7845 -13.4758   0.6291
#>   -5.6328   8.9019   1.7041   0.1826  -0.8366  15.5140   4.9567   7.0967
#>    3.1092   6.4788  -1.2836  -2.4439   4.8699  -0.5753  -1.8063  -0.2908
#>   -4.9149  -3.1244   2.2378  -6.4664   8.4599   2.4480  17.9376  -1.3604
#>    4.9571   8.2468  -3.6274  10.8410  -0.4628  14.0821  15.8331  -2.2074
#>    1.2875   9.5487   9.0903  -3.2875  -0.5327   5.0666   3.3333   2.8819
#>   -1.8578   2.8975  -0.7884  -3.9422  -6.8476  -0.5183   8.7492 -13.3142
#>    3.4518  -2.1724   1.4589  -4.0909  10.6605   3.2325   9.8330   9.4535
#>    3.1144   3.3779   7.6609   6.2296  -2.0544  13.4733   6.2199   6.0174
#>    2.2014  -6.5013   0.4931  -6.9217   9.7705   4.6136   1.5409  -3.7732
#>    7.1647  -1.2650  15.2722   7.3270  -5.5648  -9.7123  16.3829  -9.8016
#>   -6.0231   4.9656   7.5575   0.6702   3.6839   0.9675   7.0907  16.8667
#>    2.0783   3.3364  10.8075   1.8333   3.0445   9.4771   1.8831   8.6198
#>    0.0778  -0.1391   2.5579  -0.7613  -3.9089   3.0723   2.7581  10.7264
#>   -0.1358  -0.7634   5.8111   8.2727   3.0357  -0.6270  -2.8074  -3.3621
#>   -3.5338  -3.5870  -3.7635   1.3960   1.0024  -3.6628   7.7951   8.1623
#>   -7.5768  -3.6019  -8.5020  -1.0402  -6.0256   3.5779  -3.8238 -13.5096
#>   -1.4893   2.8496  -3.5824 -12.5764 -24.7718  -0.0083 -14.3237  -6.9638
#>   -3.2374  -7.8215  -6.5278  -3.4898 -12.9852  -8.9483 -16.4704 -10.1078
#>    0.2531 -12.8666  -8.5701   0.6322   0.1159   6.4095   4.2441  19.9604
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
