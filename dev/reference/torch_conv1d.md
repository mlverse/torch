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
#> Columns 1 to 8   1.4050   2.2013  -5.7405  -8.7386   1.3596   7.1850  -5.0178  -3.1134
#>    6.6810  -5.1400  -4.9908   6.7166 -12.7472   1.3122  -0.5262   6.2607
#>    8.2332  -9.1746  -6.8931  15.5918   2.3564  -9.6039   1.5151  -6.4861
#>    2.3732   5.1944  -0.4147   2.4609   6.4332   1.1745   4.6892   6.6821
#>   -0.1376  -1.7582   9.3633   9.3493  -6.0659  -5.3231   3.7596  -7.2878
#>    0.2740   2.0378  -4.0903   1.4757  -0.2514   9.0250   1.8873   2.7348
#>    8.2136   0.2529  -2.5202   3.0755  -5.6832  13.1410   4.2600   2.3733
#>    4.0955   2.4831   1.1981  -2.7104  -0.5764  -3.4012   5.8069   8.4092
#>    6.8217  -7.7755   6.3742  12.2774   3.3944 -15.7861   6.1612  15.7176
#>    5.1327  -4.4122  12.6481  -3.1304   1.3644 -19.9911  -7.2126   3.9842
#>   -3.0497   7.8028   6.3869   5.6356  -5.3789   4.9096   3.6322   5.3900
#>    1.3000   4.9357 -12.2198  -7.2741   1.5914   0.7840  -3.3279   9.3649
#>    6.2402  -4.0420  -4.5893  11.1992   8.1238  -7.2637  -5.1352  -4.8572
#>   -1.9091   8.0606  -0.9315  -6.6699   0.5949  -8.1886  -5.4587   9.2262
#>   -0.3680  -2.7482   4.2984   2.5195  -7.5826   4.1900   6.2308   8.7511
#>   -3.9930   4.0390  -9.1536   0.9195  -1.4834   1.7018   8.3014  -0.4383
#>   -1.1426  -5.0736  -2.9213   0.3118   4.7641  -3.3616   5.6935   5.9153
#>   -5.3878   0.2961   2.2831   1.4626  11.2232   1.2429   7.5371 -14.6644
#>    5.8756   8.1768  -8.5507   2.3842   1.7374  -1.4190  -6.7794   5.5920
#>    5.9355  -7.1655   1.1685 -12.9429   3.1161  -7.3715   2.0963   0.5799
#>    2.2719   6.9489   2.8146   2.0713  -3.4118  -0.4178  -7.1112 -10.2651
#>    6.7247 -15.1273   0.1809   7.4779   6.1148   2.5522  -9.2790  -6.4189
#>   -3.0653  -3.6165  11.3905  -0.1489   7.0101   4.4403  -4.0117   0.6933
#>   -3.7985  -1.1813  -7.7402  -3.6476  -7.2776   0.7452   5.4624   5.4892
#>   -2.5174   1.5838  -2.6329  -2.3291  -4.2530  11.9226 -10.5054  -2.6013
#>    3.2017   9.5322  -8.0678   2.4863   9.4194   0.0070  -0.5526  -4.7177
#>   -0.1682   0.6593  -4.6493 -11.6521   2.1963   7.4854  -6.1325  10.0046
#>    8.4727   4.7137  -8.7259  -4.5441   4.0868   3.3251  -1.5340   1.8360
#>    4.6969  -7.3875  -2.3318  -4.7583   3.0460   1.7284  17.3885  -0.7852
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
