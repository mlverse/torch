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
#> Columns 1 to 8   1.8535   7.7239  -3.7455  -2.5906  -0.9407  -0.9821  -0.5838   6.7856
#>    7.1763   5.9625  -8.9882  -1.4788  11.9596   3.5214   5.5251   2.8725
#>   -1.9949  -6.8118  -3.1819  -7.9878 -14.4037   3.4900  -3.7352  -0.4261
#>    5.6997  -2.1556 -10.3989  -2.2677   9.0671   1.2520   1.3412  -4.7530
#>    0.1387  14.9021  -8.5967 -11.3951   8.0644   4.2649   3.6915  -9.3508
#>   10.7306  10.7165   7.7883   0.8338  -0.5011   4.6793   1.3614   6.4314
#>    4.3979  -6.7254   9.2900 -20.9415 -10.1536   2.2283  -0.7884 -11.3100
#>   -4.3000   1.0782  -5.6226   9.1944   3.5339   1.2559   2.5426   4.1113
#>   -8.5369  -0.3681  -0.7026   0.2859   2.8896   2.3228  12.2641   5.8166
#>   -7.9205  -1.5670  -0.6037   5.0117  11.7187  -6.8211  -3.1452   3.7205
#>   -3.7556  -3.7088  -3.7302  -1.7601   5.1822  11.1583  -0.7700   9.5334
#>    5.4451  -3.7815   6.5385   5.6146   3.1189   7.6186   2.3014  -5.1821
#>    5.8724  -9.9590  -5.0344   7.6096  -3.2585  -8.4832  -2.4969   1.2309
#>    6.5368  -0.2087  -0.4533   5.6608 -10.4327  -3.6099   7.2071  -5.6297
#>    1.9980 -11.0369  12.3910  -2.9017  -3.2408   4.8976  -4.9290  -3.3943
#>    2.9203  10.8455   5.4685  -9.7343   1.8719  -3.7934  11.4277  -4.2108
#>   -1.1136 -18.4758 -10.3950   2.7303  -4.7162   3.7757  -4.1225  -0.0696
#>    6.8725   4.9778   4.2223   8.8530  -3.8577   3.0989   1.0509  -1.9180
#>   -8.6955  -7.5622  -5.2563  -3.5068   0.4368  -1.5047   2.8432   2.1767
#>    6.4939   7.4739  -2.4106   9.3835  12.1062  -2.2046  10.4034   6.6695
#>   11.1727  -7.2618   3.8031  -6.2391   2.2438  -2.4623  -1.5947 -12.4272
#>    9.4613  13.7369   3.7812  -6.2838  -3.6385   2.9756  11.6024  -3.1062
#>  -17.0120  -8.2185   7.3866   7.7801  21.5242  -0.5182   0.2655   1.6849
#>    3.4630   9.6188  18.6892   6.4655   1.5867   1.6803  -1.7961  13.2887
#>   -8.0089  -8.2056  -5.4121  -1.7634  -6.2538   0.5925   9.0971  -0.3470
#>   -7.6503  -2.2960  -2.1380   3.4997   8.1590 -11.9837   1.1146  -2.6098
#>   -3.6542  14.4710  10.7969   2.9027  -0.8673  12.7617   4.4055  16.3723
#>   -6.9422 -11.3975   1.0304   6.2281  -1.9088  -0.3764   3.2900  -6.7539
#>    7.0128   7.3501   5.3041   6.6298   4.6548  -5.0999  -2.9159  -6.7973
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
