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
#> Columns 1 to 8  -3.6261   8.4342   0.9138  -0.8921  -0.6939  -0.6931 -11.8593  -6.6412
#>   -0.8468   4.1338  -0.5466   5.9786  -2.6753   3.9675   6.8855   1.3959
#>    0.4317   6.7998   7.7433  -6.6647 -10.7573 -10.7264   3.4961  12.1741
#>   -7.1590   4.8439   4.7824  -0.0791   0.2762  -1.2830   1.9056  -6.6529
#>   10.8523   0.5604   2.0447 -13.3148 -15.3618  12.1844  20.4104   1.7811
#>   11.6533   0.8600   2.0946  -4.5364   1.0845  -4.4151  -5.6127  -0.0202
#>   -7.8082   2.8799   5.2297  -1.8023 -18.6863  -6.3325  11.6115   3.7763
#>    3.8299  12.4566  -0.9831 -11.8347  -1.3085  -1.1189  -2.0559   0.5823
#>   -4.2048   0.1814  -0.8609   9.1063   7.5513   4.3601  -5.1853  -9.7016
#>   11.9964   8.0371   2.7904   2.0854   5.8745  -2.9282  -5.1416  -6.1142
#>   -3.4774   0.8805  -6.0640  -3.1446   2.7432  -4.4629   3.3473  -5.3937
#>    4.5959   0.9994  -7.4947  11.1237   9.0480  14.0299 -14.4304   3.7780
#>   -2.8014   6.6470  -5.9780 -14.3344   9.9180   3.6453  -5.4093  -0.8884
#>   -3.4596  12.4616  -1.7151  -7.4249  -0.7306   0.7598 -12.8329  -2.0114
#>    3.9715  -8.6726  -1.6468 -12.9684  -0.3151   5.3235   4.2485   1.8960
#>   -2.5051  -4.6091  -6.1962  -3.3782   2.7357  -5.2825   6.2577  -4.5167
#>   13.5582 -18.6303   5.0227   7.2878  -0.6409  -8.7135   5.2513   0.8833
#>    0.7103  -5.4878  -9.9273  10.2415  -8.6544 -10.9023   5.2227   2.8803
#>   -2.5854   9.5112   0.9351  -3.0281  -7.5975  -2.0477   4.9681   6.6758
#>  -13.9955  15.2951  -1.7455   1.3667   2.6485  -4.3740  -7.8297   8.6875
#>  -12.0000  -3.6824   9.3100  -1.5921   5.7071   0.1262  -4.9482  -1.1300
#>   -6.9703   3.9955  -0.6618   8.4989  -2.7465   4.1456  -1.1537   3.0019
#>   12.8506   5.6592  -0.3032  -5.6937  -9.1416  -6.5025  -0.2969   3.7599
#>    8.5820   3.7434  -2.7986   0.5958   2.3725   1.3608 -17.4518  -1.6274
#>  -13.6092  -4.7956   2.3023  -2.2244   0.1235   3.8339   2.0149  -8.6254
#>   -7.3594   9.2810   8.3258   6.0991 -10.5257  -9.4649   4.5518   2.5952
#>   13.3410   0.7701   6.5442  -6.6037  -9.4443   1.4170  13.5947   7.6433
#>    8.5839  -1.8261  -1.9460  -7.7503  -1.8072   5.0536   1.1497  -2.3767
#>    8.2432   4.3303   0.0809   0.3276   1.4257   8.1415   2.5439  -4.0015
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
