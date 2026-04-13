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
#> Columns 1 to 8  -6.4272   1.6188   1.7423  11.6356  -5.5775  -3.3786   7.4438 -14.3389
#>   -1.8115   7.0142  -0.9566   4.7405  -3.2698  -1.1855  -2.1672  -6.2801
#>   -1.2202  -6.4592  -2.2146  -7.2621   8.9829  -0.9822  -2.1116   8.8485
#>    0.6585  -3.3908  15.9752  -5.3505  -0.9605   5.4739  -2.0210  -2.6749
#>   -9.1422  12.1836   3.6973 -10.2325  -3.6646   9.2036   5.6631   0.7489
#>    0.1723   9.2219  -4.4339  -4.5076   1.1261  10.0271  -9.8429   4.7008
#>    0.6858   9.7550  -0.9077  -4.5770   9.4168  -6.7195  -3.0291  -3.3668
#>   -1.4444   1.4261  -4.0588   5.8861  -0.2089   4.1162  -6.1455  -2.4007
#>   -1.7927  -1.3567  -5.3263   7.5962   4.8063   6.2606  -4.9897  -2.1476
#>   -0.2395   3.2602   1.0315   8.9432   1.9316  -8.5778 -11.1072   3.8721
#>    1.3211   2.3020   6.0155   4.4341  -5.6993  -1.2992   5.4055  -3.6179
#>    3.3412  -3.9558   7.6034  -2.7907   6.5736  -4.4028  -0.9246   4.3244
#>   10.5726   1.9165  -1.4934   1.9429  -1.0736  -4.0200  -6.4824   1.1634
#>    2.3763   6.0582  -4.0893   0.5629 -13.3580   5.9289   0.4608   3.8079
#>    8.2091   5.9591  11.5797  -0.6407  -2.5311 -11.8073   0.8282  11.2976
#>    3.6683  -3.5702   3.7146  -4.8770   6.5621   1.1717  -6.7988   3.0079
#>    0.5508   5.2092  -3.6233  -4.9606  -2.4252  -8.8854  11.2771   6.6903
#>   -0.5115  11.7016  11.5532   5.7327   1.5446   3.9785  -7.8199  -3.9102
#>    1.7633   9.3470 -11.0415  -6.0026  -4.3000   7.1208  -3.2318  -0.6557
#>   -3.0907  -5.2269   7.1600   0.0124  -9.8438  -3.6512  11.8367  -4.0186
#>    3.4765  -9.4307  -6.0490  -0.5228   2.6071 -15.3713  11.8251   4.1168
#>    2.1129   8.5992  -1.7469   1.4076  10.2505   0.5517  -0.4552 -18.9570
#>   -5.3583  -1.6353  -2.7031   3.1839  -2.5015  -5.1816   3.4220   0.9253
#>   -1.6056  -3.4178   3.6728   3.1433  -3.4456 -11.3176   2.5247   7.9806
#>   -3.1500  -4.0488  -9.7721  -7.2449   1.1224   4.4609  -7.1676  11.0176
#>   -5.1169   1.3904   4.0574  15.9157  -5.4383 -13.2184   0.8203  11.9584
#>   -2.0993   5.1332  -2.4267   1.2197   3.8409   2.4811  -3.9609   6.3532
#>   -4.9680   5.5721  -0.6783  -2.6916   8.3154   1.7947  -4.5611   5.8067
#>   -7.8075  -6.7460  -2.7698  -8.8013   3.3502   8.3986   3.4965  -3.5086
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
