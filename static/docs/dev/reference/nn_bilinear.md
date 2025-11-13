# Bilinear module

Applies a bilinear transformation to the incoming data \\y = x_1^T A
x_2 + b\\

## Usage

``` r
nn_bilinear(in1_features, in2_features, out_features, bias = TRUE)
```

## Arguments

- in1_features:

  size of each first input sample

- in2_features:

  size of each second input sample

- out_features:

  size of each output sample

- bias:

  If set to `FALSE`, the layer will not learn an additive bias. Default:
  `TRUE`

## Shape

- Input1: \\(N, \*, H\_{in1})\\ \\H\_{in1}=\mbox{in1\\features}\\ and
  \\\*\\ means any number of additional dimensions. All but the last
  dimension of the inputs should be the same.

- Input2: \\(N, \*, H\_{in2})\\ where \\H\_{in2}=\mbox{in2\\features}\\.

- Output: \\(N, \*, H\_{out})\\ where \\H\_{out}=\mbox{out\\features}\\
  and all but the last dimension are the same shape as the input.

## Attributes

- weight: the learnable weights of the module of shape
  \\(\mbox{out\\features}, \mbox{in1\\features},
  \mbox{in2\\features})\\. The values are initialized from
  \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\, where \\k =
  \frac{1}{\mbox{in1\\features}}\\

- bias: the learnable bias of the module of shape
  \\(\mbox{out\\features})\\. If `bias` is `TRUE`, the values are
  initialized from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\, where \\k =
  \frac{1}{\mbox{in1\\features}}\\

## Examples

``` r
if (torch_is_installed()) {
m <- nn_bilinear(20, 30, 50)
input1 <- torch_randn(128, 20)
input2 <- torch_randn(128, 30)
output <- m(input1, input2)
print(output$size())
}
#> [1] 128  50
```
