# Linear module

Applies a linear transformation to the incoming data: `y = xA^T + b`

## Usage

``` r
nn_linear(in_features, out_features, bias = TRUE)
```

## Arguments

- in_features:

  size of each input sample

- out_features:

  size of each output sample

- bias:

  If set to `FALSE`, the layer will not learn an additive bias. Default:
  `TRUE`

## Shape

- Input: `(N, *, H_in)` where `*` means any number of additional
  dimensions and `H_in = in_features`.

- Output: `(N, *, H_out)` where all but the last dimension are the same
  shape as the input and :math:`H_out = out_features`.

## Attributes

- weight: the learnable weights of the module of shape
  `(out_features, in_features)`. The values are initialized from
  \\U(-\sqrt{k}, \sqrt{k})\\s, where \\k =
  \frac{1}{\mbox{in\\features}}\\

- bias: the learnable bias of the module of shape
  \\(\mbox{out\\features})\\. If `bias` is `TRUE`, the values are
  initialized from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{1}{\mbox{in\\features}}\\

## Examples

``` r
if (torch_is_installed()) {
m <- nn_linear(20, 30)
input <- torch_randn(128, 20)
output <- m(input)
print(output$size())
}
#> [1] 128  30
```
