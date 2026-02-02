# Linear

Applies a linear transformation to the incoming data: \\y = xA^T + b\\.

## Usage

``` r
nnf_linear(input, weight, bias = NULL)
```

## Arguments

- input:

  \\(N, \*, in\\features)\\ where `*` means any number of additional
  dimensions

- weight:

  \\(out\\features, in\\features)\\ the weights tensor.

- bias:

  optional tensor \\(out\\features)\\
