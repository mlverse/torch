# Bilinear

Applies a bilinear transformation to the incoming data: \\y = x_1 A
x_2 + b\\

## Usage

``` r
nnf_bilinear(input1, input2, weight, bias = NULL)
```

## Arguments

- input1:

  \\(N, \*, H\_{in1})\\ where \\H\_{in1}=\mbox{in1\\features}\\ and
  \\\*\\ means any number of additional dimensions. All but the last
  dimension of the inputs should be the same.

- input2:

  \\(N, \*, H\_{in2})\\ where \\H\_{in2}=\mbox{in2\\features}\\

- weight:

  \\(\mbox{out\\features}, \mbox{in1\\features}, \mbox{in2\\features})\\

- bias:

  \\(\mbox{out\\features})\\

## Value

output \\(N, \*, H\_{out})\\ where \\H\_{out}=\mbox{out\\features}\\ and
all but the last dimension are the same shape as the input.
