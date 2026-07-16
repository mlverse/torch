# RNN module

Applies a multi-layer Elman RNN with \\\tanh\\ or \\\mbox{ReLU}\\
non-linearity to an input sequence.

## Usage

``` r
nn_rnn(
  input_size,
  hidden_size,
  num_layers = 1,
  nonlinearity = NULL,
  bias = TRUE,
  batch_first = FALSE,
  dropout = 0,
  bidirectional = FALSE,
  ...
)
```

## Arguments

- input_size:

  The number of expected features in the input `x`

- hidden_size:

  The number of features in the hidden state `h`

- num_layers:

  Number of recurrent layers. E.g., setting `num_layers=2` would mean
  stacking two RNNs together to form a `stacked RNN`, with the second
  RNN taking in outputs of the first RNN and computing the final
  results. Default: 1

- nonlinearity:

  The non-linearity to use. Can be either `'tanh'` or `'relu'`. Default:
  `'tanh'`

- bias:

  If `FALSE`, then the layer does not use bias weights `b_ih` and
  `b_hh`. Default: `TRUE`

- batch_first:

  If `TRUE`, then the input and output tensors are provided as
  `(batch, seq, feature)`. Default: `FALSE`

- dropout:

  If non-zero, introduces a `Dropout` layer on the outputs of each RNN
  layer except the last layer, with dropout probability equal to
  `dropout`. Default: 0

- bidirectional:

  If `TRUE`, becomes a bidirectional RNN. Default: `FALSE`

- ...:

  other arguments that can be passed to the super class.

## Details

For each element in the input sequence, each layer computes the
following function:

\$\$ h_t = \tanh(W\_{ih} x_t + b\_{ih} + W\_{hh} h\_{(t-1)} + b\_{hh})
\$\$

where \\h_t\\ is the hidden state at time `t`, \\x_t\\ is the input at
time `t`, and \\h\_{(t-1)}\\ is the hidden state of the previous layer
at time `t-1` or the initial hidden state at time `0`. If `nonlinearity`
is `'relu'`, then \\\mbox{ReLU}\\ is used instead of \\\tanh\\.

## Inputs

- **input** of shape `(seq_len, batch, input_size)`: tensor containing
  the features of the input sequence. The input can also be a packed
  variable length sequence.

- **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`:
  tensor containing the initial hidden state for each element in the
  batch. Defaults to zero if not provided. If the RNN is bidirectional,
  num_directions should be 2, else it should be 1.

## Outputs

- **output** of shape `(seq_len, batch, num_directions * hidden_size)`:
  tensor containing the output features (`h_t`) from the last layer of
  the RNN, for each `t`. If a :class:`nn_packed_sequence` has been given
  as the input, the output will also be a packed sequence. For the
  unpacked case, the directions can be separated using
  `output$view(seq_len, batch, num_directions, hidden_size)`, with
  forward and backward being direction `0` and `1` respectively.
  Similarly, the directions can be separated in the packed case.

- **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`:
  tensor containing the hidden state for `t = seq_len`. Like *output*,
  the layers can be separated using
  `h_n$view(num_layers, num_directions, batch, hidden_size)`.

## Shape

- Input1: \\(L, N, H\_{in})\\ tensor containing input features where
  \\H\_{in}=\mbox{input\\size}\\ and `L` represents a sequence length.

- Input2: \\(S, N, H\_{out})\\ tensor containing the initial hidden
  state for each element in the batch. \\H\_{out}=\mbox{hidden\\size}\\
  Defaults to zero if not provided. where \\S=\mbox{num\\layers} \*
  \mbox{num\\directions}\\ If the RNN is bidirectional, num_directions
  should be 2, else it should be 1.

- Output1: \\(L, N, H\_{all})\\ where \\H\_{all}=\mbox{num\\directions}
  \* \mbox{hidden\\size}\\

- Output2: \\(S, N, H\_{out})\\ tensor containing the next hidden state
  for each element in the batch

## Attributes

- `weight_ih_l[k]`: the learnable input-hidden weights of the k-th
  layer, of shape `(hidden_size, input_size)` for `k = 0`. Otherwise,
  the shape is `(hidden_size, num_directions * hidden_size)`

- `weight_hh_l[k]`: the learnable hidden-hidden weights of the k-th
  layer, of shape `(hidden_size, hidden_size)`

- `bias_ih_l[k]`: the learnable input-hidden bias of the k-th layer, of
  shape `(hidden_size)`

- `bias_hh_l[k]`: the learnable hidden-hidden bias of the k-th layer, of
  shape `(hidden_size)`

## Note

All the weights and biases are initialized from \\\mathcal{U}(-\sqrt{k},
\sqrt{k})\\ where \\k = \frac{1}{\mbox{hidden\\size}}\\

## Examples

``` r
if (torch_is_installed()) {
rnn <- nn_rnn(10, 20, 2)
input <- torch_randn(5, 3, 10)
h0 <- torch_randn(2, 3, 20)
rnn(input, h0)
}
#> [[1]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.0324  0.0565 -0.4417 -0.2031  0.4513  0.2139  0.0439 -0.5094  0.6978
#>  -0.0943  0.3912 -0.4445  0.7663 -0.5791  0.7683  0.5546 -0.0096  0.0086
#>  -0.3801 -0.8031  0.6340  0.5996  0.3909  0.0740  0.3281 -0.6939  0.8128
#> 
#> Columns 10 to 18 -0.7902  0.7492 -0.6968  0.5992  0.5308 -0.3890  0.0839 -0.2031  0.2066
#>   0.3239 -0.6595  0.2420  0.3245  0.4265 -0.6870 -0.3577  0.7086 -0.6614
#>   0.2984 -0.8857  0.3254  0.2932 -0.2096 -0.9337  0.2893  0.4386  0.0537
#> 
#> Columns 19 to 20  0.5546 -0.7477
#>   0.4613 -0.7382
#>   0.1962 -0.7396
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.3198 -0.1453 -0.4739 -0.1048 -0.2706 -0.3898 -0.1404 -0.6757  0.6521
#>   0.1581 -0.3963 -0.2311 -0.0991  0.0213 -0.4023  0.2812 -0.0820  0.2401
#>   0.0406 -0.1056  0.1738  0.6413  0.3182  0.2327  0.2263 -0.0223  0.1807
#> 
#> Columns 10 to 18 -0.1602 -0.2501 -0.0224  0.4143 -0.2838 -0.1324  0.1485  0.1113  0.6601
#>  -0.2504 -0.7258  0.2488  0.1446  0.2861 -0.2235  0.1356  0.1624  0.2724
#>   0.5188  0.2718  0.3700  0.1759 -0.2806 -0.1407  0.0381  0.2787  0.1640
#> 
#> Columns 19 to 20  0.1615  0.0045
#>  -0.1099 -0.1832
#>   0.1026  0.5140
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.3832 -0.3357 -0.1361  0.2636 -0.1173 -0.6729 -0.1885  0.5336 -0.2121
#>  -0.1271  0.0112  0.2042  0.0929 -0.5135 -0.5800  0.1081 -0.1298 -0.1840
#>  -0.2868 -0.1745 -0.2545  0.0229 -0.4533  0.0224  0.5657  0.3104  0.4317
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.7114 -0.8292  0.1747 -0.6265  0.1754 -0.4511  0.1596  0.8182 -0.1157
#>  -0.1911  0.1793  0.4948  0.2017 -0.1208 -0.3898 -0.0838  0.2482  0.0539
#>   0.4737 -0.4205  0.3673 -0.0444  0.7199 -0.8491 -0.1234  0.4912 -0.6347
#> 
#> Columns 10 to 18 -0.5477 -0.3688  0.5056  0.0120  0.5910  0.1501  0.5584 -0.4238 -0.5361
#>  -0.6985 -0.2705  0.3666  0.2641  0.3138 -0.1955 -0.2109  0.5433  0.3015
#>  -0.6798 -0.7841  0.5640  0.6279  0.6360 -0.3619  0.5811  0.6209  0.2695
#> 
#> Columns 19 to 20  0.5208  0.6098
#>  -0.0228 -0.2007
#>  -0.4841  0.5988
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.5265 -0.1315 -0.3702  0.3298 -0.2815 -0.4469 -0.0931  0.6082  0.1048
#>  -0.3141  0.0801 -0.3310  0.1768 -0.3126 -0.1038  0.1035 -0.2565  0.4706
#>   0.0155  0.4090 -0.6161  0.2371 -0.6675 -0.4509 -0.0363 -0.2045  0.4918
#> 
#> Columns 10 to 18  0.3160  0.2629 -0.2250  0.2268 -0.3093 -0.2215  0.1265  0.4229  0.3729
#>   0.0442 -0.1192  0.3223 -0.0555 -0.2376 -0.5956 -0.1667  0.1019  0.4384
#>   0.2050  0.2934 -0.1071  0.0964 -0.2655 -0.6301  0.0638  0.2044  0.2344
#> 
#> Columns 19 to 20 -0.3354  0.0784
#>  -0.1004 -0.1801
#>  -0.1297 -0.2289
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
