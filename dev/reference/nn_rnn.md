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
#>  Columns 1 to 9  0.4748 -0.5071 -0.1962 -0.5534  0.2004 -0.2512 -0.0958 -0.7154 -0.0478
#>  -0.3179 -0.7362  0.7724 -0.1936 -0.2680  0.0812 -0.0332 -0.9114  0.2874
#>  -0.2218 -0.9189 -0.0316  0.3717  0.3956 -0.5019 -0.0872 -0.7843  0.4475
#> 
#> Columns 10 to 18  0.2622  0.2200 -0.0817  0.3471  0.1357 -0.6164 -0.2576 -0.0569  0.3184
#>  -0.2157  0.3984  0.2958  0.3361 -0.1459  0.1175 -0.4455  0.7117 -0.8512
#>  -0.3672  0.5356  0.8862  0.5243  0.6858  0.1938  0.6429 -0.1144 -0.7566
#> 
#> Columns 19 to 20 -0.3001 -0.3029
#>   0.4724  0.0874
#>  -0.2008  0.4649
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.2224 -0.4387 -0.1852  0.0644 -0.0530 -0.4982 -0.4536 -0.2624 -0.1956
#>   0.3838 -0.5500  0.0774 -0.2817  0.0663 -0.0653  0.2847  0.1816  0.4773
#>   0.2893 -0.6457  0.3893 -0.3256 -0.1060 -0.1165  0.1811 -0.1129 -0.5525
#> 
#> Columns 10 to 18  0.0583  0.6430  0.4254  0.5586  0.1323  0.0908  0.0367 -0.2673 -0.0055
#>  -0.7115 -0.5089  0.0636  0.1287 -0.3536 -0.2783 -0.2936  0.4486  0.2509
#>   0.4076  0.4025  0.2415  0.5993 -0.2059 -0.1103  0.1776  0.4678 -0.2369
#> 
#> Columns 19 to 20  0.0119  0.3083
#>   0.1800  0.1350
#>  -0.0490  0.2155
#> 
#> (3,.,.) = 
#>  Columns 1 to 9  0.2976 -0.2212 -0.4262 -0.0176 -0.5281 -0.2426 -0.1223  0.1259  0.2640
#>  -0.2158  0.2430  0.3171 -0.0073 -0.0852 -0.0116 -0.5002 -0.4759  0.0431
#>  -0.1399 -0.5396  0.0222  0.2005  0.2569 -0.1510  0.3764 -0.4379  0.2227
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9 -0.3198 -0.8009 -0.8262  0.1147 -0.5213  0.5314 -0.5442 -0.3706  0.4308
#>  -0.1741 -0.2145 -0.3873 -0.3547  0.1966  0.3711 -0.5405 -0.1237 -0.1109
#>   0.0251  0.2149 -0.0822 -0.7533  0.3378  0.0451 -0.4348 -0.1068  0.1988
#> 
#> Columns 10 to 18  0.6591  0.5618 -0.1182  0.4875 -0.2968 -0.2127 -0.8298  0.3569 -0.2465
#>   0.2581 -0.6570  0.1603  0.1358 -0.2553 -0.1413 -0.6272 -0.0508  0.0672
#>  -0.2157 -0.2806 -0.3786 -0.4629  0.0596  0.0481 -0.6118  0.0353  0.1453
#> 
#> Columns 19 to 20  0.5121 -0.1372
#>  -0.2656  0.2130
#>   0.3421  0.4910
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.5951 -0.2198  0.4191 -0.2671 -0.1578 -0.0310  0.0077 -0.1041 -0.1452
#>   0.0466 -0.1385  0.2041 -0.0603 -0.0943 -0.1343 -0.1970 -0.4035  0.1562
#>   0.0739 -0.1060 -0.0208 -0.3670  0.1034 -0.0730  0.0863 -0.5913  0.0850
#> 
#> Columns 10 to 18 -0.2830 -0.3404  0.3140  0.2285  0.0788 -0.5894  0.1450  0.0656  0.1793
#>  -0.3018  0.2275  0.1621  0.0338  0.0445 -0.4014 -0.3904 -0.0707  0.0884
#>  -0.5691  0.4479 -0.2029  0.3344  0.1405  0.3164 -0.4997  0.5724 -0.2813
#> 
#> Columns 19 to 20  0.1823  0.1328
#>  -0.2279  0.3050
#>  -0.5580  0.1132
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
