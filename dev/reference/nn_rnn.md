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
#>  Columns 1 to 9 -0.5153  0.3137 -0.0287 -0.5833 -0.1728  0.1784  0.5030  0.5186 -0.0224
#>   0.4786 -0.3594 -0.1939 -0.7974  0.4758 -0.8277  0.4246  0.7596  0.0721
#>   0.1363 -0.5507 -0.7515  0.0203  0.7027 -0.4915 -0.1466  0.5029  0.8839
#> 
#> Columns 10 to 18  0.2595 -0.2145  0.7495  0.3350  0.2244 -0.4894 -0.4296 -0.0050  0.6137
#>  -0.3711 -0.2344  0.4126 -0.6090  0.5140 -0.1382 -0.9331 -0.0249  0.4083
#>  -0.6177  0.1405 -0.5683 -0.0076  0.0144 -0.2284 -0.1908  0.3176  0.6527
#> 
#> Columns 19 to 20  0.2675 -0.3504
#>   0.4987  0.8944
#>   0.2430  0.7188
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.4753  0.3356 -0.4448 -0.0526  0.5010  0.1525 -0.2404 -0.4162  0.3077
#>   0.1319  0.1484 -0.3450 -0.5872  0.3614  0.0509 -0.4955 -0.3049  0.3730
#>  -0.0799  0.3157 -0.0753 -0.6006  0.2692 -0.3983  0.0151  0.1530  0.1681
#> 
#> Columns 10 to 18  0.4439 -0.3813  0.1133  0.1828  0.3507 -0.2568 -0.1100  0.0477 -0.2093
#>   0.0762 -0.3078 -0.2584  0.6235 -0.1117  0.1805  0.2819 -0.2940 -0.5092
#>  -0.3054 -0.3503  0.1467  0.1046 -0.1944 -0.2255 -0.2594  0.1807  0.0458
#> 
#> Columns 19 to 20 -0.4045  0.2066
#>  -0.1041 -0.0717
#>  -0.6010 -0.3579
#> 
#> (3,.,.) = 
#>  Columns 1 to 9  0.7991 -0.1133 -0.2687  0.0331  0.2424  0.1384  0.1515 -0.3232 -0.4512
#>   0.5105 -0.0512 -0.1123  0.2209  0.4554  0.0390  0.1063 -0.0225  0.1073
#>   0.4649 -0.0800  0.0023 -0.0962  0.2029  0.2820 -0.3717 -0.2574 -0.0492
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9 -0.6870  0.3084  0.2668  0.2091 -0.2550  0.1925  0.5900 -0.4046 -0.2125
#>   0.0766 -0.1059  0.0804 -0.0549  0.2071 -0.3609 -0.0814 -0.2306 -0.6635
#>  -0.1653 -0.0709  0.1589 -0.3854  0.5453  0.6231  0.2687 -0.3612  0.0817
#> 
#> Columns 10 to 18  0.3560  0.1351 -0.7975  0.2679  0.2851  0.3174  0.1529 -0.0404  0.5705
#>   0.2796 -0.0458 -0.3225 -0.7959 -0.0583 -0.2877  0.4899  0.6353  0.8612
#>  -0.0688  0.5347 -0.4208  0.0180 -0.5816  0.1755 -0.0372  0.1661 -0.3176
#> 
#> Columns 19 to 20  0.6491 -0.3167
#>   0.8387 -0.1711
#>  -0.1934 -0.8158
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.3296  0.2953 -0.2830  0.0718  0.0717 -0.0956  0.2219 -0.3156 -0.3049
#>   0.3921  0.2198 -0.2580 -0.0442  0.6078 -0.0570 -0.1830 -0.4076 -0.4461
#>   0.0946  0.2490 -0.2766 -0.3737  0.2490 -0.0139 -0.4595 -0.3654 -0.3656
#> 
#> Columns 10 to 18 -0.0408 -0.4793  0.4348  0.5227 -0.3192 -0.1292 -0.0634  0.1546 -0.0017
#>  -0.2096 -0.4367 -0.0145  0.0736 -0.0563 -0.4735 -0.0047  0.5504  0.1762
#>   0.2827 -0.3346  0.0640  0.1703  0.0633  0.0183 -0.0977 -0.3780  0.1872
#> 
#> Columns 19 to 20 -0.5907 -0.4317
#>  -0.6404 -0.2146
#>  -0.1683 -0.3182
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
