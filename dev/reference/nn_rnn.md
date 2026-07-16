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
#> Columns 1 to 9  0.5956 -0.6885  0.3893  0.4138  0.1568 -0.1687 -0.0157  0.4969 -0.2228
#>   0.3417  0.1594  0.8740  0.4469  0.0232 -0.8721  0.7309 -0.4803 -0.7725
#>   0.7201 -0.2516  0.2783  0.6105 -0.0780 -0.0078  0.3063 -0.1227 -0.3161
#> 
#> Columns 10 to 18  0.5091 -0.2035 -0.1968 -0.2541 -0.5429 -0.1196  0.0519 -0.0238 -0.8112
#>   0.2523 -0.3947  0.5055  0.1790 -0.5306  0.0367 -0.3318  0.4726 -0.7447
#>  -0.6879 -0.5227 -0.1879  0.3766 -0.1673 -0.0781  0.5870  0.1529  0.2476
#> 
#> Columns 19 to 20 -0.5949  0.2092
#>   0.4241 -0.6225
#>   0.5237  0.2838
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2830 -0.1025 -0.2108  0.2713 -0.1520  0.2761  0.4491  0.1212 -0.0563
#>   0.3244  0.2050 -0.2929  0.2456 -0.4679  0.3668  0.4787  0.2196 -0.2932
#>  -0.0903  0.2089 -0.1635  0.2844 -0.5856  0.1203  0.2269 -0.4388 -0.0686
#> 
#> Columns 10 to 18 -0.6514 -0.1053  0.0325  0.0615 -0.2621  0.1788 -0.0931 -0.3087 -0.2993
#>  -0.2033  0.3226 -0.3447 -0.4578 -0.3390  0.1213 -0.7087 -0.1880  0.0161
#>  -0.1043  0.4966  0.2572 -0.4749  0.1395 -0.1857 -0.1433  0.5840  0.0899
#> 
#> Columns 19 to 20  0.3680 -0.1072
#>  -0.0857  0.4753
#>  -0.1063  0.6121
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.5059  0.0380  0.2579  0.1576 -0.4885  0.3649 -0.0486  0.0429 -0.2824
#>  -0.1535  0.0257  0.2096  0.4582 -0.3767 -0.5436  0.1407 -0.0216 -0.3743
#>  -0.4217 -0.4316  0.3148  0.0441 -0.6812 -0.6062  0.1079  0.2025 -0.3066
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.1950  0.2572 -0.3995 -0.0320 -0.1962 -0.5931  0.3414 -0.4887 -0.0372
#>  -0.6668 -0.2484 -0.3662 -0.4478 -0.4017  0.3716  0.7091 -0.3481  0.5519
#>  -0.4602 -0.2106  0.4789  0.2673 -0.6397 -0.5387  0.2672 -0.1889  0.0729
#> 
#> Columns 10 to 18 -0.1570 -0.0144  0.4802 -0.2377  0.4699  0.0272  0.2369 -0.0262  0.3601
#>   0.0058 -0.3593 -0.1720  0.0938  0.8011  0.4662  0.1842  0.4599  0.2744
#>  -0.3124 -0.8350  0.6396  0.1840  0.5910  0.1773 -0.3460 -0.0094  0.0929
#> 
#> Columns 19 to 20 -0.1963  0.0179
#>  -0.2098 -0.5651
#>  -0.3414  0.7035
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.1002 -0.0549 -0.2429  0.4179 -0.2592  0.0223  0.1613  0.0028 -0.2631
#>   0.1871 -0.2152 -0.4980  0.0938  0.1714  0.1850  0.2779  0.1541  0.2247
#>   0.1173 -0.0436  0.0601  0.3464  0.0362 -0.2607  0.4305  0.2261 -0.2848
#> 
#> Columns 10 to 18  0.0490  0.3402  0.0958 -0.4501  0.0615  0.2425 -0.4352  0.0651 -0.2250
#>   0.0265  0.4386  0.4723 -0.1454  0.2825  0.4110 -0.4162  0.4091 -0.0154
#>  -0.0801 -0.0965  0.1417  0.0424  0.4232  0.1879 -0.1454  0.2235 -0.1100
#> 
#> Columns 19 to 20  0.0201  0.3307
#>   0.0950  0.1830
#>   0.3017  0.4044
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
