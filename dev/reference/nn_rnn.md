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
#>  Columns 1 to 9 -0.2538  0.6051  0.4463 -0.1646 -0.5408  0.2862 -0.2930 -0.6393  0.4124
#>  -0.2134  0.0398 -0.3164  0.5972 -0.6650  0.1684  0.0054  0.4525  0.2146
#>   0.5283  0.2897  0.5776  0.0305  0.1472  0.7955 -0.6853  0.8260 -0.2576
#> 
#> Columns 10 to 18 -0.7134 -0.0702  0.0634 -0.7844  0.9320  0.4608  0.1181 -0.8428  0.3213
#>   0.5684  0.2115  0.1504  0.5492 -0.3125 -0.0887  0.3778  0.7098 -0.6781
#>  -0.4259  0.3303  0.1758  0.3304  0.4936 -0.0633 -0.0783 -0.2217  0.4856
#> 
#> Columns 19 to 20  0.7851  0.7053
#>  -0.4085 -0.2370
#>   0.9090  0.5754
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.0989  0.3653 -0.2843  0.0848  0.1610  0.4198  0.2665 -0.2988  0.2154
#>  -0.5070 -0.1874  0.0960 -0.1278 -0.3935  0.2079 -0.4575  0.2697 -0.1131
#>   0.1867  0.0203 -0.1556 -0.2119  0.1408  0.8060 -0.7257 -0.1668 -0.4543
#> 
#> Columns 10 to 18 -0.1433  0.7829  0.3020  0.0912 -0.5626  0.5542 -0.5234 -0.8619  0.4211
#>  -0.2983  0.2639 -0.2223  0.3384  0.5216  0.3600  0.0164  0.0772  0.3083
#>   0.0135  0.0592  0.3138  0.5280 -0.3607 -0.3132 -0.0120 -0.6508  0.3897
#> 
#> Columns 19 to 20  0.2772  0.4559
#>  -0.0592  0.3695
#>   0.5117  0.3262
#> 
#> (3,.,.) = 
#>  Columns 1 to 9 -0.4955  0.1496 -0.2689  0.4672  0.0422  0.2024 -0.2458 -0.4692  0.3491
#>  -0.2343  0.2475  0.2294  0.1792 -0.2553  0.4791 -0.2439 -0.1713  0.1748
#>  -0.2552 -0.1725 -0.2713  0.2526 -0.4547  0.7322 -0.5791 -0.0319 -0.1133
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9 -0.0179  0.6189  0.0331 -0.2619 -0.0087 -0.0521  0.0932  0.3551  0.0399
#>   0.0857  0.4326  0.0676  0.2438 -0.6937  0.6682 -0.4595  0.6195  0.4630
#>  -0.3517 -0.2237  0.4386  0.5895  0.2686 -0.3113 -0.3089 -0.4168 -0.5884
#> 
#> Columns 10 to 18  0.3902 -0.4854  0.7840  0.0585 -0.0212 -0.3605  0.5986 -0.0483  0.1429
#>   0.0058  0.6313  0.0154 -0.6661  0.1253  0.8201  0.2846  0.0484 -0.3380
#>   0.7829  0.4833  0.3431  0.2894 -0.0084  0.0971  0.5711 -0.6221  0.6474
#> 
#> Columns 19 to 20 -0.5881  0.5415
#>  -0.3465  0.3216
#>   0.1294 -0.1770
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.3497  0.0927 -0.0164  0.2575 -0.0444  0.0213 -0.3908  0.1658  0.1646
#>  -0.2044  0.3936 -0.3401 -0.2385  0.0694  0.4370 -0.3463  0.1661 -0.0605
#>  -0.5711 -0.2915 -0.1202  0.4739 -0.5519  0.6433 -0.3076 -0.1589  0.2496
#> 
#> Columns 10 to 18  0.1402 -0.0478  0.4700  0.1836 -0.1242 -0.0764 -0.3244 -0.2277 -0.1447
#>   0.0155  0.7552  0.0049 -0.1148 -0.1918  0.2652  0.1355  0.0466  0.5461
#>   0.1176 -0.2713 -0.1406  0.4195 -0.0083 -0.0413 -0.4330  0.1125  0.0196
#> 
#> Columns 19 to 20 -0.1626 -0.1734
#>  -0.1081  0.1231
#>   0.1522  0.1450
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
