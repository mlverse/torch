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
#> Columns 1 to 9  0.5824 -0.8954  0.7062 -0.2203 -0.4582 -0.6934 -0.7126 -0.6971  0.5471
#>   0.6061 -0.6554  0.7014 -0.6062 -0.5362  0.3581  0.4833  0.4537  0.5983
#>  -0.6281  0.1970  0.3082 -0.2091 -0.2534 -0.4911  0.2544 -0.0513  0.4127
#> 
#> Columns 10 to 18 -0.7810 -0.0368  0.6306 -0.1494  0.3085  0.8102 -0.7102 -0.7882 -0.0544
#>  -0.9155  0.5730  0.7162  0.0670  0.8390  0.8824  0.1569 -0.7401  0.3438
#>   0.4211 -0.2775  0.6453 -0.1953 -0.2757 -0.3606  0.2258 -0.7773  0.2028
#> 
#> Columns 19 to 20  0.3211 -0.7157
#>   0.4410  0.7642
#>   0.8375  0.1206
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.0400  0.0094  0.0790 -0.4797  0.4332 -0.0649 -0.0464  0.1313  0.3024
#>   0.2920 -0.3491  0.6850 -0.6343 -0.2329 -0.3005 -0.3118  0.0974  0.4052
#>   0.3048 -0.3149  0.3926  0.0997 -0.4395  0.6237 -0.1688 -0.4714  0.4132
#> 
#> Columns 10 to 18  0.2954  0.1381  0.1965 -0.1215  0.4000 -0.4764  0.6593  0.1260 -0.0158
#>   0.3005 -0.2194  0.3899  0.3361 -0.2736 -0.1900  0.1013 -0.6893 -0.2636
#>  -0.1444 -0.1181  0.4111  0.2566  0.2526  0.3795 -0.0728 -0.1462  0.0514
#> 
#> Columns 19 to 20 -0.1616  0.6373
#>   0.4256  0.5245
#>  -0.0897  0.0168
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.2684  0.1154  0.0937  0.4128  0.0276  0.4868  0.2965 -0.4178  0.4936
#>   0.3215  0.0431  0.0443 -0.2694 -0.3242  0.2454 -0.3177  0.1089  0.6213
#>  -0.1880  0.0527  0.2278 -0.2452  0.3348 -0.7787 -0.3815  0.3165  0.0922
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.5646  0.5187  0.4340  0.6031  0.4763 -0.3453 -0.3325 -0.1730  0.5643
#>  -0.1320 -0.5266  0.5084 -0.1781  0.3968 -0.1137  0.4275 -0.0637 -0.6469
#>  -0.2651  0.4836 -0.1998  0.0748  0.4712 -0.4623  0.2728  0.0280 -0.7639
#> 
#> Columns 10 to 18  0.1184  0.3020 -0.2107  0.3434  0.2896 -0.2608  0.5909  0.4542  0.6529
#>   0.8455 -0.8119 -0.6695  0.2884 -0.7071  0.7129 -0.3654 -0.8884 -0.7805
#>   0.5598 -0.1287  0.0447 -0.3701  0.2912  0.7044  0.0063 -0.8192  0.2802
#> 
#> Columns 19 to 20  0.2845 -0.4837
#>   0.3781 -0.3799
#>   0.7504 -0.4622
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.3832  0.0440  0.5079 -0.4392 -0.0558 -0.3940  0.0793  0.0203  0.2325
#>  -0.1280  0.1611  0.1566  0.1509  0.4344  0.1707  0.1703 -0.3651  0.3705
#>  -0.3136  0.3646  0.2451  0.1753  0.1695  0.4624 -0.5598 -0.0359  0.4407
#> 
#> Columns 10 to 18 -0.0973  0.1729  0.5816  0.0662  0.4070 -0.1087 -0.0993 -0.1534 -0.2760
#>   0.4203  0.1762  0.4992  0.2861  0.1945 -0.3118 -0.0278 -0.4491  0.5680
#>   0.0905 -0.0548  0.3996 -0.1860  0.0622 -0.1141  0.1750 -0.3679  0.6443
#> 
#> Columns 19 to 20  0.3119  0.4681
#>   0.0225  0.0532
#>   0.0853 -0.1112
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
