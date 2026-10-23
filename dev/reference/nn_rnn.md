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
#> Columns 1 to 9  0.6871 -0.5942 -0.1977 -0.6675 -0.7298 -0.0435 -0.5913 -0.8515  0.2333
#>   0.7237 -0.7655 -0.7194  0.6413 -0.2432 -0.6676  0.2633  0.3829 -0.3097
#>  -0.0402  0.1929 -0.7263 -0.0166 -0.2057 -0.7513 -0.1138  0.9004  0.1257
#> 
#> Columns 10 to 18  0.8192  0.3337 -0.6798 -0.3888  0.1204  0.5918 -0.3648  0.4021 -0.2972
#>   0.0692  0.3247 -0.0085  0.8355 -0.7234 -0.4802 -0.4012  0.7399 -0.7305
#>  -0.5756  0.5492  0.3454  0.4103  0.2539 -0.3863  0.4293  0.1909  0.1988
#> 
#> Columns 19 to 20 -0.1628 -0.7970
#>   0.3907 -0.8049
#>   0.3770 -0.3230
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.6369 -0.2249  0.1905 -0.4612 -0.8062 -0.1241 -0.5623 -0.2118  0.3570
#>   0.7575 -0.3750 -0.2183 -0.3029 -0.0476  0.5105  0.5198 -0.2942 -0.4221
#>   0.4964 -0.0074 -0.4553 -0.1180 -0.2704 -0.0691  0.1736  0.0948  0.6257
#> 
#> Columns 10 to 18 -0.3260  0.0022  0.0059  0.5687  0.0423 -0.8388 -0.5676  0.2408 -0.6299
#>   0.0286 -0.0407 -0.7053 -0.0463 -0.4977 -0.2734 -0.3485 -0.1077 -0.3867
#>  -0.1673  0.2708 -0.5054 -0.3720  0.1500 -0.7142  0.0955 -0.3817 -0.0263
#> 
#> Columns 19 to 20 -0.2156 -0.7813
#>   0.1242  0.0612
#>  -0.1176  0.1893
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.5799  0.0375  0.0039 -0.6898 -0.4977  0.0233 -0.0917  0.0384  0.1781
#>   0.5100 -0.4703 -0.1070  0.2166 -0.2607  0.0191  0.0599  0.1330  0.3445
#>   0.5536 -0.1716 -0.3095  0.4757 -0.2860 -0.3461  0.3765  0.7667  0.4420
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.1651  0.3177  0.3939 -0.7182  0.3122 -0.3494 -0.2465  0.3883 -0.0660
#>   0.4364  0.1680 -0.4037 -0.3378 -0.3528 -0.6304  0.0853  0.2458  0.1530
#>   0.2262 -0.5934 -0.0984  0.3244  0.0999 -0.3511 -0.4066  0.6266  0.1821
#> 
#> Columns 10 to 18  0.6650 -0.0189 -0.6572 -0.7420 -0.0438  0.2117  0.6326  0.0186  0.4226
#>  -0.0260 -0.1427  0.1411  0.2841  0.3081  0.4365 -0.2958 -0.0722 -0.1029
#>   0.5559  0.3319 -0.6792 -0.8778  0.7831  0.6825  0.8751  0.2230 -0.7080
#> 
#> Columns 19 to 20  0.3750  0.2282
#>  -0.2268  0.0405
#>  -0.4119 -0.3931
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.3012 -0.3411 -0.3330  0.1319 -0.2730  0.2089 -0.1630 -0.4239 -0.3049
#>   0.4530 -0.3230  0.1073 -0.1887 -0.3730 -0.4497 -0.1114  0.4127  0.1592
#>   0.3690 -0.4664 -0.1424  0.4413  0.1265  0.2957  0.1330  0.5662 -0.1674
#> 
#> Columns 10 to 18 -0.3232  0.0636  0.3980  0.4606 -0.0582 -0.6719 -0.1459  0.3085 -0.3000
#>   0.1443  0.0998  0.0246 -0.0512 -0.0111 -0.5024  0.0160  0.2148 -0.4422
#>  -0.1286 -0.4656  0.0692  0.6333  0.4340 -0.0806  0.1924  0.1004 -0.0741
#> 
#> Columns 19 to 20 -0.1513 -0.3515
#>   0.0809 -0.0498
#>  -0.0408 -0.0776
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
