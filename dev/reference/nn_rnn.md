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
#>  Columns 1 to 9 -0.1609 -0.2756 -0.0218 -0.5427 -0.5010 -0.2720 -0.9556  0.4584 -0.1346
#>   0.3200  0.0218  0.4899  0.3228  0.7739 -0.0959 -0.3685 -0.1809  0.6126
#>  -0.1209 -0.3947 -0.1448 -0.3077 -0.2114  0.0100 -0.6687  0.9386  0.8521
#> 
#> Columns 10 to 18  0.1659  0.5822  0.4187 -0.5031 -0.3892 -0.4107 -0.1843  0.5928  0.1588
#>  -0.7371  0.5636  0.0435 -0.7324  0.4508  0.5974 -0.7361  0.1921  0.3389
#>   0.1096  0.0146 -0.5277 -0.2751 -0.2971  0.3200  0.2967 -0.6809 -0.5031
#> 
#> Columns 19 to 20  0.8212  0.0177
#>   0.6312  0.5731
#>  -0.1002 -0.3988
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.3273  0.1813  0.0558 -0.3096  0.0172  0.1975  0.1858  0.0616 -0.0539
#>   0.1687  0.4270  0.4264 -0.5807  0.6654  0.0798  0.0345 -0.5695 -0.3094
#>   0.3331  0.0411 -0.0180 -0.3954  0.4147  0.3973 -0.0619 -0.4808 -0.3148
#> 
#> Columns 10 to 18  0.2421  0.4278  0.2546 -0.0768  0.3473  0.2618 -0.3001  0.4777 -0.2814
#>   0.1604 -0.4568  0.0987  0.4567 -0.1576 -0.0097 -0.8494  0.2828  0.3437
#>   0.1509  0.5339  0.1399 -0.6288  0.1316  0.1934 -0.5440  0.2804  0.5888
#> 
#> Columns 19 to 20  0.3522  0.3002
#>  -0.0784  0.7108
#>  -0.0284  0.1997
#> 
#> (3,.,.) = 
#>  Columns 1 to 9 -0.2244 -0.2584 -0.1148 -0.5344  0.3973 -0.1573  0.0719  0.2624  0.0146
#>  -0.2192 -0.3089  0.4500 -0.5268  0.1249 -0.6643  0.1459  0.3483  0.4479
#>  -0.5015  0.1499 -0.2678 -0.1739  0.4211 -0.0951 -0.2657  0.1424 -0.1670
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9 -0.2438  0.3630 -0.4050  0.5537 -0.3975  0.3286 -0.6391 -0.1805 -0.2170
#>  -0.4175 -0.3882 -0.1886  0.8587 -0.2586  0.0507 -0.3669 -0.1274 -0.4186
#>   0.7980  0.8932  0.2606 -0.6526 -0.1251  0.1608 -0.1736  0.7994 -0.3750
#> 
#> Columns 10 to 18 -0.5293 -0.0595 -0.4225 -0.5720  0.5068 -0.7325 -0.4935  0.7154 -0.1961
#>  -0.4583  0.0386  0.2538  0.2432  0.0565 -0.7437  0.1757 -0.0262 -0.6533
#>  -0.0170  0.6998  0.8251  0.7430  0.3494  0.1056 -0.8678 -0.1139  0.2943
#> 
#> Columns 19 to 20  0.5835 -0.4054
#>   0.3827 -0.1531
#>  -0.7402  0.2799
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.6698  0.2917  0.3672 -0.0658  0.4648 -0.0890 -0.0738  0.4007 -0.3572
#>  -0.3399  0.3391  0.1359 -0.3207  0.0343  0.2660  0.1212 -0.1118 -0.3403
#>   0.0420 -0.1049  0.3297 -0.4887  0.2945 -0.2587 -0.0332  0.0141 -0.2216
#> 
#> Columns 10 to 18  0.4149 -0.2427  0.4333  0.2080  0.3368  0.0778  0.1114 -0.5355 -0.3797
#>  -0.0190  0.1937  0.5593 -0.1620  0.2972  0.2116 -0.3731 -0.7083  0.2413
#>   0.1659 -0.6372  0.2195  0.3930  0.0102  0.2505 -0.6252  0.0575  0.2088
#> 
#> Columns 19 to 20  0.4480  0.1144
#>   0.6625  0.5506
#>   0.0202  0.5847
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
