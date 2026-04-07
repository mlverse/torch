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
#> Columns 1 to 9 -0.0013 -0.8377 -0.2916 -0.1500  0.6174  0.2819  0.7110 -0.3092  0.6582
#>  -0.5365 -0.0813 -0.7086  0.8256 -0.4214  0.1441  0.3292  0.2125  0.6040
#>  -0.3467 -0.6599 -0.5239 -0.4405  0.4440 -0.0918  0.8590  0.2378  0.5777
#> 
#> Columns 10 to 18  0.1182 -0.7709  0.2531  0.1948  0.1743  0.2012  0.5155 -0.2713 -0.3921
#>  -0.6462 -0.4349  0.8424 -0.7257  0.4408  0.4297 -0.8290  0.2266 -0.6349
#>   0.4108 -0.5794 -0.0404 -0.7501 -0.2620  0.4607 -0.9232 -0.1339  0.5774
#> 
#> Columns 19 to 20 -0.5475 -0.7334
#>  -0.8008 -0.4323
#>  -0.5211 -0.7640
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2981 -0.0893 -0.3702 -0.1711 -0.0360 -0.0299 -0.0827 -0.0700  0.6715
#>  -0.5563  0.1501  0.3443  0.7283 -0.0010 -0.0708 -0.5056  0.0204  0.6264
#>   0.2872 -0.4254 -0.2005  0.6138  0.4382 -0.2155 -0.1682  0.4549  0.6455
#> 
#> Columns 10 to 18  0.0683 -0.4800  0.2979 -0.5354 -0.3149 -0.1699 -0.4181  0.4047  0.1356
#>   0.2438 -0.1534  0.6149 -0.6537 -0.4144 -0.0450  0.0471 -0.5294  0.3879
#>   0.6007 -0.7379  0.7452 -0.6504 -0.4022 -0.3637 -0.2213 -0.4175 -0.1203
#> 
#> Columns 19 to 20 -0.0372 -0.0202
#>  -0.0214  0.2054
#>  -0.4885  0.2893
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.1209 -0.5941  0.0065  0.2455  0.2209 -0.0784 -0.3909  0.5425  0.4408
#>  -0.0432 -0.2320  0.3235  0.4114 -0.3404  0.1709 -0.1345 -0.1831  0.4738
#>  -0.1143 -0.3126  0.3545 -0.2144  0.1274  0.1508 -0.3029 -0.0672  0.4788
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.7656 -0.2403  0.3148 -0.4049 -0.2714  0.2364  0.0361 -0.1377 -0.0494
#>  -0.7934 -0.5230 -0.1944 -0.2172 -0.4211 -0.0920 -0.1049  0.1656  0.2530
#>  -0.6701 -0.0463 -0.0494 -0.5205  0.0684  0.0661  0.2464  0.4595 -0.6599
#> 
#> Columns 10 to 18  0.3324  0.8042  0.1216 -0.4240  0.4709 -0.6482 -0.3271  0.3635 -0.4819
#>   0.5053 -0.1222  0.1646 -0.5278  0.6165 -0.5683 -0.0646 -0.0881 -0.4685
#>   0.1971 -0.3850  0.6231  0.2606  0.4351 -0.1429  0.0887  0.2940 -0.3798
#> 
#> Columns 19 to 20  0.1875 -0.3926
#>  -0.3720 -0.3348
#>  -0.1926 -0.1041
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.3760 -0.1874  0.2679  0.6650  0.1628 -0.2439 -0.6429  0.2750  0.7681
#>  -0.0311 -0.4239  0.1716  0.6372 -0.5848  0.1802 -0.4713  0.3560  0.7448
#>  -0.2266 -0.4120  0.1059  0.2553 -0.3895  0.0191 -0.2136  0.2293  0.5822
#> 
#> Columns 10 to 18 -0.0833 -0.0151  0.4867 -0.5623 -0.2936  0.2508 -0.3513 -0.0951 -0.2931
#>   0.0155 -0.1023  0.4371 -0.3513 -0.3436  0.2556 -0.3578 -0.2951  0.1243
#>  -0.4700 -0.1096  0.6129 -0.4510 -0.1056  0.3750  0.1167 -0.1853  0.0492
#> 
#> Columns 19 to 20 -0.3181 -0.1781
#>  -0.0903 -0.2548
#>  -0.0974 -0.3337
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
