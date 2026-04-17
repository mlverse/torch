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
#> Columns 1 to 9  0.1746  0.2698 -0.8923 -0.2558 -0.7951 -0.6485 -0.8955  0.5525 -0.0544
#>  -0.8175 -0.4629 -0.2220  0.3162  0.6391  0.0398  0.1201 -0.2534  0.2016
#>   0.7802 -0.2065  0.2161 -0.0825  0.2929  0.6442  0.4188 -0.1134 -0.5953
#> 
#> Columns 10 to 18  0.6544 -0.0781 -0.4410 -0.9642  0.2220 -0.0104 -0.3557 -0.1990  0.5353
#>   0.4479 -0.5038 -0.1478  0.1823  0.8721  0.0652  0.0677 -0.4058 -0.6464
#>   0.4016 -0.4712  0.6093 -0.6523 -0.1402  0.0168 -0.8147 -0.4606 -0.6599
#> 
#> Columns 19 to 20 -0.6569  0.6377
#>  -0.3909 -0.3614
#>   0.6355  0.2257
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.6229  0.4706 -0.7622  0.3957 -0.7033  0.4988  0.2200  0.2753 -0.2646
#>  -0.7389 -0.6324 -0.5648  0.5788  0.0148  0.2386  0.6972  0.2248 -0.5046
#>  -0.3747  0.2592 -0.8543  0.5219 -0.1396  0.2347  0.2747  0.1511  0.1310
#> 
#> Columns 10 to 18  0.5070 -0.5117  0.5730 -0.2230 -0.1646 -0.3804 -0.1462  0.5136 -0.3099
#>  -0.1128  0.7028 -0.1204  0.3564 -0.2306  0.4135  0.3839  0.4781 -0.0354
#>   0.2226  0.1513  0.4793 -0.1806 -0.8200 -0.5080 -0.0846  0.5705 -0.6155
#> 
#> Columns 19 to 20 -0.0698  0.3669
#>   0.1751 -0.3215
#>   0.5470  0.0823
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.1134  0.1978 -0.4150  0.5153  0.1336  0.6618  0.2583 -0.2257  0.1881
#>  -0.5136 -0.4306 -0.3393  0.3797  0.2375  0.0829  0.0361 -0.2313  0.1417
#>  -0.3479  0.1302 -0.5767  0.2329 -0.1107  0.1157  0.0134 -0.1900  0.1607
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.3601  0.4102 -0.6123  0.1295  0.0999  0.0288 -0.1799 -0.0483  0.0044
#>  -0.2849 -0.1588  0.2567  0.3417 -0.2992  0.0236 -0.1933 -0.6735  0.0317
#>  -0.2022  0.5869 -0.0548  0.2377 -0.6687  0.7098  0.0682 -0.0022  0.4237
#> 
#> Columns 10 to 18  0.1095 -0.4533  0.2655 -0.1519  0.0512  0.3019 -0.2447  0.4832  0.3715
#>  -0.1810  0.3083  0.3691 -0.2731  0.8551 -0.7246  0.7268 -0.0850 -0.2916
#>   0.2022  0.0769  0.5451 -0.3947  0.0351  0.2845  0.1921  0.4491 -0.4354
#> 
#> Columns 19 to 20 -0.0914  0.4530
#>  -0.5227 -0.1352
#>   0.6214  0.3159
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.1713 -0.0946 -0.4572  0.2735  0.3633  0.0162  0.0564 -0.3140  0.0424
#>  -0.8332 -0.5633 -0.5133  0.5617  0.2038  0.4110  0.6009  0.0017 -0.1837
#>  -0.4070 -0.2702 -0.3686  0.5368 -0.0199  0.2894  0.1538  0.2462 -0.0478
#> 
#> Columns 10 to 18  0.6119  0.3413  0.1493  0.2090 -0.3101 -0.0756 -0.0863  0.4931 -0.2468
#>   0.2059  0.6337  0.5554  0.7341 -0.0138  0.3362  0.6595  0.1749  0.1299
#>   0.0811  0.2405  0.1911  0.0786 -0.0862  0.3241  0.1676  0.0613 -0.2111
#> 
#> Columns 19 to 20 -0.0375  0.2385
#>  -0.2566 -0.1814
#>   0.0464 -0.3251
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
