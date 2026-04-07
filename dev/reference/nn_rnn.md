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
#> Columns 1 to 9  0.2830 -0.4223 -0.3971  0.5760 -0.4878 -0.8758  0.9491 -0.1003  0.1355
#>  -0.5136 -0.0910 -0.5615 -0.2826  0.8107 -0.8229  0.7913 -0.6104  0.6095
#>   0.0680 -0.5498 -0.6287  0.7299 -0.9096  0.2511  0.2516  0.8830 -0.9334
#> 
#> Columns 10 to 18  0.1093  0.6411 -0.2916  0.7733  0.2490  0.2396 -0.0048 -0.7246 -0.0142
#>   0.2943 -0.6598  0.1127  0.9764 -0.7140  0.6014 -0.2643  0.2716 -0.0367
#>  -0.5099  0.6643 -0.0419 -0.8344  0.7767 -0.8910  0.1964  0.3529  0.1586
#> 
#> Columns 19 to 20  0.0657 -0.3426
#>  -0.9007  0.1028
#>   0.6207  0.1937
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.1221  0.3639  0.1243 -0.1164 -0.2877 -0.4273  0.7127 -0.0797  0.2554
#>  -0.5518  0.6584  0.2073 -0.2407 -0.1166 -0.1345  0.2078  0.2750 -0.0350
#>   0.1941  0.2199  0.1393  0.5181  0.6305 -0.1914  0.5899  0.5307  0.1003
#> 
#> Columns 10 to 18 -0.0573  0.8076 -0.3330  0.4097  0.1495 -0.1189  0.1454 -0.5697  0.1378
#>  -0.5405  0.3449 -0.2062  0.6300 -0.1605  0.1685  0.1659 -0.4037  0.3868
#>   0.2286  0.7530 -0.5883  0.3300 -0.1211 -0.4326 -0.7108 -0.6224 -0.1361
#> 
#> Columns 19 to 20  0.4377  0.6011
#>  -0.2054  0.4540
#>   0.6645  0.0155
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.1021  0.1344 -0.1564  0.0714  0.1888 -0.2604  0.4824  0.6880 -0.1765
#>  -0.0277  0.2429 -0.3359  0.2588 -0.0239 -0.4015  0.5043  0.0350 -0.1627
#>  -0.2495  0.1423  0.0232 -0.2890 -0.0723  0.3197  0.2618  0.7061  0.1855
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.6062  0.0209  0.5569 -0.1453 -0.1651  0.7652  0.4822  0.0175 -0.0636
#>  -0.2040  0.2698 -0.5728 -0.0126 -0.2469  0.1808 -0.4602  0.0194 -0.7234
#>   0.5213  0.2809  0.5731  0.2653 -0.2871  0.8840  0.1829  0.3020  0.1700
#> 
#> Columns 10 to 18  0.3590 -0.5789  0.2687 -0.3358 -0.4293 -0.0633 -0.3630  0.2569  0.1534
#>  -0.0796  0.0808  0.1772  0.2126  0.2695  0.4721 -0.1792 -0.7691 -0.1229
#>   0.2019  0.0518  0.0961 -0.1729 -0.2651  0.2482 -0.7822 -0.9180 -0.3577
#> 
#> Columns 19 to 20  0.2653  0.1172
#>  -0.3426  0.5761
#>  -0.4674 -0.6118
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2697  0.2415 -0.1468 -0.1749  0.0841  0.1255  0.1801  0.8110 -0.0442
#>   0.2333 -0.0395  0.1603  0.2921  0.2228 -0.2168  0.5483  0.5808 -0.2080
#>  -0.0048 -0.0867 -0.0891 -0.0544  0.3904 -0.3689  0.6093  0.4179 -0.0501
#> 
#> Columns 10 to 18  0.1117  0.6032 -0.0803 -0.3546  0.1329 -0.1781 -0.3278 -0.4343  0.3674
#>  -0.0832  0.5789 -0.3039  0.0409 -0.1529 -0.4097 -0.1686 -0.1243  0.0155
#>   0.4839  0.8018 -0.4658 -0.2113 -0.1515 -0.3101 -0.4679 -0.2676  0.0014
#> 
#> Columns 19 to 20  0.1258  0.6731
#>   0.5699  0.4140
#>  -0.0475  0.7002
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
