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
#>  Columns 1 to 9 -0.4182 -0.3624 -0.3511  0.4731  0.3078  0.0268  0.5331  0.7451  0.8066
#>   0.1578  0.7195 -0.3817 -0.7198 -0.9218  0.3832 -0.0911  0.9083  0.2801
#>   0.1450  0.0230  0.3251 -0.6554 -0.8559 -0.5830 -0.1037  0.8637 -0.6867
#> 
#> Columns 10 to 18 -0.4633 -0.4249  0.5863 -0.4270 -0.2836 -0.5655 -0.5385 -0.5653 -0.0782
#>   0.0346  0.1541  0.7839 -0.2087  0.1045 -0.7735 -0.1975  0.6002  0.8995
#>   0.7045 -0.5184 -0.0228 -0.5333  0.2108  0.5421 -0.4013 -0.8461 -0.5227
#> 
#> Columns 19 to 20  0.2618 -0.2120
#>   0.8594 -0.4169
#>  -0.4294 -0.5445
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.2089  0.4627 -0.1904 -0.5967 -0.1719  0.2771  0.0713 -0.3479 -0.3249
#>  -0.3769  0.0606 -0.0347 -0.6653 -0.2163  0.3965 -0.4868 -0.2239  0.0128
#>  -0.3707  0.1817  0.2806 -0.6554 -0.1310  0.1773 -0.0625 -0.5955 -0.0707
#> 
#> Columns 10 to 18  0.3355 -0.3131  0.0306  0.1554 -0.0478 -0.1972  0.4953 -0.1421 -0.6784
#>   0.5264 -0.1269 -0.4124  0.0936  0.2252 -0.1374  0.1845 -0.5046 -0.2402
#>   0.3547  0.4868  0.2351  0.0360 -0.0542  0.0231 -0.3004 -0.6513 -0.5359
#> 
#> Columns 19 to 20 -0.0300  0.0157
#>   0.2375 -0.3523
#>  -0.3331 -0.4965
#> 
#> (3,.,.) = 
#>  Columns 1 to 9 -0.3227  0.0604  0.1985 -0.4190 -0.0249 -0.1133 -0.5545 -0.2884  0.1740
#>  -0.0926  0.3253 -0.3854 -0.5555  0.0705 -0.3188 -0.4132 -0.2341 -0.1389
#>  -0.1947 -0.3599  0.1021 -0.2790 -0.0208 -0.0370 -0.2615  0.1237  0.2859
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9 -0.7505  0.3020  0.0537 -0.0548 -0.4270  0.0983  0.4509 -0.1441  0.3420
#>   0.4980 -0.4635 -0.6892 -0.6417  0.4651 -0.2116 -0.0568  0.4838 -0.1642
#>  -0.7986  0.0229 -0.3229 -0.8561 -0.2237  0.0796  0.1790  0.2333 -0.0567
#> 
#> Columns 10 to 18 -0.7776  0.1926 -0.5262  0.2823  0.0316 -0.7937  0.8304 -0.1088 -0.3563
#>  -0.3101 -0.3739 -0.0986  0.2845 -0.1501  0.8522  0.3250  0.1974 -0.0406
#>  -0.6316  0.7312 -0.3280 -0.3181  0.5591 -0.5550  0.6670 -0.0570 -0.5196
#> 
#> Columns 19 to 20  0.1466 -0.0892
#>  -0.4777 -0.0271
#>   0.1756  0.3332
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.1641  0.1316 -0.0743 -0.2066 -0.1351 -0.0195  0.0895 -0.1147  0.1430
#>   0.0939 -0.0373  0.1089 -0.5018 -0.1491 -0.0697 -0.4917  0.1533  0.0866
#>  -0.0066  0.3153 -0.0907 -0.6003 -0.5518 -0.2541 -0.0276 -0.1879 -0.1857
#> 
#> Columns 10 to 18  0.6406 -0.4544 -0.0277 -0.3324  0.2349  0.0311  0.0004 -0.1234 -0.3033
#>  -0.0314 -0.1518 -0.1115  0.1024 -0.0423  0.1476 -0.4903 -0.2340  0.4884
#>   0.1989 -0.3228  0.1292 -0.3620  0.1660  0.0864 -0.1702 -0.1524 -0.1616
#> 
#> Columns 19 to 20  0.2478 -0.5160
#>  -0.1797 -0.6570
#>   0.2016 -0.4003
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
