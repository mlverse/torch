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
#>  Columns 1 to 9 -0.7323 -0.5874 -0.2433 -0.3747  0.2921 -0.4218  0.3216  0.9125  0.3968
#>  -0.0038 -0.6767 -0.2009  0.8466 -0.7240  0.6404  0.7791  0.6428  0.2759
#>  -0.7966  0.0588  0.2629  0.1473 -0.1399  0.1129  0.1767  0.2209  0.0736
#> 
#> Columns 10 to 18  0.2195  0.7344 -0.2339 -0.2160  0.6749 -0.6571 -0.6704 -0.4715 -0.0272
#>  -0.8791 -0.7596 -0.1061  0.2371 -0.0559  0.3866  0.0492  0.0554 -0.3230
#>   0.0941  0.2645  0.8844 -0.2546  0.2871 -0.4397  0.3500 -0.7269 -0.2721
#> 
#> Columns 19 to 20  0.8413  0.3858
#>   0.6126 -0.2673
#>   0.8033 -0.0465
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.5416 -0.4283  0.1406 -0.4829  0.1446  0.3912  0.1659  0.1610 -0.6260
#>   0.0485 -0.5265  0.3859  0.0564  0.1036  0.2903  0.4840  0.1682  0.1143
#>   0.5449 -0.8473  0.2001  0.5343  0.2698  0.3395 -0.3270  0.2768 -0.6632
#> 
#> Columns 10 to 18  0.5544  0.3968  0.2875 -0.2659 -0.1279 -0.0238 -0.1464 -0.0730 -0.1714
#>  -0.3950  0.3925 -0.0178  0.2742 -0.3278 -0.1148 -0.3566 -0.3102  0.1373
#>   0.6809  0.8775 -0.1494  0.0081  0.7098 -0.2275 -0.4143 -0.4834 -0.1369
#> 
#> Columns 19 to 20  0.2230  0.6351
#>  -0.4160 -0.1278
#>  -0.0974  0.3171
#> 
#> (3,.,.) = 
#>  Columns 1 to 9 -0.1127 -0.7024 -0.1461  0.3789 -0.1598  0.6346 -0.1728  0.4875 -0.6189
#>  -0.1311 -0.6490  0.3431  0.0165 -0.0346 -0.1049  0.2461  0.1299 -0.2073
#>  -0.1872 -0.3865  0.3640  0.3611  0.0689 -0.2274  0.0370 -0.1685 -0.8497
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9 -0.6138 -0.6535  0.0572 -0.2633  0.5399  0.7785  0.4190 -0.5913  0.0351
#>  -0.3813  0.1715 -0.0470 -0.3526 -0.7166 -0.1021 -0.5092 -0.0724  0.8387
#>  -0.3568 -0.6490 -0.0235 -0.0823  0.3169  0.2891 -0.4772 -0.4817 -0.4098
#> 
#> Columns 10 to 18  0.1659  0.3839 -0.7243 -0.1727 -0.4260 -0.4924  0.5995 -0.3626  0.4260
#>   0.8756 -0.0907 -0.0630 -0.5216  0.0187 -0.4343 -0.0846  0.2410  0.3073
#>   0.2743  0.2413  0.1807 -0.3578 -0.7289 -0.1140 -0.0009  0.4765  0.5104
#> 
#> Columns 19 to 20  0.6953  0.0478
#>   0.6696  0.7459
#>  -0.1268 -0.3259
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.1957 -0.6942  0.0467  0.3213  0.4583 -0.1234 -0.1384  0.1905 -0.7143
#>  -0.7714 -0.4055 -0.0094  0.1030  0.0359 -0.2679  0.1510  0.3752 -0.4224
#>  -0.1581 -0.6524  0.1544  0.0761  0.1492  0.4774  0.0609  0.0450 -0.7134
#> 
#> Columns 10 to 18  0.3307  0.8094 -0.1352 -0.1134  0.3269 -0.3475 -0.5155 -0.1394 -0.3999
#>   0.1941  0.4928 -0.1179  0.0278 -0.2258 -0.2309 -0.2788  0.0675 -0.3899
#>  -0.1206  0.4749  0.3364 -0.2265  0.2121 -0.0746 -0.4317 -0.2801 -0.5456
#> 
#> Columns 19 to 20  0.3042  0.5019
#>   0.3565  0.1119
#>  -0.2148  0.3721
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
