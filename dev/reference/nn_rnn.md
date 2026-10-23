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
#> Columns 1 to 9  0.5125 -0.2815 -0.4622  0.2144  0.4657 -0.1332 -0.3862  0.1091  0.2873
#>  -0.8573  0.4258  0.0383 -0.2848  0.3730 -0.4983  0.8481 -0.7042  0.6617
#>  -0.0839  0.2004 -0.1999  0.4028  0.1811 -0.5440 -0.1482  0.2802  0.6233
#> 
#> Columns 10 to 18  0.8179 -0.7546 -0.8789  0.1591 -0.0443 -0.6275  0.2658 -0.3900 -0.0769
#>  -0.4948 -0.0410 -0.2435 -0.4217 -0.1383 -0.7583 -0.6696 -0.5125  0.3826
#>  -0.6593  0.1613 -0.5813  0.0056 -0.7662 -0.3914 -0.0327 -0.2140  0.6937
#> 
#> Columns 19 to 20 -0.1634  0.7861
#>   0.3132  0.6115
#>  -0.4561  0.4399
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.3413 -0.0312 -0.0388  0.3895  0.2593 -0.4179  0.7748 -0.3627  0.7620
#>   0.6149 -0.0759  0.7303  0.6511  0.1857  0.4563  0.2647 -0.4869 -0.0178
#>   0.7102  0.0688  0.4253  0.4724  0.2808  0.5359 -0.1857  0.1388 -0.1241
#> 
#> Columns 10 to 18 -0.1209  0.2252 -0.5743 -0.1014  0.3190 -0.8154  0.0367 -0.0592  0.5634
#>  -0.1708 -0.5834 -0.1460  0.8758  0.5638 -0.2804 -0.0192 -0.4163 -0.1469
#>  -0.0394  0.1632 -0.3394  0.6688  0.2784  0.1386  0.3173  0.1433  0.2790
#> 
#> Columns 19 to 20 -0.3250  0.4888
#>  -0.7896  0.6394
#>  -0.6327 -0.1210
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.1265  0.0171  0.2480  0.0351  0.8337  0.0816  0.1839  0.0615  0.2660
#>  -0.3892  0.2442  0.0812 -0.1686  0.4180 -0.4086 -0.2639  0.0407  0.4379
#>   0.0928  0.4316  0.2954  0.2549  0.4781 -0.0878  0.5576 -0.4634 -0.0128
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.5234  0.4789 -0.0464  0.6562  0.3649 -0.6058 -0.3710  0.8435  0.0980
#>  -0.3812  0.5608 -0.1646  0.3702  0.0295 -0.0867 -0.1586 -0.0010 -0.1166
#>  -0.1670  0.0844 -0.7561 -0.5077  0.4779 -0.5064  0.2860  0.2518  0.5340
#> 
#> Columns 10 to 18 -0.6613 -0.1206 -0.4388  0.4356 -0.4633 -0.1400  0.0824 -0.4104 -0.0193
#>  -0.5341  0.3772 -0.2572 -0.0138 -0.2409  0.1592  0.1532  0.2207  0.2266
#>  -0.0437  0.2458 -0.0294  0.2062  0.4481  0.3443 -0.3891  0.5518 -0.0914
#> 
#> Columns 19 to 20  0.4904 -0.1637
#>   0.2346 -0.2408
#>  -0.5949  0.5022
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.4564 -0.1405  0.4991  0.3600 -0.0683  0.0441  0.6187 -0.5921  0.2881
#>   0.2558 -0.0207 -0.0226  0.2478  0.4654 -0.1879  0.3347 -0.2914  0.5527
#>   0.1053  0.4823  0.1739 -0.2592  0.5808 -0.1978  0.1756 -0.3154  0.4371
#> 
#> Columns 10 to 18 -0.1020 -0.3657 -0.0809  0.6273  0.4197 -0.6681  0.2603 -0.4039  0.1875
#>  -0.2789  0.0188 -0.2691  0.2264  0.2597 -0.5303 -0.1968 -0.2161  0.0644
#>  -0.1961  0.2910 -0.1242 -0.2614  0.2705 -0.2096  0.1369 -0.1610  0.5319
#> 
#> Columns 19 to 20 -0.7241  0.7025
#>  -0.4662  0.5717
#>   0.0559 -0.4144
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
