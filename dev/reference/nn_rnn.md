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
#>  Columns 1 to 9  0.0949  0.9057 -0.6047  0.2190 -0.1997  0.9803 -0.3228 -0.1294 -0.7351
#>   0.1118 -0.6393 -0.6403  0.2025  0.8542 -0.4941 -0.1364 -0.3332 -0.4199
#>   0.2889 -0.8421 -0.5301  0.0929  0.1619  0.5222 -0.0730 -0.3532 -0.7154
#> 
#> Columns 10 to 18 -0.2154  0.3258  0.6350 -0.5168 -0.0318 -0.4356 -0.1123  0.7586 -0.3848
#>   0.1776  0.2255 -0.3677 -0.1715 -0.4439 -0.4110 -0.3718 -0.3335  0.6186
#>  -0.8037 -0.3052 -0.7871 -0.7937  0.0626  0.4083 -0.3504 -0.1126  0.2872
#> 
#> Columns 19 to 20 -0.2510 -0.9076
#>  -0.0020 -0.4050
#>  -0.2151  0.0006
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.2020  0.2225 -0.4284 -0.0662  0.3729  0.5742  0.1837  0.6621 -0.0670
#>  -0.2058  0.4066 -0.3663 -0.3172  0.6638  0.2807 -0.2943  0.0816 -0.6936
#>  -0.5389  0.2473  0.0022  0.1467  0.5527  0.5008  0.4952  0.4680 -0.2857
#> 
#> Columns 10 to 18  0.4088  0.2356  0.4997 -0.0030  0.1380  0.5787 -0.7883 -0.1453  0.1758
#>  -0.8114  0.1751  0.8003 -0.0111  0.0686  0.0813 -0.3180  0.2990  0.5339
#>   0.0017 -0.1519  0.3952  0.4531  0.0062  0.2981 -0.7232 -0.1159 -0.3623
#> 
#> Columns 19 to 20 -0.3645 -0.3296
#>  -0.3356 -0.0864
#>   0.1290  0.2775
#> 
#> (3,.,.) = 
#>  Columns 1 to 9  0.0740 -0.1759 -0.4808 -0.4909  0.3974  0.1872  0.4001  0.2763 -0.1528
#>   0.3461 -0.3675 -0.3355 -0.3902  0.5409 -0.1220 -0.0710 -0.3245  0.2152
#>   0.2915 -0.2352 -0.5264 -0.5244  0.5625  0.0311  0.2441  0.1630 -0.4283
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9  0.5224  0.4550 -0.6785  0.3609 -0.3593  0.5291  0.4286 -0.1183  0.0680
#>  -0.2312  0.6110 -0.0544 -0.4936 -0.0322  0.2778  0.1945 -0.0610 -0.6433
#>   0.4406  0.3531 -0.6900  0.5485  0.4796  0.1133  0.6400  0.0570 -0.5972
#> 
#> Columns 10 to 18 -0.0222 -0.4164 -0.2153 -0.3556  0.0454  0.5291  0.0104  0.1318  0.6783
#>  -0.0422  0.4323 -0.1451 -0.7132 -0.3827  0.8175  0.2839  0.4994  0.1090
#>  -0.0863  0.3935 -0.2451 -0.2647  0.1088 -0.1455  0.1999 -0.2419  0.5767
#> 
#> Columns 19 to 20 -0.2359  0.4923
#>  -0.0447 -0.5850
#>  -0.2331  0.1098
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.3356  0.1609 -0.2495 -0.4919  0.6064  0.2938  0.0462  0.0094 -0.1428
#>  -0.0866 -0.4390 -0.2177 -0.5065  0.5069 -0.2473 -0.1305 -0.0163 -0.2143
#>  -0.1077 -0.0462 -0.2759 -0.3805  0.5250  0.3761  0.1448 -0.1179 -0.0374
#> 
#> Columns 10 to 18  0.0165 -0.0282  0.5389 -0.2717 -0.0388  0.1296 -0.3345 -0.4148 -0.1935
#>   0.1255 -0.5899  0.5106 -0.1600 -0.4733  0.3657 -0.5317 -0.1855 -0.3606
#>   0.0205 -0.0775  0.2803 -0.2281  0.2858  0.4280 -0.3797 -0.1711 -0.2632
#> 
#> Columns 19 to 20 -0.1610  0.0262
#>  -0.0025  0.0448
#>   0.0834 -0.1618
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
