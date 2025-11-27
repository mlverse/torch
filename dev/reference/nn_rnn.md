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
#>  Columns 1 to 9 -0.7659 -0.4664  0.7265  0.7193 -0.9091  0.1080  0.4863  0.2011 -0.0475
#>   0.9118 -0.4978 -0.7432 -0.2284 -0.6570  0.8578 -0.3216 -0.1339 -0.6015
#>  -0.4953  0.0129  0.7925  0.5148  0.3574 -0.8450  0.7578 -0.6108  0.4064
#> 
#> Columns 10 to 18 -0.5542  0.1922 -0.8140 -0.6886  0.0332 -0.9404  0.0965  0.9672  0.3407
#>   0.5853  0.8008 -0.2017  0.7767 -0.6963 -0.1213  0.1851 -0.5766 -0.3808
#>  -0.5137 -0.1316  0.5256  0.1684 -0.0706 -0.6415  0.5292 -0.1776  0.2648
#> 
#> Columns 19 to 20  0.6440  0.6238
#>   0.3931  0.3984
#>   0.4345 -0.0052
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.1175 -0.2165 -0.5091 -0.0238  0.7589  0.0997 -0.1954 -0.1874  0.2708
#>   0.3992 -0.2161  0.3632  0.4888  0.1514 -0.3246 -0.5627 -0.3665 -0.2349
#>  -0.3871 -0.1110 -0.4752  0.4838  0.3786  0.2241  0.6330 -0.2019  0.5148
#> 
#> Columns 10 to 18 -0.1946  0.3855  0.0456 -0.1184  0.1859  0.3833  0.6390 -0.0550 -0.4398
#>   0.2681  0.1434  0.3196  0.2177  0.2814  0.2456  0.6739 -0.1503  0.4907
#>  -0.3680  0.3359  0.3359 -0.3661 -0.2206 -0.2015  0.3687 -0.4177 -0.4078
#> 
#> Columns 19 to 20 -0.6946  0.1321
#>   0.1964  0.2160
#>  -0.6506  0.3262
#> 
#> (3,.,.) = 
#>  Columns 1 to 9  0.2721 -0.1490 -0.6896  0.7175 -0.4059  0.2102  0.2193 -0.2300 -0.0961
#>  -0.0785 -0.2891 -0.2347  0.6925  0.0040 -0.3013 -0.3372 -0.0089  0.0238
#>  -0.2203  0.1837 -0.5440  0.6242  0.0761  0.3985  0.2264 -0.4336  0.2436
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9 -0.3444 -0.3285  0.0321 -0.4724  0.7147 -0.2329 -0.3475 -0.0217 -0.2189
#>  -0.3973 -0.3615  0.3691  0.0917  0.5146 -0.0061 -0.1049 -0.4850  0.3425
#>  -0.5730  0.4151  0.8179 -0.2801  0.1063  0.4577 -0.4696 -0.0492  0.0438
#> 
#> Columns 10 to 18  0.1856 -0.0655  0.6316 -0.1778 -0.0969 -0.1968  0.4373  0.0063  0.2149
#>   0.0990  0.3738  0.3810 -0.1369  0.2669 -0.1041  0.0588  0.0018 -0.0758
#>   0.1229 -0.4017 -0.3088  0.0403  0.5105  0.4519  0.6941 -0.5373  0.6954
#> 
#> Columns 19 to 20  0.6689 -0.5611
#>   0.0817  0.1187
#>   0.5514 -0.1937
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.2021  0.0120 -0.2577  0.6633 -0.0617  0.2872 -0.2439 -0.1996 -0.1352
#>   0.0791  0.0607 -0.1947  0.4644 -0.1879  0.3303 -0.4832 -0.1965 -0.4163
#>  -0.3170 -0.1649 -0.2651  0.7638 -0.1371  0.5358 -0.2498 -0.4149 -0.3514
#> 
#> Columns 10 to 18 -0.0759  0.4818  0.3314  0.2573  0.0677  0.1818  0.5982 -0.0590 -0.5850
#>  -0.0988  0.4646  0.4355  0.3592  0.3719  0.4229  0.6668  0.1865 -0.1875
#>   0.0415  0.5476  0.2305 -0.1801  0.2340  0.5992  0.6987  0.2461 -0.0976
#> 
#> Columns 19 to 20 -0.1613  0.2130
#>  -0.1136  0.4945
#>   0.0850  0.1574
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
