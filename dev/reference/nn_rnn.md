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
#> Columns 1 to 9  0.7005  0.0559  0.4968 -0.3010 -0.4650  0.9155 -0.5935 -0.8370  0.5976
#>   0.2299  0.3294 -0.0562 -0.4178  0.0800  0.4791 -0.1432 -0.1237  0.8600
#>   0.1713  0.0918 -0.6494 -0.0506  0.7380  0.4885 -0.3825 -0.0619 -0.1222
#> 
#> Columns 10 to 18  0.0934 -0.4194 -0.1749 -0.6633 -0.4017 -0.6233  0.1571  0.0080  0.3490
#>  -0.4257 -0.2128 -0.5560  0.2602  0.8641 -0.7947  0.3613  0.3234 -0.1081
#>   0.0254 -0.1389 -0.6782 -0.4905  0.5265  0.4226 -0.4837 -0.2657  0.0032
#> 
#> Columns 19 to 20  0.0382  0.3482
#>   0.6558  0.2165
#>  -0.9415  0.6975
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.3328  0.1065 -0.3447  0.0167 -0.1642  0.1409 -0.5756  0.1551  0.7191
#>  -0.0096  0.1175 -0.3184 -0.2514 -0.5468 -0.4263 -0.2947  0.2896  0.2385
#>  -0.3610 -0.3205 -0.5788 -0.6320 -0.0178  0.2284 -0.4880 -0.1892  0.2777
#> 
#> Columns 10 to 18 -0.1346  0.0579  0.1123  0.2873  0.1510 -0.1100  0.6277 -0.2287  0.5177
#>  -0.5753 -0.0312 -0.0940  0.3189  0.2361 -0.2815  0.0955 -0.0157  0.2890
#>  -0.8518  0.4713  0.1051  0.1548 -0.2366  0.1709  0.2534 -0.4838  0.5795
#> 
#> Columns 19 to 20 -0.1423  0.4908
#>   0.1055  0.2430
#>  -0.0966 -0.1335
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.0675 -0.1534 -0.1054 -0.5206  0.1007 -0.0305  0.0537  0.1306  0.3821
#>  -0.0623  0.1703 -0.0150 -0.4744 -0.0287 -0.2005  0.2667 -0.3327  0.2845
#>  -0.3499 -0.0802 -0.1252 -0.2853  0.2490  0.4495  0.2451  0.1291  0.8086
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.4829  0.2568  0.2697 -0.2755  0.2073  0.0288 -0.2020 -0.5043  0.6743
#>  -0.4000  0.2286  0.0942  0.5272  0.3380  0.4799  0.3610 -0.3094  0.6370
#>   0.6684 -0.3955 -0.0675 -0.0891 -0.4892 -0.3138  0.5554  0.3818  0.3221
#> 
#> Columns 10 to 18 -0.0912  0.2613  0.4913  0.3489 -0.4243 -0.4451 -0.1953 -0.2495  0.0835
#>  -0.2740 -0.1987  0.3901 -0.2421 -0.5570  0.0866 -0.6079 -0.0296 -0.0531
#>   0.5393 -0.7443  0.0352 -0.1418 -0.4339  0.8175 -0.2911  0.2210 -0.8115
#> 
#> Columns 19 to 20 -0.6806  0.0685
#>  -0.4483  0.1319
#>   0.3709 -0.6040
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.1652 -0.2559 -0.0125 -0.2416 -0.0279 -0.3891 -0.0124 -0.4073  0.5139
#>   0.2359  0.0007 -0.0568 -0.3617 -0.2609 -0.0061 -0.1104 -0.2803  0.7065
#>   0.2944  0.0604 -0.5525 -0.3188 -0.3698  0.4537 -0.1861  0.4226  0.6741
#> 
#> Columns 10 to 18 -0.8490  0.3141  0.3355  0.3375  0.2195  0.3333  0.0773  0.4498  0.2011
#>  -0.8338  0.3742  0.2761  0.0261  0.3953 -0.1079  0.2467  0.2484  0.0438
#>  -0.0260 -0.1061 -0.3534 -0.3846  0.5211 -0.6138  0.3660 -0.2242  0.1948
#> 
#> Columns 19 to 20 -0.0423  0.5241
#>   0.3065  0.1124
#>  -0.4289  0.4027
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
