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
#>  Columns 1 to 9  0.5336  0.7385 -0.1702  0.8946  0.5489  0.4577 -0.6100  0.3837 -0.2113
#>   0.5149  0.5170 -0.3932  0.2605 -0.7742  0.6378 -0.8764 -0.6030  0.6703
#>  -0.6465 -0.7565  0.2658 -0.4789 -0.6121 -0.7822 -0.6875 -0.2040  0.8524
#> 
#> Columns 10 to 18 -0.4313 -0.3112  0.5503 -0.4635  0.4659  0.4330 -0.4477 -0.4642 -0.1739
#>   0.6729  0.3674 -0.7854 -0.3010 -0.0329  0.6652 -0.6028 -0.3420 -0.7138
#>  -0.4576  0.4652 -0.1436 -0.8748 -0.3997  0.5135  0.5969 -0.5472 -0.7552
#> 
#> Columns 19 to 20 -0.1085 -0.8933
#>   0.2050 -0.7869
#>   0.4866 -0.8155
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.3378  0.5512 -0.6012 -0.1774  0.0709  0.3978 -0.5676  0.7610  0.2318
#>  -0.6532 -0.2849 -0.2463  0.1308 -0.3581 -0.2793 -0.3553 -0.0813  0.5263
#>  -0.6874  0.0305  0.5229  0.5767  0.3622 -0.3725  0.2080  0.0476  0.2523
#> 
#> Columns 10 to 18 -0.0127  0.5236 -0.1666 -0.2486 -0.4238  0.1061  0.1221  0.0427  0.0751
#>  -0.4802 -0.3173  0.0086 -0.3551  0.0449  0.5629 -0.2253 -0.8216 -0.3838
#>   0.1921 -0.2343  0.0920 -0.5654 -0.2106  0.2449 -0.4299 -0.2227 -0.1978
#> 
#> Columns 19 to 20 -0.0165 -0.5769
#>  -0.5177  0.0348
#>  -0.5863  0.0111
#> 
#> (3,.,.) = 
#>  Columns 1 to 9 -0.3301 -0.0342 -0.2219  0.5956 -0.2213 -0.2874  0.2217 -0.2680  0.3362
#>  -0.7505 -0.0707  0.3008  0.5669 -0.2736 -0.5126  0.5113 -0.0566  0.4427
#>  -0.3840  0.2430  0.4328  0.5987 -0.1425 -0.2271 -0.1556  0.2931  0.0870
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9  0.4746  0.2448  0.2263 -0.3123  0.1691 -0.0862 -0.2115  0.1364  0.3606
#>  -0.0427 -0.2690 -0.1439  0.1881  0.3559 -0.7563 -0.7234 -0.3362  0.3827
#>  -0.3051 -0.2255  0.5426 -0.4946  0.3450 -0.2787  0.1788  0.2281  0.6788
#> 
#> Columns 10 to 18 -0.0465 -0.3255 -0.0975  0.1391  0.5480 -0.2833  0.0816 -0.5084  0.0402
#>   0.2749 -0.0250 -0.6497  0.3652 -0.4857 -0.3534  0.0736 -0.4114 -0.6087
#>  -0.3513 -0.2443  0.0887  0.4410 -0.5391  0.2359  0.8310  0.4941  0.7023
#> 
#> Columns 19 to 20 -0.0271  0.0505
#>  -0.3850  0.5837
#>  -0.9213 -0.6088
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.2549  0.1756  0.2124  0.2662 -0.0108 -0.1531 -0.3832  0.1307  0.3121
#>   0.2063  0.3140 -0.0864  0.2344  0.1531 -0.0108 -0.2327  0.1950  0.3488
#>   0.0135 -0.0336  0.2284  0.6409 -0.0711 -0.2831  0.3964 -0.1305  0.6077
#> 
#> Columns 10 to 18  0.2224  0.1722 -0.4920 -0.1563 -0.0371 -0.3329  0.0425 -0.1240 -0.3245
#>  -0.4126 -0.1157 -0.4409 -0.0087  0.2259  0.0846 -0.2070 -0.5810 -0.1939
#>   0.2366  0.2339 -0.3316 -0.6490 -0.1148  0.1552  0.2948  0.0804  0.2170
#> 
#> Columns 19 to 20  0.0610 -0.3165
#>  -0.1835 -0.5385
#>   0.0863 -0.3652
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
