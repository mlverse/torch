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
#> Columns 1 to 9 -0.6109 -0.2477 -0.4033  0.0902  0.2760  0.9151  0.0725 -0.3148  0.1384
#>  -0.1494 -0.4742 -0.4221 -0.1770 -0.4342  0.7461  0.4733 -0.2080 -0.6915
#>  -0.4967  0.7797 -0.4751  0.3058 -0.1703  0.7933 -0.1362 -0.0835 -0.4750
#> 
#> Columns 10 to 18  0.6151 -0.8191  0.9228 -0.0911 -0.4666  0.8801  0.6732 -0.3628  0.5683
#>  -0.3247 -0.1911  0.8953  0.1111 -0.4159  0.5799  0.0325  0.1012  0.2529
#>   0.0934  0.0388 -0.9484 -0.7318 -0.4158  0.4264  0.6094 -0.5478 -0.6004
#> 
#> Columns 19 to 20  0.7409 -0.6682
#>   0.4391 -0.8075
#>  -0.2258 -0.2148
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.3017  0.0005  0.4585 -0.0952 -0.2496  0.1663 -0.2628  0.0121  0.1996
#>   0.4565  0.2767  0.0572  0.3745  0.2676 -0.0122  0.0900  0.3558  0.2202
#>  -0.4123  0.2521 -0.2087  0.1602 -0.4750  0.6799 -0.0274 -0.2035 -0.6051
#> 
#> Columns 10 to 18  0.5923  0.6870 -0.3167 -0.7386 -0.5198 -0.2483 -0.3514 -0.0524  0.2835
#>   0.4313  0.2486  0.0186 -0.1661  0.2083 -0.1921 -0.2584  0.1904  0.2037
#>   0.1566  0.4171 -0.4434 -0.1692 -0.4346 -0.5727  0.2750 -0.1005 -0.3883
#> 
#> Columns 19 to 20  0.3749 -0.0308
#>   0.2970 -0.2171
#>  -0.0501 -0.1729
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.1201 -0.1101 -0.1745 -0.3269 -0.4072  0.3155 -0.5667 -0.4551  0.0949
#>   0.1868 -0.0869 -0.1181 -0.2281  0.1955  0.4358 -0.3851 -0.1928  0.0520
#>  -0.2129  0.3099  0.0388 -0.1130 -0.2782  0.1286 -0.3346 -0.2662 -0.3117
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.2480  0.6483 -0.8087 -0.2834  0.2226 -0.6076 -0.4213  0.2717 -0.3824
#>   0.5932 -0.7614 -0.4743  0.4266 -0.1471  0.1414  0.8697 -0.2157  0.0797
#>  -0.3279 -0.1570  0.1832  0.1066  0.2176  0.2946  0.4675  0.0521 -0.1280
#> 
#> Columns 10 to 18 -0.1506  0.5246 -0.1632 -0.5945  0.0572  0.2646  0.7313  0.2628  0.6688
#>  -0.6233  0.0363  0.3517  0.4601  0.2156 -0.3649  0.1787  0.0364  0.5288
#>  -0.5841 -0.1524 -0.6148 -0.7315  0.5794  0.5478  0.1685 -0.2683  0.3029
#> 
#> Columns 19 to 20  0.4981  0.3884
#>  -0.1568  0.3427
#>  -0.3175 -0.1389
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.2870  0.2412 -0.3836  0.0973 -0.2687 -0.0594 -0.5886 -0.3119  0.2023
#>  -0.1060 -0.2995  0.5406 -0.4112 -0.1497  0.7463 -0.1427 -0.0772  0.1158
#>  -0.1576  0.1764  0.0322 -0.1685 -0.3566  0.4108 -0.1860 -0.1164 -0.2012
#> 
#> Columns 10 to 18  0.2601 -0.4094  0.1904 -0.7682 -0.1454 -0.2601  0.2624 -0.0150 -0.0624
#>   0.4374  0.1963  0.0067 -0.3516 -0.6324 -0.4711  0.3023  0.3383 -0.1184
#>   0.3570  0.0645 -0.1615 -0.4427 -0.0105 -0.3480  0.4350 -0.5230  0.0255
#> 
#> Columns 19 to 20  0.5965  0.1709
#>   0.4165 -0.1320
#>  -0.1812 -0.1894
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
