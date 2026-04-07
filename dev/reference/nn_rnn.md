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
#> Columns 1 to 9 -0.3050 -0.5603 -0.0215  0.6101  0.6078 -0.6100 -0.1119 -0.5256 -0.2205
#>  -0.8993 -0.5183  0.8250 -0.4815  0.2858  0.4151 -0.0904 -0.2855  0.4268
#>   0.3485 -0.9150  0.0028  0.2682 -0.5690 -0.6546  0.9535  0.4388  0.7347
#> 
#> Columns 10 to 18 -0.3523 -0.6267  0.7504  0.1127  0.5819  0.4209  0.0590  0.4119  0.9587
#>   0.7394 -0.8235 -0.7017 -0.0944 -0.7865  0.0260  0.3767  0.2072  0.3207
#>   0.3679  0.7201 -0.2168  0.3453 -0.4727  0.5593  0.1227  0.8122  0.0721
#> 
#> Columns 19 to 20  0.4604  0.5532
#>   0.6851  0.5420
#>  -0.2511 -0.6866
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.1770 -0.0873 -0.2799 -0.5476 -0.2519  0.2968  0.6261 -0.0218 -0.1566
#>  -0.3302 -0.3374  0.3182 -0.4756 -0.5712  0.1457 -0.1666 -0.3629 -0.2173
#>   0.4952  0.0588 -0.6209  0.4224 -0.0507 -0.3670 -0.2870 -0.3260  0.7541
#> 
#> Columns 10 to 18  0.4357  0.0287 -0.4847 -0.1923  0.2888  0.6610  0.7314 -0.2712  0.2151
#>   0.3076 -0.1226 -0.4856 -0.0273  0.2693 -0.1048 -0.3182  0.1767  0.2853
#>   0.0838  0.4532  0.3652  0.7519  0.3084  0.3839  0.5714  0.3656 -0.1491
#> 
#> Columns 19 to 20  0.5615 -0.1551
#>   0.3379 -0.2743
#>   0.2622  0.2255
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.0916 -0.3387  0.0333 -0.3070 -0.3271 -0.3425  0.1183 -0.5713 -0.0984
#>  -0.0270 -0.6547 -0.2260 -0.4403 -0.2063  0.2831 -0.1043  0.4513  0.4910
#>   0.3947  0.0111 -0.0576 -0.0792 -0.4476 -0.5269  0.1800 -0.0058  0.0963
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.3574  0.4165  0.1987 -0.3475  0.3238 -0.6283 -0.0412 -0.1734  0.3632
#>  -0.3134 -0.4454 -0.7677  0.8294  0.8278 -0.6372  0.1023 -0.1570  0.3775
#>  -0.0892  0.4726  0.3732  0.8050  0.8998 -0.5289  0.3874 -0.3632  0.2111
#> 
#> Columns 10 to 18 -0.1105 -0.1629 -0.4617  0.3626 -0.3006 -0.3973 -0.2647  0.2972  0.0209
#>  -0.6906 -0.0299  0.3183 -0.4152  0.4280  0.5241 -0.1877  0.6020  0.6305
#>   0.2363  0.4361 -0.5276  0.3479  0.7381  0.2028 -0.4956  0.8023  0.3898
#> 
#> Columns 19 to 20  0.0574 -0.2130
#>   0.2779  0.5040
#>  -0.4645  0.3453
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2375 -0.1308 -0.0119  0.0239 -0.3309 -0.3261  0.1733 -0.2054  0.4084
#>   0.2601 -0.6202 -0.0135 -0.5111 -0.6029 -0.4938 -0.0337  0.2048  0.0361
#>   0.4148 -0.2796 -0.5896 -0.1912 -0.4696 -0.5838  0.4768 -0.2636  0.6677
#> 
#> Columns 10 to 18  0.1144 -0.2174 -0.2385 -0.1163 -0.0469  0.3883  0.5582  0.2205 -0.0324
#>  -0.2607  0.2435  0.1375  0.7229  0.5298  0.1091 -0.0962  0.4072  0.3034
#>  -0.0228 -0.2000  0.1500  0.2918  0.3117  0.2826  0.6186  0.3845  0.0937
#> 
#> Columns 19 to 20  0.0935  0.1983
#>   0.7199  0.3537
#>   0.5976  0.1926
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
