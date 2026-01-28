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
#>  Columns 1 to 9  0.2244 -0.7998 -0.2026  0.9213  0.3853 -0.4040  0.2976 -0.0673  0.4454
#>  -0.1846  0.3338 -0.1712 -0.4354  0.8297 -0.6812 -0.7887  0.5604 -0.3268
#>   0.8066 -0.4494  0.2473  0.3681  0.3123  0.1391  0.3652 -0.2240 -0.2453
#> 
#> Columns 10 to 18  0.4064 -0.4578 -0.6061 -0.1781  0.2239  0.7553  0.8369  0.4483  0.7002
#>   0.1834  0.8277 -0.0057  0.1073  0.2701  0.5754 -0.4234 -0.4428 -0.5358
#>   0.8404  0.5748 -0.3523 -0.6295  0.0809  0.3726  0.1408  0.2154 -0.3226
#> 
#> Columns 19 to 20 -0.0435  0.9478
#>  -0.0364 -0.7805
#>  -0.0730 -0.4818
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.0525  0.6728 -0.1097  0.4532 -0.4428  0.2882  0.0805 -0.5829  0.2316
#>   0.3924  0.1003 -0.0430 -0.0880  0.5729 -0.4485 -0.4333  0.7068  0.0561
#>  -0.2334  0.2426  0.0647  0.1179 -0.0466 -0.3793 -0.6316 -0.1300 -0.0715
#> 
#> Columns 10 to 18  0.4182 -0.3622  0.0759 -0.6443  0.5685  0.2866 -0.0072 -0.3069  0.1978
#>  -0.6548  0.4185 -0.7514  0.4047  0.0530  0.0373  0.3219 -0.7727 -0.1424
#>   0.2967 -0.2956 -0.0340 -0.0145 -0.1889  0.1593 -0.4782 -0.6404 -0.0386
#> 
#> Columns 19 to 20 -0.0943 -0.0354
#>  -0.1591 -0.3763
#>   0.3063 -0.6623
#> 
#> (3,.,.) = 
#>  Columns 1 to 9  0.1347  0.6298  0.6716  0.0371  0.0364 -0.3736  0.0760 -0.3134 -0.5366
#>   0.1251 -0.3902 -0.4634  0.2931  0.8424 -0.1009 -0.0240  0.6928 -0.0193
#>  -0.0929  0.2466  0.4983 -0.1816  0.0074 -0.2390 -0.4911  0.2415  0.3190
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9  0.5562  0.7461 -0.2090  0.5314  0.1476  0.1061 -0.4273  0.3355  0.6972
#>   0.8202  0.4815 -0.0306 -0.6813 -0.5815  0.1610  0.7661  0.0993 -0.3739
#>   0.5966  0.2877  0.5004 -0.2120 -0.2609  0.2310  0.6091  0.0294  0.3318
#> 
#> Columns 10 to 18 -0.4731 -0.3436 -0.5240 -0.3054  0.1101 -0.2727  0.3177 -0.0488  0.7358
#>   0.9037  0.2001  0.5412  0.2193  0.0270 -0.2036 -0.8913 -0.8672  0.1635
#>   0.5722 -0.5811 -0.0916 -0.2397 -0.3167  0.1875  0.0724 -0.2059  0.5207
#> 
#> Columns 19 to 20  0.2477  0.3408
#>   0.0851 -0.6571
#>   0.0008 -0.6156
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.4654 -0.1972  0.1402  0.3678  0.5503  0.3486 -0.2664  0.4936  0.1332
#>  -0.0278 -0.0214  0.0414  0.4857  0.1600  0.1424 -0.5937  0.0294 -0.0226
#>  -0.0123 -0.3912  0.0128  0.3839  0.3053  0.1054  0.0432  0.1060 -0.0683
#> 
#> Columns 10 to 18  0.1629 -0.2277 -0.5581  0.3683  0.0179  0.3351  0.6429 -0.5393  0.1916
#>   0.1674 -0.0931 -0.0186 -0.4469 -0.0998  0.1786 -0.5019 -0.7252 -0.5913
#>   0.0782 -0.0046 -0.5364  0.2955  0.4500  0.1280  0.1119 -0.6710  0.0277
#> 
#> Columns 19 to 20 -0.0036 -0.3871
#>   0.5146 -0.5684
#>  -0.0390 -0.2293
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
