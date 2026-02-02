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
#>  Columns 1 to 9 -0.3763  0.3551 -0.2163  0.1002 -0.6509 -0.2254 -0.2570 -0.4182 -0.1809
#>   0.0539  0.3199 -0.2707 -0.2288 -0.8224  0.0157  0.8323  0.1804  0.2699
#>   0.5869 -0.0312 -0.2315  0.4414 -0.6852 -0.2719 -0.3958  0.1024  0.3876
#> 
#> Columns 10 to 18 -0.3032 -0.2660 -0.7903 -0.2620 -0.4205  0.4657  0.1409  0.4051  0.1209
#>   0.7360 -0.1650  0.0802  0.7569  0.2455  0.8503 -0.8808 -0.2741  0.6353
#>  -0.0563  0.2198 -0.8790  0.2950  0.4276  0.2822 -0.2442 -0.1617  0.4407
#> 
#> Columns 19 to 20 -0.8829 -0.6541
#>  -0.7876 -0.6133
#>  -0.1247  0.5365
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.1186  0.4834  0.5841 -0.1772 -0.3904  0.4730 -0.4503  0.2977  0.0265
#>   0.1385 -0.2205  0.2249 -0.4435 -0.5206 -0.2244  0.2764 -0.2844 -0.8644
#>  -0.0356  0.7127 -0.2968  0.1492  0.1010 -0.5283  0.6895 -0.2999 -0.4700
#> 
#> Columns 10 to 18 -0.5639  0.3941 -0.6137  0.1592 -0.0465  0.6305  0.0019  0.3544 -0.2227
#>  -0.2581  0.3420 -0.5794 -0.1441  0.1134 -0.1836  0.2750 -0.5068 -0.1298
#>  -0.5546 -0.3244 -0.3556 -0.5773  0.2890 -0.0199 -0.5962  0.2392  0.5093
#> 
#> Columns 19 to 20 -0.3159  0.2617
#>  -0.2191 -0.2660
#>  -0.0836  0.2087
#> 
#> (3,.,.) = 
#>  Columns 1 to 9  0.2234  0.5124 -0.0530  0.0281 -0.0280 -0.1335  0.4022  0.1757 -0.1238
#>   0.0590 -0.0408  0.0370 -0.0374 -0.4159  0.0750  0.0582 -0.4154 -0.2562
#>   0.5915  0.3138  0.2685  0.3781 -0.3249 -0.0385 -0.3071  0.2469  0.0955
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9  0.1508  0.5330 -0.1478 -0.3610 -0.5575  0.5234 -0.1467  0.1850 -0.0347
#>   0.3123  0.5671 -0.2263  0.4305 -0.2335  0.0373  0.5285 -0.0907 -0.1537
#>  -0.1381  0.8512 -0.2589  0.0702 -0.2206 -0.4764  0.2089  0.7541  0.3730
#> 
#> Columns 10 to 18  0.2840 -0.2084 -0.8404 -0.0510 -0.2923  0.2427  0.5268 -0.3403  0.7122
#>  -0.3691 -0.4907  0.3362 -0.3248  0.5001  0.1739 -0.1426  0.4353 -0.1353
#>  -0.2643  0.2214 -0.8032 -0.1004 -0.6472  0.7631  0.1747 -0.6074  0.1851
#> 
#> Columns 19 to 20  0.4205 -0.0854
#>   0.5094 -0.4854
#>  -0.8502 -0.1653
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.5453  0.4503 -0.3441  0.1033 -0.1262 -0.4017  0.2880  0.3749 -0.4724
#>   0.0742  0.1863  0.3400  0.0191 -0.1920 -0.0189  0.1154 -0.1056 -0.2038
#>   0.5734  0.4957 -0.2767  0.2259 -0.4297  0.1195 -0.0028  0.0121  0.3391
#> 
#> Columns 10 to 18 -0.3144 -0.1523 -0.3075 -0.3019 -0.0710  0.0024 -0.3365  0.3548  0.4857
#>  -0.2323 -0.0390 -0.4849  0.2730 -0.2806  0.3390  0.1394 -0.2478  0.1624
#>  -0.3678 -0.1060 -0.7058  0.0123 -0.1241  0.5395 -0.3224 -0.1608  0.3343
#> 
#> Columns 19 to 20 -0.1040  0.0188
#>  -0.1362  0.2834
#>  -0.4203 -0.0476
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
