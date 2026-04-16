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
#> Columns 1 to 9  0.4466 -0.1112 -0.3646  0.1855  0.5848  0.8225  0.5672 -0.5251 -0.8251
#>   0.3722 -0.8934 -0.2308  0.4402 -0.6634  0.3017  0.4947  0.1507 -0.1036
#>  -0.4700 -0.7628 -0.7675  0.1952 -0.1076  0.6299 -0.1237 -0.3198  0.1374
#> 
#> Columns 10 to 18  0.5754 -0.6301  0.8493  0.8060 -0.5782 -0.3936  0.0165  0.1035 -0.3335
#>  -0.3461 -0.8420 -0.4936  0.1395 -0.8163 -0.4720  0.9418 -0.4552 -0.6632
#>   0.4040 -0.3140 -0.6549 -0.3243  0.1489  0.1094  0.8432  0.1919 -0.8698
#> 
#> Columns 19 to 20 -0.6133  0.1456
#>  -0.2889  0.6383
#>  -0.4020 -0.0443
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.6736 -0.5245 -0.2960 -0.3947 -0.4034 -0.5878  0.5109 -0.6062  0.2417
#>   0.5050 -0.3856  0.1312 -0.1654 -0.2590 -0.2943  0.5554  0.3254 -0.2482
#>  -0.1433 -0.3155 -0.0369  0.1048  0.0861 -0.1670  0.2996  0.1949 -0.1291
#> 
#> Columns 10 to 18  0.0098  0.4498 -0.6742 -0.0549  0.7708 -0.0067  0.6549 -0.6345 -0.3546
#>   0.2150 -0.3555 -0.6180  0.4325 -0.0453 -0.5169  0.6422 -0.4741  0.0479
#>  -0.3815 -0.2903 -0.2700 -0.1952 -0.4387 -0.7300  0.3908 -0.2938  0.1350
#> 
#> Columns 19 to 20 -0.1950  0.3748
#>  -0.7925  0.1140
#>  -0.5110  0.7333
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.1658 -0.3113 -0.3573 -0.0269 -0.4761 -0.2008  0.1561  0.4091  0.0133
#>   0.5626 -0.5822  0.1082 -0.0795 -0.0802 -0.2862  0.3779 -0.0013  0.3152
#>   0.2912 -0.3624 -0.1204 -0.1755 -0.0662  0.2983  0.7758 -0.3077  0.4387
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.0502 -0.1562  0.3430 -0.1667  0.3539 -0.2070  0.5546  0.6116  0.5331
#>  -0.0841  0.4860  0.3383 -0.0376 -0.5407 -0.1046  0.6015 -0.0541  0.5503
#>  -0.5138 -0.0286  0.0237  0.2555  0.2188  0.1084  0.3622  0.3468  0.1800
#> 
#> Columns 10 to 18  0.2698  0.1705 -0.2301  0.3147  0.1481 -0.3885  0.0681 -0.3175  0.5085
#>  -0.2935  0.4818  0.0508  0.1331  0.7710 -0.7227 -0.9040 -0.2649 -0.0668
#>   0.8341 -0.2914 -0.1638 -0.4013 -0.3812 -0.0632  0.1603  0.3081  0.6200
#> 
#> Columns 19 to 20  0.3537 -0.4213
#>   0.6702 -0.1432
#>   0.1983  0.6540
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.4622 -0.4288 -0.1492 -0.1879  0.0284  0.1992  0.6711 -0.2111  0.2872
#>  -0.0086 -0.3806  0.0497  0.0978 -0.1604  0.2588  0.4922 -0.1165  0.1808
#>   0.6396 -0.3300  0.0788 -0.0062 -0.0372  0.2429  0.1972  0.2539  0.2091
#> 
#> Columns 10 to 18 -0.2460  0.0458 -0.1658  0.2685 -0.3765 -0.0743  0.3088 -0.4712 -0.0158
#>  -0.1139 -0.5205  0.0760  0.1749 -0.0685  0.0363  0.1524 -0.2672 -0.2240
#>   0.1385 -0.3311 -0.1144  0.4490 -0.0267  0.1744  0.4171 -0.4025  0.0580
#> 
#> Columns 19 to 20 -0.1615 -0.0108
#>   0.0743  0.4289
#>   0.1285 -0.1953
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
