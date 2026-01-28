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
#>  Columns 1 to 9  0.2469 -0.3254 -0.2730  0.2144 -0.6880  0.3725 -0.5709 -0.2692  0.4289
#>  -0.7219 -0.0790  0.1385 -0.4720  0.1534  0.6622  0.3074 -0.5153 -0.6556
#>  -0.2920 -0.5074  0.2195 -0.1915  0.4886  0.2364  0.9042 -0.8632  0.3452
#> 
#> Columns 10 to 18 -0.2590  0.1289 -0.1009  0.5023 -0.0373  0.1046  0.7714 -0.1986  0.5477
#>   0.9343 -0.1879  0.4518  0.9274  0.2349 -0.6497 -0.6867 -0.7023 -0.2588
#>   0.2068  0.6222  0.7684  0.4019  0.0403 -0.9109  0.5131  0.2466 -0.4151
#> 
#> Columns 19 to 20 -0.7505 -0.3223
#>  -0.6810  0.8650
#>  -0.8628  0.9484
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.0910  0.0973 -0.4141  0.2074 -0.6600  0.1515 -0.1659 -0.2317  0.3393
#>  -0.4234 -0.2775 -0.1147  0.7146 -0.6796  0.1980 -0.1689 -0.3448  0.2824
#>  -0.6119 -0.1143 -0.3515  0.4323 -0.7719  0.0543 -0.6881 -0.6397 -0.4336
#> 
#> Columns 10 to 18 -0.0509  0.0016  0.4581 -0.5205 -0.4438  0.0991  0.2632 -0.4168 -0.3121
#>   0.2171 -0.0221 -0.1905  0.5115 -0.4389  0.1741  0.6131 -0.5822 -0.1572
#>   0.0565 -0.5449 -0.5498 -0.2208 -0.1252  0.7453 -0.4824 -0.4044 -0.3246
#> 
#> Columns 19 to 20 -0.2109  0.2808
#>   0.6373 -0.2196
#>   0.3917  0.1472
#> 
#> (3,.,.) = 
#>  Columns 1 to 9 -0.3265 -0.0380 -0.5395  0.4421 -0.1425  0.2336  0.2181 -0.5327  0.4919
#>   0.5909  0.3350 -0.2951 -0.5009 -0.2511  0.1417 -0.0692 -0.1540  0.2542
#>   0.1997  0.3068 -0.3898  0.2159 -0.2126  0.2509  0.6025 -0.2283  0.1860
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9  0.8054  0.5985 -0.0513  0.0616 -0.2659 -0.0371 -0.4182  0.0048 -0.3129
#>   0.2301  0.8088 -0.3947 -0.6377  0.0531 -0.0722 -0.5163 -0.2979 -0.4181
#>   0.3684 -0.0626  0.0880 -0.3403 -0.4062 -0.7936 -0.7704 -0.8059  0.0990
#> 
#> Columns 10 to 18 -0.1247 -0.4941  0.1487 -0.7567 -0.5611  0.3439 -0.0727  0.4292  0.5710
#>   0.5478 -0.8902 -0.4001 -0.4406 -0.2484  0.7748 -0.3580 -0.1069 -0.5230
#>   0.1440 -0.1018  0.0346 -0.0432  0.7124  0.3141 -0.5655  0.1374  0.2399
#> 
#> Columns 19 to 20  0.1434  0.4370
#>  -0.2571  0.6235
#>  -0.3677  0.6449
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.1363 -0.0513 -0.5441  0.2868 -0.0623 -0.2715 -0.0735 -0.4606  0.1481
#>  -0.2441 -0.1516 -0.5399  0.4819 -0.3022  0.2361  0.0028 -0.2855  0.3424
#>   0.0784 -0.4385 -0.4283  0.2337 -0.1432 -0.0378  0.0186 -0.3349  0.2065
#> 
#> Columns 10 to 18  0.1786 -0.2942 -0.4970  0.0084  0.0702 -0.0663  0.0809  0.1380 -0.0450
#>  -0.1236  0.0871  0.1566 -0.1031 -0.2602 -0.0983  0.2958 -0.3303 -0.4122
#>  -0.2537  0.1783  0.3975 -0.0312 -0.3200 -0.4172  0.2944 -0.3399 -0.2931
#> 
#> Columns 19 to 20 -0.1375 -0.3028
#>  -0.4573  0.2456
#>  -0.2835  0.0394
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
