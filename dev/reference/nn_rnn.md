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
#> Columns 1 to 9 -0.8197 -0.6143  0.3221 -0.5879 -0.1362  0.1268  0.4539 -0.4105  0.0651
#>  -0.1626  0.0339 -0.4703 -0.5555  0.2482 -0.1632  0.8846 -0.0646 -0.3980
#>  -0.1384 -0.2219  0.6471 -0.5535 -0.1111  0.4788 -0.3977 -0.9484  0.7172
#> 
#> Columns 10 to 18 -0.1065  0.8007  0.5000 -0.5220 -0.0595 -0.1995 -0.6975 -0.3821 -0.5774
#>   0.1046 -0.7322  0.2827 -0.7593 -0.7883  0.1783  0.5546 -0.1164  0.7041
#>  -0.5165 -0.5710 -0.0992 -0.2005 -0.0954  0.3490 -0.7586 -0.0182  0.4255
#> 
#> Columns 19 to 20 -0.3216  0.3370
#>   0.8432 -0.4800
#>   0.1482  0.3629
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.4403 -0.4047  0.1623 -0.1803 -0.0302  0.2228 -0.3169 -0.3688 -0.2432
#>  -0.5163 -0.1789 -0.4039 -0.5029  0.3692 -0.0475 -0.2463 -0.6360  0.6377
#>   0.3201  0.2918 -0.2299  0.1456  0.0780  0.0926 -0.6315 -0.0529 -0.3688
#> 
#> Columns 10 to 18  0.5705  0.0665 -0.0481 -0.0737  0.1097 -0.0306 -0.4378 -0.0830 -0.2073
#>   0.5553  0.0836 -0.7024 -0.0161 -0.7881 -0.5450 -0.2283 -0.2382  0.1341
#>   0.3728  0.6560  0.0414  0.0946  0.0328 -0.7667 -0.8020 -0.2127 -0.5787
#> 
#> Columns 19 to 20 -0.3360  0.2136
#>  -0.4323  0.2841
#>  -0.6253  0.2195
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.0716  0.4438  0.2700 -0.4058  0.4227  0.1210 -0.2785 -0.2548  0.3131
#>   0.2525  0.3183 -0.0714 -0.5583  0.4327  0.0595  0.3224 -0.4312 -0.0871
#>   0.4397  0.4276  0.4215 -0.4344  0.5355  0.2242  0.2855 -0.2857  0.2493
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.5411  0.7484  0.6997  0.5286  0.4021  0.8834  0.7815 -0.2797 -0.5119
#>   0.5303 -0.0704 -0.0618  0.4996  0.6646  0.3570  0.4547 -0.2228  0.1762
#>  -0.2249  0.8608  0.6036  0.7228  0.6135  0.8489  0.7913 -0.5673 -0.1399
#> 
#> Columns 10 to 18 -0.1629  0.2797 -0.0730  0.4167 -0.2143 -0.8382 -0.2372  0.5119 -0.3691
#>   0.1138  0.1469 -0.6720  0.0224  0.2391 -0.1569 -0.2138 -0.0112 -0.2275
#>  -0.1922  0.0715  0.0511  0.0831 -0.5017 -0.4280  0.3750  0.8742 -0.2688
#> 
#> Columns 19 to 20 -0.6918 -0.4546
#>   0.2190 -0.0451
#>  -0.5289  0.5217
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.0834 -0.2871 -0.0510 -0.1800  0.4223 -0.1958 -0.3376 -0.3968  0.1362
#>  -0.0240  0.4318 -0.1435 -0.4382  0.1395 -0.1772 -0.3043 -0.0994  0.2653
#>  -0.2076  0.0712  0.2268 -0.4261  0.5143 -0.2692 -0.1741 -0.2304  0.0209
#> 
#> Columns 10 to 18  0.7652 -0.0883  0.0868 -0.1554 -0.5299 -0.5614 -0.5789 -0.0649  0.0872
#>   0.4501 -0.4700 -0.0142 -0.1015 -0.3927 -0.5083 -0.2506 -0.1180  0.1355
#>   0.7558 -0.4238 -0.1315 -0.2449 -0.2685 -0.3544 -0.0713  0.3818  0.1533
#> 
#> Columns 19 to 20  0.0311 -0.1536
#>  -0.1239 -0.1721
#>   0.2458  0.0069
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
