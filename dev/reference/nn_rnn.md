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
#> Columns 1 to 9  0.7051 -0.3734  0.4338 -0.4644 -0.3677 -0.3280 -0.5642 -0.9072 -0.7533
#>   0.2327  0.1812  0.0889  0.4387  0.6317 -0.0003  0.1883 -0.0956 -0.6274
#>  -0.1161 -0.7461  0.8789 -0.7482 -0.4890 -0.6971 -0.7596 -0.4737 -0.4403
#> 
#> Columns 10 to 18  0.5526  0.1719 -0.2806 -0.5382  0.5820  0.6789  0.1030  0.5425  0.6462
#>   0.7206 -0.1415  0.3086 -0.7677 -0.7962  0.1351 -0.7614 -0.7406 -0.0231
#>  -0.4912  0.3947 -0.7919 -0.6656  0.3975 -0.4944 -0.0505  0.4157 -0.0322
#> 
#> Columns 19 to 20  0.0262  0.1340
#>   0.3389  0.0078
#>   0.6843  0.2878
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.2383 -0.4532 -0.1235  0.1586  0.0597  0.4467 -0.0239 -0.3733  0.4315
#>  -0.5899  0.2977  0.2309 -0.3574 -0.1178 -0.2239 -0.4209 -0.3059 -0.4741
#>  -0.2369 -0.6700 -0.2701  0.7461  0.3087  0.4232 -0.0685  0.0392 -0.0077
#> 
#> Columns 10 to 18  0.1687 -0.4538 -0.4888 -0.1473 -0.1001  0.1756  0.2851  0.1840 -0.2412
#>  -0.3910 -0.0652 -0.4670 -0.2407  0.3322 -0.6558 -0.1236  0.8053  0.1203
#>  -0.1097  0.4612 -0.0486 -0.2400 -0.6050  0.1964  0.1209  0.3669  0.5795
#> 
#> Columns 19 to 20  0.3606  0.4925
#>  -0.1026 -0.2728
#>   0.1769  0.1224
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.7256 -0.6138 -0.5214  0.4505 -0.1926 -0.1473  0.0161 -0.2520 -0.4700
#>  -0.2810  0.0279 -0.7740  0.7185 -0.0178  0.2988 -0.3996  0.5118 -0.0908
#>  -0.4881  0.1202 -0.1016 -0.1594 -0.0992 -0.6113 -0.1293  0.3002 -0.1935
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.2503 -0.0451  0.2011  0.0891 -0.0416 -0.3880 -0.2552 -0.1730 -0.0992
#>   0.4640  0.1789 -0.2395 -0.1433 -0.0372 -0.0020  0.4869 -0.5396 -0.7524
#>  -0.2342  0.3033 -0.2258  0.5504 -0.3855 -0.0380  0.2577 -0.4921  0.3445
#> 
#> Columns 10 to 18 -0.0863  0.2792  0.3049  0.3599  0.6456  0.5867  0.1288  0.7536  0.6821
#>  -0.5052 -0.0999 -0.4120  0.4400  0.8242 -0.5701  0.1824  0.1379  0.1204
#>  -0.5600 -0.2917 -0.4880 -0.3655  0.4290  0.2593  0.1666  0.4722 -0.2810
#> 
#> Columns 19 to 20 -0.3299 -0.4994
#>  -0.2686  0.4480
#>  -0.0222 -0.6362
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.2058 -0.1632 -0.2648  0.5715 -0.1222  0.0565  0.0145 -0.0285 -0.1220
#>  -0.1733 -0.5844  0.2176  0.3277 -0.2100 -0.0177  0.2953 -0.2845 -0.2384
#>  -0.4647 -0.0453  0.3969 -0.4804 -0.0463 -0.4959 -0.1937  0.1323 -0.7245
#> 
#> Columns 10 to 18  0.2676 -0.2363 -0.5540 -0.4212 -0.4359  0.3176 -0.2032  0.0383 -0.3023
#>   0.0912  0.3427  0.1931 -0.1852 -0.5735 -0.0469  0.1668  0.6834  0.5431
#>  -0.2831 -0.3647 -0.3077  0.0172 -0.3419  0.2346 -0.2251  0.1493  0.1689
#> 
#> Columns 19 to 20 -0.2699  0.0326
#>   0.4126  0.1596
#>   0.3313 -0.2762
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
