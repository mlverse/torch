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
#> Columns 1 to 9  0.9736 -0.3511 -0.0823 -0.0777 -0.4612 -0.2261 -0.3107  0.8960  0.2196
#>  -0.5068 -0.0488  0.0088  0.4160  0.0851 -0.2644 -0.0133 -0.2351 -0.7108
#>   0.6066  0.3889  0.0241  0.4507  0.2118  0.8816 -0.0523 -0.1215  0.1114
#> 
#> Columns 10 to 18 -0.4147 -0.5471  0.5096 -0.9559  0.8604  0.6160 -0.0885 -0.6669 -0.8128
#>  -0.8730 -0.8901  0.8902  0.5892  0.2729 -0.7515 -0.3696  0.1670  0.4309
#>  -0.4008  0.3871 -0.5161 -0.3880  0.5604  0.3230  0.1564  0.7458 -0.5139
#> 
#> Columns 19 to 20  0.6305  0.6934
#>  -0.4403 -0.7417
#>   0.3794 -0.5532
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.2386  0.0610 -0.2495 -0.2384  0.2442 -0.3804  0.3827  0.2451 -0.4257
#>   0.0932  0.5406 -0.3161  0.4741 -0.0740  0.5745 -0.5273  0.0508 -0.5067
#>   0.1964  0.2685  0.0897 -0.1936 -0.2862  0.3406  0.0167  0.4558  0.2097
#> 
#> Columns 10 to 18 -0.5425 -0.7429  0.5362  0.1607 -0.3103 -0.2004  0.2873 -0.3492  0.8106
#>  -0.4521  0.2128  0.4350 -0.6944  0.5514  0.1660  0.3277 -0.0226 -0.4073
#>  -0.2488 -0.5529 -0.1020  0.0104  0.1686  0.6822  0.2821  0.1058 -0.1387
#> 
#> Columns 19 to 20 -0.3107 -0.0859
#>   0.2645 -0.7302
#>  -0.3604 -0.0196
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.0954 -0.0568 -0.3972  0.3526 -0.0354  0.6258 -0.1919 -0.3268 -0.4758
#>  -0.2142 -0.3394  0.3264 -0.2664  0.0504  0.3271  0.2863  0.3510 -0.2414
#>   0.5149 -0.4692 -0.3258 -0.3498  0.0635  0.7531  0.0526 -0.5786  0.0002
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.1199  0.0186 -0.5316 -0.1005 -0.3996 -0.2966 -0.1531  0.6093 -0.5131
#>  -0.0881 -0.5597 -0.1126  0.4731  0.0182 -0.2280  0.5266 -0.2721 -0.0492
#>  -0.0247  0.5591  0.5118 -0.3300  0.4130  0.7657 -0.5813 -0.0273  0.4776
#> 
#> Columns 10 to 18 -0.3925  0.6076  0.2676 -0.4010  0.2650  0.6381 -0.1305 -0.3594  0.1557
#>  -0.0476 -0.0934 -0.1193 -0.4509  0.4704  0.3304 -0.1540 -0.1320  0.5342
#>   0.1849  0.4118  0.2906  0.4319  0.6239  0.6063  0.3819 -0.3613 -0.0106
#> 
#> Columns 19 to 20  0.1675  0.3957
#>   0.1140  0.0687
#>   0.4265  0.0202
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.1154 -0.0940 -0.4448  0.3645 -0.3134  0.4124 -0.3486 -0.4334 -0.2839
#>   0.2351 -0.2112 -0.2214  0.1217 -0.1370  0.5366 -0.2664  0.0634 -0.3881
#>  -0.3509  0.0665 -0.5603 -0.3583  0.4231  0.4111  0.3108 -0.6015 -0.4038
#> 
#> Columns 10 to 18 -0.0192 -0.4433  0.7001 -0.2382 -0.0923  0.1684  0.4556 -0.1934  0.0869
#>  -0.2189 -0.1192  0.3638 -0.6059  0.2163  0.3586  0.4069 -0.1450 -0.1736
#>   0.1173  0.0889  0.6644 -0.3517 -0.2950  0.0505  0.2531  0.3266  0.0733
#> 
#> Columns 19 to 20  0.1218 -0.0740
#>  -0.3404 -0.3915
#>   0.3086 -0.1717
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
