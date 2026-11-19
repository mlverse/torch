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
#> Columns 1 to 9 -0.8474  0.5992  0.4333 -0.8829 -0.1547  0.8129 -0.2334 -0.6607 -0.8155
#>   0.1281  0.0302 -0.7424  0.7189  0.4010 -0.9603 -0.0330 -0.7329  0.6664
#>   0.1508 -0.5503  0.8047 -0.6479  0.0047 -0.3792  0.6847 -0.3578  0.0810
#> 
#> Columns 10 to 18  0.9386 -0.4859  0.1988 -0.3092 -0.8724  0.6816 -0.4362  0.3842 -0.4040
#>   0.5328  0.0525 -0.7767  0.0913 -0.7298 -0.0411  0.2896  0.1580  0.7952
#>   0.7255 -0.6327 -0.2600  0.4097 -0.0096 -0.0029 -0.2456 -0.4415  0.7638
#> 
#> Columns 19 to 20 -0.3651  0.8899
#>   0.9767 -0.7811
#>   0.6667 -0.4013
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.3390 -0.0147  0.1587 -0.1204 -0.0052 -0.2529  0.3746  0.5201 -0.1334
#>  -0.4485 -0.2295 -0.2694 -0.4077 -0.2592 -0.0642 -0.6259 -0.7924 -0.4111
#>  -0.0495 -0.2295 -0.5415 -0.3802  0.2374 -0.1657 -0.0585 -0.5502 -0.0531
#> 
#> Columns 10 to 18  0.5914 -0.0180  0.0349  0.1256 -0.1846  0.6224  0.1516 -0.5588 -0.3020
#>   0.3966 -0.4099 -0.5901  0.1468 -0.3515 -0.2407 -0.1242  0.3199  0.4150
#>  -0.1582 -0.2387 -0.6800  0.3348 -0.3620  0.2772 -0.4190  0.6191  0.5148
#> 
#> Columns 19 to 20 -0.0678 -0.6734
#>   0.6307 -0.2348
#>   0.6555 -0.7760
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.1298 -0.4136 -0.3314  0.0316  0.5198 -0.2329  0.3689 -0.4247 -0.0391
#>  -0.2057  0.1526 -0.2424 -0.3817  0.0565  0.2233 -0.1141 -0.5304 -0.3629
#>  -0.5158  0.3011  0.2128 -0.4000  0.2426 -0.2972 -0.3849 -0.8812 -0.2461
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.2973  0.1797 -0.1475 -0.2129  0.2790 -0.0413 -0.0896  0.5218  0.0441
#>  -0.2221  0.3216 -0.5689  0.3240  0.0634 -0.2030 -0.5613  0.5680  0.1381
#>  -0.6183  0.4440 -0.3936  0.2571  0.8066  0.5226 -0.6715  0.7876  0.4961
#> 
#> Columns 10 to 18  0.2188  0.3667 -0.1113  0.1807  0.0085  0.1171 -0.4133  0.4743  0.0459
#>   0.6746 -0.2124 -0.6434  0.2504 -0.5689  0.1118  0.7718  0.0686  0.7563
#>   0.0585  0.4622 -0.0846  0.6203  0.0345  0.1406 -0.0198  0.3550  0.1561
#> 
#> Columns 19 to 20  0.2290  0.6519
#>  -0.4536 -0.4759
#>  -0.1257  0.0514
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.1496  0.0807  0.1927 -0.0708  0.2250 -0.2289 -0.0375 -0.4535 -0.1177
#>  -0.5726  0.0713 -0.0949 -0.1714 -0.2219 -0.3720 -0.6130 -0.3631 -0.3770
#>  -0.2825 -0.4126  0.2225 -0.4686  0.5272 -0.1816 -0.1786 -0.3107 -0.0474
#> 
#> Columns 10 to 18  0.1301 -0.0140 -0.2244 -0.0379 -0.1493  0.0982 -0.1875  0.3282  0.2019
#>  -0.2775  0.0783 -0.5492 -0.6586  0.0038 -0.2762 -0.3355 -0.2130  0.4833
#>  -0.0980 -0.0079 -0.2516 -0.3642 -0.4923  0.1438 -0.1755  0.0028  0.6914
#> 
#> Columns 19 to 20  0.3334  0.0908
#>   0.3352 -0.4333
#>   0.2277 -0.4127
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
