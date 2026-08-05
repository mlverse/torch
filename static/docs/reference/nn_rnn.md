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
#> Columns 1 to 9  0.8482  0.2864 -0.1451 -0.0504  0.5802  0.1810 -0.4860 -0.4347  0.0258
#>   0.3703  0.6023 -0.8538  0.6738  0.4949  0.0385 -0.7915 -0.3109 -0.7036
#>  -0.7664 -0.1933 -0.0369 -0.6386  0.1576 -0.3481  0.3551 -0.5765  0.3941
#> 
#> Columns 10 to 18  0.1947  0.0135 -0.3923  0.2456 -0.0804  0.3513  0.3525  0.4213 -0.2700
#>   0.0781  0.3971 -0.3563  0.2085  0.4587 -0.3630  0.4111 -0.2470  0.6517
#>  -0.1662  0.0179 -0.6016 -0.0077 -0.6021 -0.1966 -0.1749 -0.1420  0.7434
#> 
#> Columns 19 to 20 -0.0632  0.5249
#>   0.2351 -0.5537
#>  -0.0531  0.1515
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.4366  0.3498 -0.4265  0.3417 -0.0486 -0.4619 -0.2691 -0.6033 -0.2949
#>   0.5105  0.1461 -0.7032 -0.1662  0.3979  0.0407 -0.1119 -0.4899 -0.7239
#>  -0.0754 -0.2532  0.2380  0.1613 -0.3116 -0.0924  0.1604 -0.3021 -0.6654
#> 
#> Columns 10 to 18  0.1237 -0.0874 -0.2769  0.0153 -0.3980  0.2965 -0.0470  0.1740  0.1674
#>  -0.4320 -0.1869 -0.0651  0.0062 -0.1281  0.1116  0.4573 -0.1788  0.3866
#>  -0.2807 -0.0680 -0.0816  0.0859  0.2013  0.0014  0.1157 -0.5711 -0.1465
#> 
#> Columns 19 to 20 -0.4187  0.1758
#>   0.2600  0.0600
#>  -0.5924  0.2722
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.1755  0.1665 -0.4473 -0.1380 -0.1258 -0.4865  0.0094 -0.4242 -0.4039
#>   0.4676  0.2578 -0.4572  0.1642 -0.2467 -0.4239 -0.6048 -0.6626 -0.3869
#>   0.7284  0.2181 -0.2271  0.2933 -0.4918 -0.3673 -0.2844 -0.3894 -0.6805
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.6340 -0.4676 -0.1222  0.5859  0.3573 -0.4882  0.5625 -0.4112 -0.0759
#>   0.0008  0.3357 -0.1239 -0.0749  0.4235 -0.3046 -0.0643 -0.4059  0.3679
#>   0.5604  0.0249  0.0544  0.3106  0.1672  0.3686 -0.7828 -0.7575 -0.7330
#> 
#> Columns 10 to 18  0.2331 -0.4879  0.1906  0.5138 -0.0249  0.1928  0.5180 -0.2309 -0.2321
#>  -0.6938  0.3643 -0.3787 -0.1394 -0.4019  0.5586 -0.2465  0.0243 -0.3626
#>   0.0083 -0.6693  0.1874 -0.0032  0.2403  0.3152  0.1516  0.1451  0.6819
#> 
#> Columns 19 to 20 -0.2457  0.4975
#>  -0.4497  0.4235
#>   0.0360  0.0974
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.3847  0.0003 -0.7679 -0.1620 -0.2987 -0.2113 -0.1089 -0.2420 -0.5028
#>   0.3627 -0.1089 -0.3968  0.2663 -0.2402 -0.4713 -0.2922 -0.2948 -0.5463
#>   0.3917  0.4316 -0.3631 -0.1123  0.1141 -0.6824 -0.1885 -0.5437 -0.3228
#> 
#> Columns 10 to 18 -0.2695 -0.3698 -0.2764  0.2509 -0.2509  0.2376 -0.2683  0.2039 -0.1439
#>   0.0685 -0.0897 -0.3980  0.0743 -0.1498 -0.0733 -0.3664  0.1781  0.2761
#>  -0.1589 -0.5151 -0.2766  0.1781 -0.1930  0.2466  0.2703  0.0622  0.4015
#> 
#> Columns 19 to 20 -0.0902 -0.2783
#>  -0.3770 -0.2446
#>  -0.1592 -0.1659
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
