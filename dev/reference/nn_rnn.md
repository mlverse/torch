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
#>  Columns 1 to 9 -0.7020  0.0176 -0.3694  0.7342  0.2399 -0.7832  0.5800 -0.7389 -0.1946
#>  -0.7744  0.6163  0.2224  0.9346  0.7123 -0.3731  0.4948 -0.4382 -0.8941
#>  -0.4503 -0.2009 -0.3195 -0.1745  0.2650  0.3409  0.5371  0.3842 -0.3458
#> 
#> Columns 10 to 18  0.6579 -0.7225 -0.0072  0.6882 -0.5550  0.2850 -0.3350 -0.9368  0.4023
#>   0.1513 -0.4861  0.2858  0.1932  0.1451 -0.2826 -0.3798 -0.6207  0.3975
#>   0.5648 -0.0563 -0.6085 -0.3333 -0.2140  0.5832 -0.5706 -0.1957  0.2409
#> 
#> Columns 19 to 20 -0.7721 -0.3318
#>  -0.4840 -0.8064
#>  -0.3239 -0.4490
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.1970  0.3052 -0.2687 -0.2436 -0.1008  0.4756  0.5979  0.3606 -0.3930
#>  -0.0417  0.2313 -0.1669 -0.4060  0.6446  0.2465  0.7854 -0.1512 -0.5089
#>   0.3787  0.0003  0.3840 -0.4049  0.7528  0.6058  0.0665  0.3813  0.3220
#> 
#> Columns 10 to 18 -0.7978 -0.8500 -0.1269  0.4942  0.7710  0.0945 -0.5341  0.3162  0.6448
#>  -0.5632 -0.6814 -0.5927  0.0336  0.4194  0.1743 -0.7736 -0.1797  0.7178
#>  -0.4693  0.3065 -0.0637 -0.4775 -0.2876 -0.2131 -0.5655  0.3993  0.4987
#> 
#> Columns 19 to 20  0.3621  0.1325
#>   0.4390  0.3630
#>   0.3638 -0.4015
#> 
#> (3,.,.) = 
#>  Columns 1 to 9  0.1564 -0.4207  0.0971 -0.1825  0.4703 -0.4309  0.3137 -0.3325  0.1887
#>   0.4144 -0.0817  0.5118 -0.1968  0.7200  0.3086  0.1680 -0.0006  0.0012
#>   0.2714  0.5198  0.4725 -0.3791  0.7349  0.4758 -0.0247  0.2150  0.1999
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9  0.0427 -0.2642  0.5743  0.1969 -0.3929  0.2919  0.1403  0.5509  0.0952
#>  -0.2791  0.7348 -0.0778  0.0269  0.6384 -0.2734 -0.3137  0.4901  0.0009
#>   0.8867  0.3673 -0.0909  0.2387 -0.6786 -0.1865 -0.6570  0.1227  0.7782
#> 
#> Columns 10 to 18  0.0575 -0.5918 -0.2437  0.5481  0.2779  0.3087  0.3644 -0.3702  0.4620
#>  -0.2099  0.6197  0.7969 -0.7310  0.8029  0.4810  0.8752  0.4317  0.5012
#>   0.1418 -0.1136  0.3734 -0.2147 -0.3478  0.4612 -0.0773  0.7640  0.1327
#> 
#> Columns 19 to 20  0.9178 -0.3424
#>  -0.1913  0.7495
#>   0.1840 -0.6322
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.1035  0.1231  0.0100 -0.1525  0.3941  0.0829  0.5611 -0.0434  0.0785
#>   0.0422  0.0407  0.3001 -0.7745  0.0395  0.3437  0.0005  0.0977 -0.3957
#>   0.1577  0.5695 -0.4061 -0.3239  0.7502  0.6881 -0.0413 -0.2472 -0.3793
#> 
#> Columns 10 to 18 -0.2747 -0.0717 -0.5049  0.1196 -0.3594  0.0734 -0.4763 -0.0687  0.3070
#>  -0.2669  0.5808 -0.2034 -0.5473  0.2510 -0.0714 -0.5860 -0.3792  0.3310
#>   0.2521 -0.1054  0.0150 -0.0111 -0.5537 -0.6389 -0.7290  0.3219 -0.1412
#> 
#> Columns 19 to 20  0.2816  0.0448
#>   0.3190  0.3969
#>   0.2926  0.5304
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
