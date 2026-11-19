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
#> Columns 1 to 9  0.3458 -0.7930 -0.3527  0.5164  0.1734 -0.3929 -0.4567  0.5675  0.1087
#>  -0.6063 -0.2288 -0.6613  0.3812 -0.3778 -0.4879  0.6664  0.6922  0.2510
#>   0.4507  0.5866  0.2026  0.8672  0.3919 -0.6355 -0.6563  0.7996  0.0551
#> 
#> Columns 10 to 18  0.6839 -0.3295  0.4628 -0.8181  0.0390  0.0156  0.7109 -0.7149  0.2595
#>   0.0162  0.2817  0.5780 -0.8547  0.2648  0.8957  0.1435 -0.4699 -0.4915
#>   0.2496  0.7762 -0.0623  0.4187  0.7549 -0.3089  0.3056 -0.1087 -0.2774
#> 
#> Columns 19 to 20 -0.3531  0.6207
#>  -0.2903  0.2035
#>  -0.4745  0.7272
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.0309  0.1016 -0.2526  0.5326 -0.4793 -0.3463 -0.1492  0.4184 -0.1660
#>  -0.3475 -0.3440 -0.2970  0.0190 -0.3350 -0.6001 -0.1547  0.3369  0.0292
#>   0.2980 -0.1220 -0.7747  0.2614 -0.3155 -0.1728  0.2958  0.1738 -0.1689
#> 
#> Columns 10 to 18  0.6244  0.2131 -0.5274  0.3486  0.5722 -0.4752 -0.0406 -0.4874 -0.5548
#>   0.3953 -0.3971 -0.4878 -0.4683 -0.3748 -0.2942  0.1624 -0.4199 -0.7803
#>   0.3622  0.6382  0.0755  0.2675  0.6898 -0.6799 -0.5852 -0.5346 -0.3084
#> 
#> Columns 19 to 20  0.1426  0.4515
#>   0.0264  0.2980
#>  -0.6118  0.1211
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.1041  0.0051 -0.4589  0.1688 -0.0075 -0.2365  0.5026  0.1028  0.3467
#>   0.2785  0.0398 -0.3375  0.2433  0.1804 -0.4154 -0.1296  0.1796 -0.3716
#>   0.1727 -0.4335 -0.4535 -0.3588  0.1138 -0.2203  0.3204  0.2367  0.0863
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.0711  0.1839  0.5179  0.2051 -0.1576  0.1153  0.2241  0.1610 -0.5580
#>   0.3149 -0.1878  0.6252  0.0370  0.1709 -0.2345  0.2394 -0.1146  0.0884
#>   0.8730  0.3779 -0.3576 -0.7162 -0.4357 -0.3384 -0.6232  0.0327  0.3476
#> 
#> Columns 10 to 18  0.4688 -0.4266  0.1297  0.3051 -0.0365 -0.3653  0.2595 -0.5112 -0.4133
#>  -0.0262 -0.4287 -0.1472  0.1941  0.2959 -0.3648 -0.4096 -0.2480 -0.2267
#>   0.5570  0.3276 -0.3944 -0.5632  0.2267  0.3341  0.0619 -0.7773 -0.1221
#> 
#> Columns 19 to 20  0.3807 -0.5946
#>   0.1649  0.2685
#>  -0.0378 -0.4395
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.1145 -0.1773 -0.5220  0.0686  0.0927 -0.3573  0.0590  0.1951 -0.0237
#>   0.3931 -0.1236 -0.4269  0.1249 -0.1706 -0.4303 -0.0045  0.4689  0.0230
#>   0.1579  0.1568 -0.5909  0.6750 -0.5873 -0.0796  0.1292  0.4699  0.0297
#> 
#> Columns 10 to 18  0.5481  0.2079 -0.0827 -0.1937  0.3270 -0.4957 -0.3978 -0.6505 -0.1958
#>   0.3616  0.2417  0.0480 -0.4121  0.0744 -0.2745 -0.0254 -0.4962 -0.1971
#>   0.7305  0.1962 -0.2290 -0.1280  0.6941  0.0952 -0.2536 -0.2530 -0.2987
#> 
#> Columns 19 to 20 -0.3454  0.1909
#>  -0.2425  0.4730
#>   0.1261  0.5990
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
