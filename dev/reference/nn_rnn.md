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
#> Columns 1 to 9  0.1423 -0.8355 -0.4298 -0.8844 -0.4771  0.8008 -0.4003  0.7338 -0.2602
#>  -0.5263 -0.5398  0.2307 -0.5028 -0.3652 -0.5146 -0.2695 -0.6053  0.4704
#>  -0.4857 -0.5097 -0.1556 -0.6658 -0.1171 -0.6134 -0.7627  0.5994  0.0055
#> 
#> Columns 10 to 18 -0.9688 -0.0361  0.6540 -0.2222  0.0437 -0.6341  0.9025 -0.3144 -0.3179
#>  -0.7976 -0.6086  0.5850  0.5443  0.3122  0.4573 -0.5887  0.4566 -0.4837
#>  -0.6769 -0.7309 -0.1931  0.0500 -0.5658 -0.5356 -0.5708  0.7053  0.2416
#> 
#> Columns 19 to 20 -0.0333 -0.0854
#>  -0.4317 -0.1367
#>   0.4105  0.5816
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.3031  0.2786 -0.0005 -0.3754 -0.2315  0.1076 -0.7462  0.0330 -0.0121
#>  -0.8617  0.2694 -0.1110 -0.3395 -0.4537 -0.1769 -0.7083  0.2229  0.0244
#>  -0.7257  0.4617  0.1110 -0.4007 -0.4789 -0.2261 -0.2385  0.0962  0.0003
#> 
#> Columns 10 to 18 -0.6434 -0.5946  0.2749  0.3077 -0.0153 -0.6177  0.0110  0.7045 -0.0608
#>  -0.6582  0.0902 -0.0725  0.6135 -0.0006  0.2510  0.0994 -0.4230 -0.0473
#>  -0.0597 -0.1269  0.0821 -0.0533  0.5755 -0.3170 -0.2826  0.1407 -0.1815
#> 
#> Columns 19 to 20 -0.2084  0.5624
#>   0.3629  0.2856
#>  -0.6413  0.2703
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.6251  0.1071  0.1520 -0.2270 -0.7679 -0.3979 -0.3304 -0.0662  0.1302
#>  -0.0629  0.0790  0.3598 -0.0958 -0.4825 -0.4053  0.0804  0.1947  0.3332
#>  -0.4918  0.5895 -0.0672 -0.2389 -0.3495 -0.5558 -0.4584 -0.1887  0.3706
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.4197 -0.4302  0.5614  0.4931  0.3462 -0.0575 -0.0062  0.0653 -0.0625
#>   0.1184 -0.3196 -0.3218  0.0102  0.0724  0.2925  0.1435 -0.0280  0.2492
#>  -0.3804  0.5936 -0.3696  0.0312 -0.3656 -0.2366  0.0637  0.0273  0.4792
#> 
#> Columns 10 to 18 -0.5238 -0.6163  0.1174 -0.3223 -0.1826 -0.0320  0.4580  0.1073 -0.5223
#>  -0.3654 -0.2316 -0.2634 -0.3827  0.1216  0.1254  0.4327 -0.5631  0.2804
#>   0.7003 -0.1055  0.0972 -0.0770 -0.2059 -0.4588 -0.0209 -0.3806 -0.2301
#> 
#> Columns 19 to 20 -0.5213  0.2004
#>  -0.2078  0.0341
#>   0.3614  0.1793
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.4588  0.1685 -0.0784 -0.4087 -0.4740 -0.1548 -0.2655  0.0223 -0.0493
#>  -0.1626  0.0348 -0.0321 -0.2638 -0.0307 -0.3101  0.0346 -0.0077  0.5387
#>  -0.1900 -0.4940 -0.1783 -0.1060 -0.0105 -0.0718  0.0660  0.0176  0.2829
#> 
#> Columns 10 to 18 -0.5952 -0.0578  0.0654  0.4199  0.1729 -0.0768  0.2562  0.0951 -0.1542
#>  -0.0301 -0.0678  0.2698  0.0605  0.1391 -0.3689  0.1815  0.0532 -0.3238
#>  -0.4064  0.2210  0.2046  0.4343  0.2106 -0.0710  0.4940 -0.2326 -0.3005
#> 
#> Columns 19 to 20 -0.1415  0.1751
#>  -0.5198 -0.3820
#>  -0.0299 -0.6420
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
