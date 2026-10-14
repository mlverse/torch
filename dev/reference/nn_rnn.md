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
#> Columns 1 to 9  0.1590  0.4412 -0.1217 -0.3907 -0.0519  0.9079 -0.5143 -0.9936 -0.2570
#>   0.1063  0.6034 -0.2126  0.5632  0.4650  0.2102 -0.4094 -0.7210  0.2983
#>  -0.2893  0.1714  0.0091  0.6163  0.4433 -0.1222 -0.5360 -0.7693  0.0546
#> 
#> Columns 10 to 18 -0.8639  0.7523  0.6349  0.0191  0.9003 -0.7388 -0.7875 -0.6887 -0.6633
#>   0.1699 -0.3294  0.3786 -0.5572 -0.5529 -0.4585  0.0084 -0.3418  0.2832
#>  -0.2541 -0.1434  0.2717 -0.4203 -0.3322  0.0014 -0.1189 -0.7954  0.3263
#> 
#> Columns 19 to 20 -0.6278 -0.2113
#>  -0.4651  0.6018
#>   0.3967  0.8413
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.3943 -0.1565 -0.2288  0.3142  0.1510  0.3103  0.5644 -0.0610  0.0056
#>  -0.1314 -0.1797 -0.2140  0.2569 -0.0811  0.4049  0.0495 -0.8175  0.2293
#>  -0.4345 -0.6855 -0.2637  0.1332 -0.1065 -0.1743  0.1838 -0.7338 -0.0855
#> 
#> Columns 10 to 18  0.5075 -0.0340  0.4157  0.3712 -0.1259 -0.2751  0.2873 -0.5363 -0.4691
#>  -0.0433  0.0300  0.0344 -0.3089 -0.5488 -0.4695 -0.3635 -0.0020 -0.0063
#>  -0.3803 -0.1580  0.2897 -0.2950 -0.6267 -0.0303 -0.4558  0.0523 -0.3621
#> 
#> Columns 19 to 20  0.4663  0.6984
#>   0.2010  0.0035
#>   0.4772  0.2737
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.2601 -0.5690 -0.3482  0.3183 -0.5083  0.3900 -0.5166 -0.7923  0.1926
#>  -0.5549 -0.3488 -0.6558  0.5393 -0.0191 -0.1541 -0.2679 -0.5730  0.0966
#>  -0.5889 -0.6244 -0.7276  0.3170 -0.0825 -0.7212 -0.2487 -0.0970  0.2696
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.0486  0.4708  0.4129 -0.2548 -0.0531  0.3620 -0.3439 -0.1220 -0.3957
#>  -0.3804 -0.0472  0.3765  0.7868  0.8619  0.7635  0.7635 -0.8201  0.4742
#>   0.6686  0.2727  0.9226  0.1131  0.2773  0.8213  0.7136 -0.5109  0.1113
#> 
#> Columns 10 to 18 -0.6605  0.5206  0.7017 -0.5439 -0.5592  0.8041 -0.3487 -0.2646  0.8870
#>  -0.4398 -0.4627 -0.4044  0.6050  0.1633  0.8840 -0.6474  0.5921 -0.6095
#>   0.0443  0.7459 -0.2941 -0.5134  0.1865  0.5723 -0.7637 -0.0856  0.4387
#> 
#> Columns 19 to 20  0.5631 -0.7164
#>   0.6793 -0.1764
#>   0.4909  0.2198
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.1429  0.0963 -0.5172  0.3080 -0.0996  0.0383 -0.5205 -0.7519  0.0673
#>  -0.8291 -0.6204 -0.5531  0.2600  0.0825 -0.3958 -0.1532 -0.1699  0.0981
#>  -0.5948 -0.3881 -0.2449 -0.3463  0.4657 -0.3039 -0.0334 -0.4929  0.2036
#> 
#> Columns 10 to 18 -0.1934  0.1838  0.1557 -0.0024  0.0729  0.0125 -0.4603 -0.4975  0.2004
#>  -0.1923 -0.3146  0.4015 -0.4324 -0.3191  0.4442 -0.0426  0.6958 -0.4053
#>  -0.1528  0.1650  0.2822 -0.5565 -0.2365 -0.1279 -0.1912  0.1763  0.1336
#> 
#> Columns 19 to 20  0.2143  0.4738
#>   0.2947  0.4164
#>   0.2651 -0.1592
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
