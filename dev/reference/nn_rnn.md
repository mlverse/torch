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
#> Columns 1 to 9 -0.3504  0.7354  0.6754 -0.8939 -0.5322  0.6439 -0.3156 -0.2674  0.9357
#>   0.5558 -0.4488 -0.3924  0.1628 -0.1483  0.8400  0.8504  0.2613 -0.6408
#>   0.0251 -0.8501 -0.8548 -0.4671 -0.5788  0.5920  0.4704  0.5419 -0.8425
#> 
#> Columns 10 to 18 -0.8704 -0.2046 -0.2835  0.7634  0.8547  0.7321  0.9335 -0.7483 -0.3275
#>   0.2332 -0.5564  0.7207 -0.5094  0.7370 -0.5589 -0.4858  0.4465 -0.1224
#>  -0.3449  0.2016 -0.3791  0.4002 -0.4086  0.1760 -0.9048  0.3664 -0.4972
#> 
#> Columns 19 to 20 -0.9602 -0.7873
#>  -0.4969  0.2056
#>  -0.8834 -0.7722
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.0613 -0.7199 -0.1193  0.1301  0.0197  0.2815 -0.2125 -0.2031 -0.6595
#>  -0.4617  0.0244  0.0932 -0.4844 -0.3576  0.4609 -0.1172 -0.6848  0.0029
#>   0.4683  0.5044 -0.3930 -0.0951 -0.5320  0.4749 -0.3077 -0.4448  0.0408
#> 
#> Columns 10 to 18  0.4515  0.0517 -0.1272  0.2992 -0.0135 -0.4274 -0.8611  0.4106 -0.6727
#>  -0.4286 -0.4460  0.7732 -0.0595 -0.5568  0.2900 -0.1835 -0.3568 -0.4334
#>  -0.3203  0.1783  0.2609  0.4240  0.6536  0.3745  0.6231 -0.8358 -0.0507
#> 
#> Columns 19 to 20 -0.1090 -0.3659
#>   0.2384 -0.0641
#>   0.0284 -0.2173
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.3534  0.6570  0.3355 -0.0162 -0.2184  0.3934  0.5036 -0.6855  0.2050
#>   0.0038 -0.3662 -0.1620  0.0619 -0.1574  0.2063  0.4451 -0.0813 -0.1068
#>   0.1431 -0.4772 -0.1927  0.4004 -0.2913  0.5851  0.1466  0.1904 -0.1320
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.9269  0.5731  0.7090  0.4526 -0.5865 -0.4785 -0.8088 -0.5855  0.1502
#>   0.1180 -0.2152  0.7002 -0.0711  0.0086 -0.1937  0.5481 -0.2914 -0.2897
#>   0.2716  0.1415  0.2804  0.3233 -0.3770  0.3466  0.1489  0.1850  0.4895
#> 
#> Columns 10 to 18  0.1336 -0.2649  0.7859 -0.8916  0.4760 -0.1341 -0.5628 -0.6985 -0.7901
#>   0.2605  0.3143 -0.6860 -0.7678 -0.2922 -0.1940 -0.1530  0.6091 -0.3157
#>  -0.5741 -0.1251 -0.7474 -0.3138 -0.6680 -0.1000  0.5622  0.1372 -0.5775
#> 
#> Columns 19 to 20  0.2746  0.6809
#>  -0.4459  0.1984
#>  -0.2923  0.5539
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.1600 -0.1273  0.1681  0.1970 -0.6032  0.4877 -0.4197 -0.1578  0.0178
#>   0.0206 -0.1661  0.2073 -0.0862  0.2210  0.3146  0.2925 -0.2828  0.2327
#>  -0.0088 -0.4997  0.3034  0.1937 -0.0426  0.3573  0.1986 -0.3256 -0.1098
#> 
#> Columns 10 to 18 -0.3669 -0.6186  0.7010  0.2920  0.0073 -0.0205 -0.6680  0.4339 -0.5426
#>  -0.3867 -0.2504  0.5209 -0.2035  0.1752  0.2274 -0.3826 -0.0996 -0.3595
#>  -0.0979 -0.0381  0.3571  0.0852 -0.4943  0.6569 -0.4625 -0.0298 -0.8677
#> 
#> Columns 19 to 20 -0.0853 -0.2213
#>  -0.0281 -0.1837
#>  -0.2122 -0.5050
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
