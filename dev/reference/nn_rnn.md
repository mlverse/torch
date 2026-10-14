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
#> Columns 1 to 9 -0.2200  0.0293  0.0137 -0.5187 -0.4739  0.8616  0.6075  0.2222 -0.0801
#>   0.5993  0.6161 -0.2653 -0.5957 -0.0805 -0.6847  0.1769  0.1061  0.3112
#>   0.0956 -0.2208  0.3063 -0.7418  0.9131 -0.5956  0.6861  0.2580  0.1449
#> 
#> Columns 10 to 18 -0.4837 -0.1488  0.1097  0.4812  0.5129 -0.5603 -0.1324  0.0008  0.7191
#>  -0.6707  0.2986  0.3557  0.4933  0.5304  0.2678  0.3567 -0.4046 -0.4663
#>  -0.3607  0.4350  0.4335 -0.8921  0.2559 -0.8013  0.1199  0.1916  0.8191
#> 
#> Columns 19 to 20 -0.2654  0.2035
#>   0.7561 -0.8062
#>   0.8014 -0.7175
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.0070  0.4314  0.3095 -0.0134 -0.0475  0.0853  0.2426 -0.0099  0.3870
#>   0.4781  0.8310  0.6022 -0.2873  0.4735 -0.6442  0.6082 -0.1277 -0.5504
#>  -0.4262  0.5293  0.5740 -0.8763 -0.1498 -0.6595  0.5228 -0.5004 -0.1851
#> 
#> Columns 10 to 18 -0.5963  0.3442  0.3528  0.3885 -0.1581 -0.4015 -0.4556 -0.4340  0.6211
#>   0.3852  0.3224  0.1727  0.6432 -0.4911  0.1726  0.5503 -0.6136  0.5843
#>  -0.2368  0.2132  0.0596 -0.1021 -0.2312  0.0174  0.8350 -0.3185 -0.6456
#> 
#> Columns 19 to 20 -0.2314  0.3314
#>  -0.1351  0.4824
#>   0.0377 -0.1437
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.2250  0.1012  0.3911 -0.2775 -0.0529  0.0338  0.4789 -0.2258 -0.0748
#>  -0.0804  0.5057  0.7645 -0.6917  0.5012 -0.6938  0.1017  0.1809  0.0797
#>   0.5613  0.3016  0.1521  0.0172  0.6173 -0.1716  0.6345  0.0568  0.0220
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.7377  0.3849 -0.0536  0.1406 -0.5680 -0.6291 -0.0867  0.7363  0.8734
#>  -0.1169  0.4160  0.4575  0.0838 -0.6003 -0.1043  0.2238  0.3573  0.5848
#>   0.1285 -0.2266 -0.1639 -0.4796 -0.2603 -0.4470 -0.0829 -0.2794  0.0478
#> 
#> Columns 10 to 18 -0.3419 -0.5896 -0.3662  0.0344  0.6977  0.6274  0.0119  0.5531  0.3586
#>   0.2706 -0.4104 -0.0055  0.1730  0.2437 -0.0757  0.4917 -0.3149 -0.2484
#>   0.1777  0.7461 -0.4904  0.3338 -0.0834  0.4476  0.6583  0.2263 -0.0373
#> 
#> Columns 19 to 20 -0.5727  0.7943
#>  -0.0376  0.6011
#>   0.4314  0.3532
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.7177  0.2648 -0.0378  0.2117 -0.2351  0.1013  0.0496  0.3527 -0.1518
#>  -0.1015  0.5567  0.4955 -0.5790  0.0769 -0.2311  0.1097 -0.2235 -0.4355
#>   0.4563  0.2660  0.3578 -0.5359  0.4817 -0.2496  0.5597  0.1396 -0.5824
#> 
#> Columns 10 to 18 -0.4084 -0.2111  0.4438  0.0960  0.2539 -0.5580  0.1175 -0.0017  0.4751
#>  -0.1684 -0.1026  0.0063 -0.0146  0.1650 -0.1726  0.0913 -0.4316  0.2003
#>   0.0116  0.0952  0.5545 -0.0893 -0.2557 -0.1749  0.3479 -0.1961  0.3420
#> 
#> Columns 19 to 20 -0.6304  0.0119
#>  -0.5999 -0.4005
#>  -0.2673 -0.0947
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
