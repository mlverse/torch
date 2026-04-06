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
#> Columns 1 to 9 -0.4986 -0.1002 -0.4664  0.1787  0.2407 -0.3052  0.5056 -0.7278 -0.2423
#>  -0.8732 -0.3856  0.3815  0.4084  0.3661  0.7057 -0.7296 -0.8707 -0.5935
#>   0.5572 -0.5179 -0.2461 -0.5621  0.0220  0.8800 -0.6355  0.8466  0.8882
#> 
#> Columns 10 to 18 -0.6773 -0.0941  0.4430 -0.4328 -0.7309 -0.5136 -0.6040  0.0644 -0.7417
#>   0.4560  0.1534 -0.0064 -0.2584 -0.8392 -0.5868  0.8027 -0.3086 -0.3138
#>   0.0674  0.8314 -0.8624  0.2432  0.5178  0.3386  0.8611  0.4873  0.9183
#> 
#> Columns 19 to 20 -0.1953 -0.5927
#>   0.6647  0.6122
#>  -0.9404  0.5939
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.0217  0.6609 -0.3270  0.0649  0.7846  0.3950 -0.1655 -0.6441  0.0063
#>   0.2241 -0.5173 -0.6689 -0.1514  0.1459  0.6727  0.1564 -0.7391  0.2708
#>  -0.0966  0.0862 -0.0689 -0.7727  0.1004 -0.2936  0.1763  0.1228 -0.2838
#> 
#> Columns 10 to 18  0.1566 -0.3901 -0.0758 -0.0859 -0.1265 -0.5769  0.5780  0.3735 -0.2593
#>   0.0650  0.1469 -0.0770  0.6429  0.2350  0.0733  0.7022 -0.6104 -0.1034
#>   0.6877  0.5444 -0.4723 -0.0008  0.3208  0.5097 -0.0995 -0.6246  0.5541
#> 
#> Columns 19 to 20 -0.4727  0.1437
#>   0.2433  0.4649
#>  -0.3432  0.2396
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.4457 -0.1081 -0.4360 -0.0134  0.3569  0.2215  0.3515 -0.6984 -0.1501
#>   0.3378 -0.3870 -0.4451 -0.1604  0.5241  0.3956  0.1516 -0.3165  0.1397
#>  -0.2865 -0.6666 -0.6953 -0.0373 -0.4253  0.2234  0.1567  0.0978  0.3837
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.0985 -0.6011  0.2917  0.8593  0.2831 -0.5138 -0.2895  0.5892  0.4309
#>  -0.1727  0.6722 -0.5479 -0.0225 -0.5280 -0.5168 -0.2523 -0.1648 -0.7263
#>  -0.1402 -0.2644  0.2838  0.2916 -0.3562 -0.6681  0.6401  0.4618 -0.0288
#> 
#> Columns 10 to 18  0.5021  0.4475 -0.4647 -0.2717 -0.5991  0.6227 -0.0124  0.1630  0.2212
#>   0.4983  0.1418  0.2756 -0.8041  0.4630 -0.5136 -0.2461 -0.3575 -0.4074
#>  -0.1315  0.0650 -0.9131 -0.3030  0.6670  0.6651  0.4816 -0.5438 -0.2403
#> 
#> Columns 19 to 20 -0.4268  0.6652
#>   0.0788 -0.1525
#>   0.1376  0.5743
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2668 -0.0452 -0.6082 -0.4603  0.6882 -0.3865  0.4308 -0.4601  0.0771
#>   0.1032  0.0740 -0.0706  0.3345  0.5626  0.5001 -0.1720 -0.3358  0.1752
#>  -0.0264  0.3771 -0.8520 -0.3365  0.5179 -0.2336  0.4461 -0.0705  0.1304
#> 
#> Columns 10 to 18  0.0906  0.2550  0.3650  0.0438 -0.0513 -0.2861  0.0903 -0.1745 -0.4912
#>   0.5054  0.0723  0.1287  0.1822 -0.1823  0.0199  0.6164 -0.4334 -0.1851
#>   0.2787  0.0514 -0.1279  0.3134  0.1625  0.1054  0.1529 -0.0228  0.0925
#> 
#> Columns 19 to 20  0.4450 -0.0604
#>  -0.0096 -0.1516
#>  -0.6079  0.1213
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
