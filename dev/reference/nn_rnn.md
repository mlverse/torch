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
#> Columns 1 to 9  0.8027  0.5607  0.8804  0.4150 -0.5540  0.4892 -0.4944 -0.8889 -0.0814
#>  -0.5844 -0.6929 -0.7066  0.2494 -0.5056  0.2779  0.8494  0.7433  0.0642
#>   0.5550  0.5315  0.8477  0.8521  0.4256  0.5664  0.2244  0.2915 -0.0571
#> 
#> Columns 10 to 18  0.2144  0.7605  0.5184 -0.4695  0.8733 -0.1217 -0.0770 -0.5475  0.5418
#>   0.8604 -0.5622  0.2987  0.1842  0.9218  0.8533  0.1013 -0.6929  0.2848
#>  -0.2790 -0.2733 -0.4456 -0.6357  0.4069 -0.6497  0.6625 -0.5843  0.4969
#> 
#> Columns 19 to 20 -0.7699  0.5536
#>  -0.0428  0.7656
#>   0.1203 -0.6286
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.1629  0.1318 -0.0250  0.5133  0.5764 -0.1705  0.6522  0.4912 -0.3979
#>   0.3682 -0.2500  0.1947 -0.8736 -0.6247 -0.0675  0.1484  0.2400  0.1036
#>   0.0015  0.2630  0.6980  0.7995  0.2039 -0.3633  0.2852 -0.1687 -0.4479
#> 
#> Columns 10 to 18  0.4628 -0.2085  0.2863 -0.2245 -0.4135 -0.1201 -0.1861 -0.2645  0.7538
#>  -0.4056  0.0854 -0.2612 -0.0842 -0.4138 -0.1192  0.3102 -0.3808  0.1371
#>   0.3204  0.2810 -0.1535 -0.3839  0.1120 -0.2369  0.0676 -0.3482  0.2237
#> 
#> Columns 19 to 20  0.6875 -0.1792
#>  -0.1536  0.2770
#>   0.4237  0.1032
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.5909 -0.4819  0.2139  0.2977 -0.2653 -0.3914  0.3606  0.4830 -0.6731
#>  -0.1699 -0.2458 -0.2306  0.2042 -0.3225  0.3148  0.4447  0.4209 -0.1112
#>   0.5201  0.0944  0.4060  0.3305  0.0709 -0.4718  0.1277  0.1698 -0.1206
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.3743  0.6714 -0.7002  0.3166  0.9035 -0.5708  0.0122  0.5927  0.1573
#>  -0.5567  0.4427 -0.6493  0.1156 -0.0476  0.3679 -0.3497  0.3952  0.7784
#>  -0.4214  0.0753  0.2586  0.3658 -0.0130 -0.5707  0.3372 -0.5680  0.3573
#> 
#> Columns 10 to 18 -0.2560  0.0548 -0.0372 -0.2775 -0.7561 -0.2488 -0.6530  0.6582  0.1743
#>   0.3107 -0.5372 -0.0174  0.3656 -0.9054 -0.2198 -0.2891  0.6638 -0.2308
#>   0.3642  0.4741  0.1950 -0.3302 -0.1060  0.3721  0.6654  0.3149  0.4180
#> 
#> Columns 19 to 20 -0.3507 -0.1877
#>   0.0983  0.4414
#>  -0.3541 -0.2110
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.3841 -0.1111  0.6514  0.3993 -0.5099 -0.4007  0.3019 -0.3872 -0.7179
#>   0.3635 -0.3208  0.5128  0.6593 -0.3870 -0.1766  0.0900  0.2440 -0.5782
#>   0.7243 -0.1214  0.5430  0.1635 -0.3612  0.0625  0.6991 -0.0825  0.1227
#> 
#> Columns 10 to 18  0.3618 -0.0303 -0.7709 -0.3843  0.2969 -0.5748  0.3891 -0.4009  0.3482
#>   0.5146 -0.1709 -0.5594 -0.4624  0.5733 -0.7418  0.5074 -0.3701  0.5819
#>  -0.1439  0.2089 -0.0289 -0.1651  0.3697 -0.5830  0.7457 -0.3094 -0.0559
#> 
#> Columns 19 to 20 -0.0728 -0.7085
#>  -0.0027 -0.8274
#>  -0.2602 -0.3254
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
