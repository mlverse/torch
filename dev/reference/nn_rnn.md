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
#>  Columns 1 to 9  0.3547 -0.6190  0.5223 -0.5057 -0.2564 -0.0219 -0.9601 -0.0410 -0.6800
#>  -0.2157  0.3987  0.9264  0.0323 -0.3094 -0.1742 -0.4000  0.6093 -0.0689
#>  -0.6643 -0.1576 -0.3740 -0.7296 -0.0185  0.1869 -0.0414 -0.0419  0.2401
#> 
#> Columns 10 to 18  0.5043  0.6123 -0.0043  0.2301  0.1515 -0.7714 -0.1117  0.2751 -0.5599
#>   0.8544  0.6878  0.4722  0.5794  0.1338 -0.2772 -0.4542  0.1199 -0.4015
#>  -0.2547 -0.2389 -0.4725  0.7778  0.3876  0.2904 -0.0675 -0.7929 -0.5718
#> 
#> Columns 19 to 20  0.3250 -0.8597
#>   0.6977 -0.5926
#>   0.5429 -0.0109
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.1167 -0.0912  0.2444  0.0630  0.3435 -0.8376  0.5564  0.0185  0.1179
#>   0.1452  0.1186 -0.2028  0.2983  0.2714 -0.4635  0.4615 -0.3078  0.4832
#>  -0.0130 -0.3963  0.0552 -0.6213  0.3521  0.3211 -0.4047  0.2092  0.2398
#> 
#> Columns 10 to 18 -0.5091  0.0322 -0.3994  0.0274 -0.1549 -0.5214 -0.1092  0.1968  0.1334
#>  -0.0539  0.0338  0.0246 -0.3476  0.2041 -0.5296  0.2264  0.5774 -0.3523
#>   0.0914  0.2174 -0.4546  0.1138  0.4103  0.1705 -0.1221 -0.4673 -0.2531
#> 
#> Columns 19 to 20 -0.2615  0.1763
#>  -0.3653  0.2096
#>   0.2600 -0.1663
#> 
#> (3,.,.) = 
#>  Columns 1 to 9 -0.2501  0.2143  0.6533 -0.3499 -0.1254  0.5221 -0.7402 -0.5802  0.0883
#>  -0.5047  0.5471  0.3680 -0.2191  0.1280  0.4291 -0.6376 -0.5321  0.0219
#>   0.0567 -0.0229  0.6422 -0.2965  0.2693  0.2768 -0.3239 -0.1265  0.3079
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9  0.6686 -0.7609 -0.0439  0.2973  0.5193 -0.1165 -0.7112 -0.6510 -0.4516
#>  -0.5786  0.5349  0.1044 -0.4652  0.3613 -0.6901 -0.6746 -0.3773 -0.3348
#>   0.0442 -0.7620  0.5406  0.6711 -0.3813  0.3076 -0.0884 -0.1134  0.0020
#> 
#> Columns 10 to 18  0.7558  0.4241 -0.8805  0.1688 -0.5699  0.7179  0.2211 -0.6245 -0.2719
#>   0.6349 -0.3974  0.3818  0.4010 -0.4605  0.0067  0.5170 -0.7112 -0.7727
#>  -0.4818  0.1193 -0.0315 -0.5995 -0.4883  0.6255 -0.4961 -0.0000  0.6660
#> 
#> Columns 19 to 20 -0.4818  0.6052
#>  -0.2051  0.0272
#>  -0.4504  0.3381
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.0584 -0.1267  0.4806 -0.2270  0.1915  0.5498 -0.4938 -0.1451  0.5909
#>  -0.4135  0.1365 -0.1300 -0.2621  0.0351 -0.2862  0.5519  0.0059  0.3768
#>   0.0797  0.1733  0.6057  0.3876 -0.0490  0.0879 -0.5439 -0.3819 -0.0906
#> 
#> Columns 10 to 18  0.4858  0.1553  0.2539 -0.1911  0.6499 -0.2508  0.1468 -0.2249 -0.5336
#>   0.0307 -0.3798 -0.0562 -0.3755 -0.0917 -0.2717  0.1045  0.4909 -0.5025
#>  -0.6471 -0.0786  0.1451  0.2186 -0.0771 -0.4271 -0.0108 -0.0855  0.3546
#> 
#> Columns 19 to 20  0.2095 -0.4698
#>  -0.1163 -0.3529
#>  -0.1539  0.0542
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
