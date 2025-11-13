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
#>  Columns 1 to 9  0.8992 -0.7479  0.5449 -0.1769  0.1839 -0.3725  0.3061 -0.0411 -0.1171
#>   0.4635 -0.5718 -0.7097 -0.3329 -0.3324 -0.8544  0.2768 -0.4314  0.8506
#>   0.0861 -0.0540 -0.4192 -0.3945 -0.3145 -0.5920  0.7369 -0.6699  0.3084
#> 
#> Columns 10 to 18  0.1796 -0.8419 -0.0386 -0.1415  0.3218 -0.0557  0.3791 -0.6309  0.5690
#>  -0.1753 -0.6442 -0.2949 -0.5357  0.0793 -0.6684  0.8989 -0.0451  0.3404
#>  -0.1365 -0.6792 -0.8059 -0.1779  0.7624 -0.7613  0.6307  0.1553  0.5234
#> 
#> Columns 19 to 20  0.2672  0.1679
#>  -0.7069 -0.1944
#>  -0.2233 -0.2936
#> 
#> (2,.,.) = 
#>  Columns 1 to 9  0.3114 -0.2711  0.2147 -0.3964  0.1839 -0.2305 -0.4907  0.3789  0.2747
#>   0.0853 -0.0941  0.2015 -0.4957  0.2647  0.1158 -0.2899  0.4579  0.0928
#>   0.2283 -0.3651  0.0712 -0.7809  0.2160 -0.2467 -0.1940  0.2540 -0.2174
#> 
#> Columns 10 to 18 -0.3048 -0.0115  0.0325  0.2491  0.5141  0.0665  0.6233 -0.2167  0.3495
#>   0.2806 -0.5029 -0.1472  0.5051  0.2586  0.5045  0.3606  0.1119  0.7947
#>   0.3233 -0.2247 -0.1837  0.3616  0.3112  0.3989 -0.1113 -0.0479  0.8825
#> 
#> Columns 19 to 20 -0.2581 -0.2073
#>   0.2143 -0.3234
#>  -0.0244 -0.3230
#> 
#> (3,.,.) = 
#>  Columns 1 to 9  0.0666 -0.4968 -0.6725 -0.4780  0.3324 -0.2915 -0.4431  0.7409  0.0281
#>   0.2620 -0.3876 -0.0812 -0.5438 -0.0290 -0.2195 -0.1768  0.0997  0.0973
#>   0.2294 -0.3137  0.2217 -0.1263 -0.2020  0.2131  0.1222  0.0332  0.3068
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 9  0.4429 -0.2170 -0.6161 -0.2286  0.7504 -0.2073  0.1675  0.4533  0.5716
#>  -0.2553 -0.3537  0.6410  0.5224  0.5864  0.4317  0.5189  0.3510  0.0042
#>   0.1058 -0.3037 -0.5433 -0.4363 -0.1062  0.7780  0.3380  0.0701 -0.0197
#> 
#> Columns 10 to 18 -0.6178 -0.4900  0.6259 -0.2598 -0.0406  0.2538  0.7263  0.1468  0.4714
#>  -0.7843  0.4969 -0.2746 -0.1554  0.8656 -0.2606 -0.0631  0.1564  0.1305
#>  -0.0588 -0.4475 -0.1547 -0.0015 -0.5550  0.1292  0.5570  0.3857  0.4541
#> 
#> Columns 19 to 20  0.3015 -0.7245
#>   0.0775  0.4947
#>  -0.2705 -0.0607
#> 
#> (2,.,.) = 
#>  Columns 1 to 9 -0.0280  0.2526 -0.3190  0.1592  0.2199 -0.3755 -0.2914  0.5784  0.4079
#>  -0.1180 -0.5578 -0.3757 -0.6876  0.4885 -0.4709 -0.5656  0.3006  0.3691
#>   0.1484  0.0933 -0.2673  0.1407 -0.2055 -0.1146  0.0891  0.5051  0.2367
#> 
#> Columns 10 to 18 -0.1911 -0.6239 -0.0222  0.4351  0.4280  0.2804  0.0591  0.1502  0.3536
#>   0.4558  0.3076  0.0565  0.3862  0.6175  0.1238  0.3024  0.5155  0.5897
#>  -0.1354 -0.4195 -0.4402  0.1582  0.6234 -0.2951  0.0448 -0.0478  0.5607
#> 
#> Columns 19 to 20  0.0806 -0.0814
#>  -0.0135 -0.0902
#>  -0.4803 -0.0556
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
