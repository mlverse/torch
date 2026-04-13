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
#> Columns 1 to 9 -0.7073  0.4918 -0.5826  0.3236 -0.2160  0.6665  0.1464  0.5633 -0.8270
#>   0.1861  0.5546 -0.8259 -0.7609  0.8336  0.1131 -0.4344  0.5943  0.4693
#>  -0.8832  0.8930 -0.5854 -0.2961 -0.5455  0.3875 -0.0776 -0.7862 -0.2782
#> 
#> Columns 10 to 18  0.2930 -0.2514 -0.6231 -0.3676  0.2651  0.3028 -0.2067 -0.8146 -0.2170
#>  -0.1415  0.1369 -0.5399 -0.8136  0.1087 -0.8787 -0.4384 -0.0778 -0.0351
#>  -0.7632 -0.7928  0.0669 -0.0668  0.2589 -0.5928  0.5546 -0.5408  0.1948
#> 
#> Columns 19 to 20 -0.2395 -0.5020
#>  -0.5019  0.0130
#>   0.6526 -0.1599
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.5113  0.3288 -0.6932 -0.2761  0.1512  0.1415  0.3104 -0.7623  0.4342
#>  -0.2249  0.4918 -0.0089 -0.5171  0.2923 -0.6659  0.3477 -0.2291  0.4720
#>   0.3802 -0.3470 -0.3373 -0.2430  0.1889 -0.0424  0.0284 -0.4639  0.1363
#> 
#> Columns 10 to 18 -0.2815  0.1599 -0.1775 -0.5239  0.1004  0.0646 -0.1152 -0.5612 -0.2693
#>  -0.3566  0.2552 -0.2041  0.2643  0.1077 -0.1560 -0.3902  0.5347 -0.4085
#>  -0.2382 -0.0678 -0.4498 -0.4131  0.4000  0.0678  0.3599 -0.6352  0.0756
#> 
#> Columns 19 to 20  0.4263  0.3836
#>  -0.3936  0.0812
#>  -0.2224  0.6247
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.1213  0.3368 -0.1460 -0.7129  0.4655 -0.4502 -0.0696 -0.1749  0.3798
#>  -0.5316  0.5223 -0.6170  0.3055 -0.3187 -0.0792 -0.4573 -0.2135  0.4139
#>   0.2252  0.1347 -0.2371 -0.0748  0.4197 -0.2585  0.0959 -0.0608 -0.1154
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.0916 -0.5161 -0.4149 -0.7244  0.0493  0.3722  0.1626 -0.2323 -0.4111
#>   0.0066  0.0192  0.4596  0.2804 -0.7647  0.3718 -0.1830 -0.1809 -0.1679
#>   0.5139 -0.1996  0.0424 -0.2592 -0.6428  0.5366 -0.1048  0.0309 -0.0975
#> 
#> Columns 10 to 18 -0.3551  0.2989 -0.3880  0.3074  0.1596 -0.4713  0.3174  0.5664 -0.0555
#>   0.5189 -0.7397 -0.0950  0.6629  0.3512 -0.4983 -0.5887 -0.0419 -0.6953
#>   0.5491 -0.7455  0.4847  0.1068  0.0888 -0.1483 -0.0843  0.0287 -0.3534
#> 
#> Columns 19 to 20 -0.3207  0.3606
#>  -0.5126  0.4878
#>  -0.1915 -0.2755
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.0075  0.2761 -0.2758 -0.3395  0.4348  0.0146  0.1063  0.2910 -0.1920
#>   0.1109  0.4362 -0.7018  0.0181  0.1940  0.3112  0.0065 -0.3993  0.1209
#>   0.1264  0.6690 -0.4615 -0.1471  0.0208 -0.1097 -0.0863 -0.3087  0.2569
#> 
#> Columns 10 to 18  0.1181 -0.0197 -0.2740 -0.2477  0.3424  0.0015  0.0952 -0.1674 -0.5474
#>   0.0701 -0.2587 -0.1377 -0.3092  0.7617  0.1919 -0.1252 -0.4366 -0.4225
#>  -0.2260  0.2796 -0.3659 -0.0857  0.7143 -0.3569 -0.1806 -0.4658 -0.4855
#> 
#> Columns 19 to 20 -0.2337 -0.1381
#>  -0.0501  0.3658
#>   0.1032  0.2965
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
