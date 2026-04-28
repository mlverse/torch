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
#> Columns 1 to 9 -0.8476  0.4115  0.2513 -0.6438  0.5592 -0.5318 -0.6618  0.4586 -0.4675
#>  -0.3209 -0.2483 -0.1577 -0.2579  0.0053  0.5986  0.6924 -0.1729 -0.6738
#>  -0.6468  0.0180  0.5657 -0.6046  0.6248 -0.9138 -0.6930  0.6150 -0.2824
#> 
#> Columns 10 to 18  0.2926 -0.6202 -0.1154  0.3844 -0.3588  0.0045  0.0221  0.6370  0.6067
#>  -0.7502 -0.1328  0.3491 -0.3825 -0.7597 -0.6569 -0.0271 -0.1175  0.0269
#>  -0.8083 -0.3631 -0.7010  0.2975  0.8919  0.8661  0.2407  0.6370  0.4529
#> 
#> Columns 19 to 20 -0.7491 -0.3607
#>  -0.0163 -0.5270
#>  -0.7537 -0.4495
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.1147  0.1432 -0.3715 -0.2027  0.4123 -0.1111  0.7461 -0.2196  0.1280
#>  -0.0272 -0.0354  0.4653 -0.1791 -0.0699  0.1484 -0.1128  0.2717 -0.0477
#>  -0.1396  0.8214 -0.0268 -0.2368  0.3647  0.0459  0.6670 -0.4271  0.2178
#> 
#> Columns 10 to 18 -0.7025 -0.0997  0.4239  0.4138  0.1809  0.1798  0.5686  0.4098 -0.5339
#>  -0.5640 -0.0761  0.3550 -0.0069 -0.3445  0.2611  0.1989 -0.0541 -0.1007
#>  -0.3236 -0.0370  0.4810 -0.0298  0.1993  0.3551  0.3946  0.4486  0.2391
#> 
#> Columns 19 to 20 -0.2634 -0.5910
#>  -0.4012 -0.2327
#>  -0.4067 -0.2275
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.2279  0.1958  0.0847 -0.6814  0.0865 -0.2849  0.0538  0.2767 -0.5583
#>  -0.2337  0.1077  0.1834 -0.1550  0.2051 -0.0011  0.2096 -0.2643 -0.3434
#>  -0.3860  0.4895  0.5996 -0.5834  0.0573  0.4018 -0.2528 -0.3145 -0.2389
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.3507 -0.2646 -0.3904  0.2377  0.4367  0.0065  0.4967 -0.2070  0.0475
#>   0.3286 -0.4935 -0.1450  0.1471  0.3940  0.1866  0.1273 -0.6692 -0.5823
#>   0.3948 -0.2338 -0.5940 -0.7796  0.8108 -0.1107  0.1127  0.1725  0.1222
#> 
#> Columns 10 to 18 -0.3822 -0.5559  0.3343 -0.4751 -0.4854  0.0242  0.5179 -0.2863 -0.0291
#>  -0.6446 -0.4245  0.3067  0.3435 -0.4124  0.0209  0.8421 -0.2769  0.2187
#>  -0.2594  0.2168 -0.5468  0.3331 -0.2213  0.0726 -0.6439 -0.1250  0.4247
#> 
#> Columns 19 to 20 -0.1885  0.3161
#>   0.1393  0.1977
#>  -0.5080 -0.5162
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.1279  0.4619  0.0886 -0.0987  0.3316  0.1159 -0.1551 -0.3298 -0.1985
#>  -0.3244  0.2742  0.1401 -0.4428  0.1042 -0.1907  0.1265 -0.1189 -0.3784
#>  -0.5238 -0.2127  0.2202  0.6034  0.0503  0.3091  0.5493  0.0055 -0.1505
#> 
#> Columns 10 to 18 -0.5314 -0.5977  0.3802 -0.1914  0.3591  0.0194  0.6327  0.4566 -0.0971
#>  -0.5487 -0.5789  0.6008  0.2359  0.2486  0.0989  0.6479  0.3580 -0.3558
#>  -0.6220 -0.4878 -0.0455 -0.0677 -0.5418  0.0548  0.3823 -0.2283  0.2659
#> 
#> Columns 19 to 20 -0.3969 -0.2092
#>  -0.0346 -0.5525
#>   0.2150 -0.4823
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
