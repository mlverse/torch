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
#> Columns 1 to 9 -0.4092 -0.3584  0.0730 -0.0533  0.4060  0.8720  0.7812 -0.0811 -0.1944
#>   0.4774 -0.2560  0.7081 -0.5837  0.7605 -0.1136  0.7339 -0.7891 -0.8308
#>  -0.5925 -0.7227  0.3297 -0.9593  0.9361 -0.1308 -0.2634  0.2171 -0.5568
#> 
#> Columns 10 to 18  0.5558  0.5676 -0.3930 -0.3339 -0.2242 -0.8270 -0.4976  0.2180 -0.0802
#>  -0.0579 -0.2832 -0.8113 -0.0014  0.2592  0.1616 -0.4491 -0.6968 -0.8036
#>  -0.7957 -0.3542  0.7955  0.5220  0.3573  0.2803  0.1376 -0.3905 -0.5378
#> 
#> Columns 19 to 20  0.0105 -0.4112
#>   0.0660  0.5464
#>   0.3352  0.1774
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.4819 -0.2188 -0.0840  0.2744 -0.2758  0.0617  0.2772 -0.0370 -0.0972
#>  -0.7506 -0.3110  0.0753  0.0161 -0.1311  0.7750  0.2693 -0.5667 -0.8005
#>  -0.5001  0.4042  0.3970  0.3678 -0.8288  0.7721 -0.2645  0.4008 -0.4544
#> 
#> Columns 10 to 18  0.4845 -0.3163  0.2550 -0.5988 -0.5564 -0.7222 -0.2032  0.4632 -0.1281
#>   0.6371 -0.3296  0.0022  0.2914 -0.4402 -0.2681 -0.1019 -0.0559 -0.2725
#>   0.5936 -0.4424  0.4886 -0.3412 -0.2070 -0.2435  0.7220  0.0327  0.0504
#> 
#> Columns 19 to 20 -0.3548 -0.0143
#>   0.0533 -0.3154
#>  -0.2865  0.0311
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.6957  0.3103  0.4017  0.4345 -0.2458 -0.1079  0.3117  0.0300  0.0224
#>  -0.3591  0.1316  0.2694  0.2204 -0.2068  0.0769  0.2427 -0.3279 -0.0032
#>  -0.4720 -0.1357  0.1576 -0.1176  0.2064 -0.3097  0.6349  0.1495  0.2125
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.0934  0.3470  0.4547 -0.0475  0.1513 -0.0639  0.1528  0.3510  0.6830
#>   0.1202  0.5619  0.3321 -0.0293  0.3296 -0.3122 -0.1368 -0.3968  0.2843
#>   0.5396  0.7246  0.8422  0.1962  0.2829 -0.4896 -0.3032  0.3669  0.4648
#> 
#> Columns 10 to 18 -0.4560  0.6305 -0.1406  0.6554  0.5750  0.3994  0.5332  0.0751  0.0360
#>  -0.5089 -0.0212  0.5667 -0.6810  0.2450 -0.2364  0.6258 -0.5854 -0.0710
#>   0.1045 -0.0634  0.6005 -0.6853 -0.2348 -0.4808 -0.2587 -0.4493 -0.1382
#> 
#> Columns 19 to 20  0.6303  0.3952
#>  -0.0360  0.4766
#>  -0.2669  0.1678
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.2554  0.4092  0.3987  0.6425 -0.3340 -0.2600  0.3363 -0.0810 -0.0153
#>  -0.5328 -0.3139 -0.0001  0.0954 -0.3074 -0.0982  0.6966  0.2615 -0.0306
#>  -0.6992  0.1469 -0.0511  0.2172 -0.4540  0.0176  0.4741 -0.0753  0.2242
#> 
#> Columns 10 to 18 -0.1872 -0.2119  0.1547 -0.2506 -0.4384 -0.1986  0.6046  0.6493 -0.1876
#>   0.2027 -0.4412  0.1696 -0.0788 -0.1899 -0.6888 -0.0916  0.1776  0.0903
#>  -0.3406 -0.2445  0.5735 -0.0293  0.0677 -0.1414  0.3675  0.3082 -0.3328
#> 
#> Columns 19 to 20  0.1487  0.4663
#>  -0.5647  0.2380
#>  -0.3561  0.1476
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
