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
#> Columns 1 to 9 -0.5352 -0.3314 -0.4149  0.5846  0.9379  0.4522  0.2786  0.7146  0.0311
#>  -0.3043 -0.2387 -0.2394  0.4594 -0.8111 -0.0489  0.5635 -0.0965  0.4015
#>  -0.6321  0.0748  0.0429  0.8431 -0.2342  0.8703  0.6124  0.8581 -0.1647
#> 
#> Columns 10 to 18  0.3382 -0.5386  0.8480  0.0956 -0.2035 -0.4617  0.1769 -0.9330  0.8293
#>   0.2654  0.4205 -0.7255 -0.7141  0.2440 -0.8682 -0.4115  0.4892  0.5542
#>   0.5930 -0.0778  0.6818  0.6233 -0.0800 -0.7097 -0.4011  0.0788  0.6924
#> 
#> Columns 19 to 20  0.4199 -0.5524
#>  -0.1384  0.7930
#>  -0.2227  0.1695
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.3034  0.6410 -0.3574 -0.4159  0.0571  0.1019 -0.2292 -0.0963 -0.2592
#>  -0.8442 -0.3316 -0.3140 -0.0319  0.2593 -0.0535 -0.1864  0.3329  0.3609
#>  -0.3071  0.0450 -0.6291  0.0168  0.5795  0.1590  0.0691  0.2383  0.1089
#> 
#> Columns 10 to 18 -0.1073  0.3733 -0.3860 -0.2241 -0.6203 -0.3481  0.5158  0.6809 -0.6760
#>  -0.3330  0.7161  0.4950  0.5468 -0.4374 -0.2258  0.3755 -0.5114  0.0426
#>  -0.2648  0.0153  0.3308  0.3395 -0.6899 -0.6749  0.4995 -0.1146  0.0547
#> 
#> Columns 19 to 20 -0.0439 -0.0614
#>  -0.3403  0.4348
#>  -0.6005 -0.0102
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.5646 -0.6357  0.1182 -0.0833 -0.5852  0.2265  0.1529 -0.0412 -0.0550
#>  -0.3864  0.5115 -0.4471  0.2214 -0.1575 -0.3434  0.4227  0.0379  0.1428
#>  -0.3311  0.2339 -0.3699 -0.4179  0.0795 -0.2566  0.4835  0.0378  0.1709
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.4316  0.3723  0.5120  0.2776  0.5681  0.2315 -0.5752 -0.0199 -0.2475
#>  -0.0387 -0.1352 -0.1502 -0.1844 -0.7217  0.1616 -0.5591 -0.1379 -0.0319
#>   0.6028  0.4412  0.5941  0.1286 -0.5992 -0.3510 -0.7076  0.0828  0.7030
#> 
#> Columns 10 to 18 -0.0842  0.1726  0.0466 -0.1259  0.5784  0.0225 -0.5371 -0.2727  0.6115
#>   0.2642 -0.6353  0.0799  0.2567  0.0260 -0.0496 -0.2862  0.6133  0.6020
#>   0.0545 -0.7227  0.4947  0.2887  0.7312 -0.5028 -0.2340 -0.4619  0.8392
#> 
#> Columns 19 to 20 -0.2332  0.1851
#>  -0.1706 -0.3712
#>  -0.1942  0.4972
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.1861 -0.1513 -0.3763  0.1990 -0.0214  0.4877  0.1547  0.4519 -0.3221
#>  -0.4270  0.0417 -0.2668 -0.0691 -0.5914  0.1146  0.1427  0.0628 -0.0637
#>  -0.3097 -0.6454  0.1808  0.4159 -0.3688  0.4745  0.0740  0.4474 -0.0013
#> 
#> Columns 10 to 18 -0.0724 -0.3046 -0.1810 -0.0906 -0.4431 -0.6521  0.5705  0.2715 -0.0063
#>  -0.3994  0.0601 -0.3752 -0.3262 -0.0441 -0.3009  0.2876  0.4626  0.1753
#>   0.2111 -0.5474 -0.2808 -0.1101  0.2203 -0.5143  0.3737  0.1612  0.2968
#> 
#> Columns 19 to 20 -0.1048  0.0499
#>   0.0889  0.1439
#>   0.2607  0.5198
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
