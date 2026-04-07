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
#> Columns 1 to 9  0.0648  0.8049 -0.5068 -0.4507 -0.3178 -0.6341  0.4676  0.3981 -0.3785
#>   0.8769  0.0345 -0.8109 -0.2243 -0.8030  0.6362  0.2408 -0.3157 -0.1018
#>  -0.5209  0.1095 -0.6634 -0.4797 -0.5506  0.4888  0.4104  0.1821 -0.0747
#> 
#> Columns 10 to 18 -0.1631 -0.4317 -0.6793 -0.1205  0.0604 -0.2333  0.1213  0.0635 -0.1435
#>  -0.1097 -0.8836 -0.6113 -0.4594 -0.7464  0.7756  0.2633 -0.0996 -0.9058
#>  -0.8861 -0.2698 -0.4068 -0.4714 -0.2882 -0.7582  0.6918  0.8558 -0.9194
#> 
#> Columns 19 to 20  0.8053  0.8382
#>   0.5713  0.1988
#>  -0.4787 -0.0708
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.5869 -0.0118 -0.1859  0.3654  0.4692  0.0542  0.5414 -0.2190 -0.1800
#>   0.3098  0.1516 -0.2890  0.3818 -0.6235  0.4188  0.1811 -0.0983  0.2531
#>   0.1725 -0.1749 -0.3372  0.2545 -0.0087  0.0399 -0.1489 -0.4248  0.5973
#> 
#> Columns 10 to 18 -0.3339 -0.2290 -0.0985  0.6284 -0.5629 -0.3034  0.0531 -0.4794 -0.7210
#>   0.0842 -0.2459 -0.5117  0.7387 -0.0891  0.3926 -0.0958  0.1353 -0.7573
#>  -0.5301 -0.6502  0.2375 -0.3881 -0.6702 -0.7685  0.1234 -0.6685 -0.3048
#> 
#> Columns 19 to 20  0.4142  0.1448
#>   0.0778  0.0854
#>   0.1820  0.4791
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.1667  0.3943 -0.4534 -0.2407  0.3893  0.0410  0.6437 -0.3190 -0.4185
#>   0.0356  0.3591 -0.6506 -0.0979  0.1146 -0.0878  0.4355 -0.4080 -0.2839
#>  -0.2263 -0.3610 -0.1306 -0.2805  0.2311  0.1430  0.1133 -0.2450  0.2199
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.1018 -0.3101  0.2470  0.7596  0.4682  0.3427 -0.2189  0.1965 -0.1310
#>  -0.0731 -0.1475 -0.3076 -0.1201  0.0598  0.5164 -0.3094 -0.0502  0.5357
#>   0.1281  0.2868  0.3722 -0.2336 -0.1325 -0.1693  0.2271 -0.4917  0.0482
#> 
#> Columns 10 to 18  0.4627 -0.2734 -0.5075  0.0948 -0.2742 -0.5735 -0.8710  0.5414  0.3222
#>   0.4781  0.5149 -0.2533 -0.2510  0.6563 -0.0295 -0.2890 -0.0687  0.0281
#>   0.4865  0.3244 -0.6988 -0.1362  0.0929  0.1990 -0.3802 -0.4947 -0.1964
#> 
#> Columns 19 to 20  0.6730  0.5192
#>   0.7680 -0.6095
#>   0.4629 -0.1827
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.1521 -0.3132 -0.4977 -0.3142  0.3593 -0.3204  0.4564 -0.1295  0.1264
#>  -0.2068  0.0659 -0.1760  0.1141  0.6452 -0.3664 -0.0562 -0.3152  0.1849
#>   0.1841 -0.0253  0.1011  0.4245  0.3818 -0.0084  0.2017  0.0743  0.3232
#> 
#> Columns 10 to 18  0.0423 -0.6294  0.1987 -0.2621 -0.4710 -0.2676  0.0941 -0.2869 -0.0124
#>   0.0344 -0.5033  0.3998 -0.0686 -0.1531 -0.4093 -0.0009 -0.5847 -0.3018
#>  -0.0530 -0.6641  0.1532 -0.0441 -0.2323 -0.6416  0.4866 -0.1614 -0.4522
#> 
#> Columns 19 to 20  0.3869  0.1773
#>   0.2773 -0.2079
#>   0.2977  0.3198
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
