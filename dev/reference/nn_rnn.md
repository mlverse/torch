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
#> Columns 1 to 9  0.3561 -0.8618 -0.0348 -0.3873 -0.6217  0.8975  0.4562 -0.0116 -0.4995
#>  -0.7862 -0.7561  0.9303 -0.3360  0.7966 -0.1878 -0.3227 -0.2709  0.2720
#>   0.2678 -0.5839 -0.8439 -0.4052  0.8095 -0.3091  0.1700  0.7317  0.3240
#> 
#> Columns 10 to 18 -0.2340  0.5823  0.3878  0.8237  0.2945  0.7870  0.0963  0.3002  0.7864
#>   0.0488 -0.1928  0.8497 -0.5455  0.3516 -0.2588  0.2784  0.4395  0.2105
#>   0.8334 -0.6863  0.8175  0.2879  0.8172 -0.5513 -0.0667 -0.5242  0.4477
#> 
#> Columns 19 to 20  0.6456 -0.7351
#>   0.1828  0.2476
#>   0.5234 -0.6567
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.5510 -0.2252 -0.4862 -0.3758  0.8345 -0.5083 -0.1134  0.0687  0.1979
#>  -0.3048  0.5550  0.5681  0.4787  0.4585 -0.0305 -0.6219 -0.3220  0.1457
#>  -0.0757 -0.4067  0.0310  0.0138  0.3097  0.0836  0.2928 -0.2499  0.1812
#> 
#> Columns 10 to 18  0.2060 -0.4234  0.0488  0.7848  0.3857 -0.6874 -0.0344  0.3812 -0.0141
#>  -0.2740  0.5696  0.7848  0.4903  0.4037  0.6152  0.4386 -0.1796  0.1234
#>  -0.1469  0.3081 -0.4153  0.5724  0.2710 -0.0629  0.1817  0.2509  0.2469
#> 
#> Columns 19 to 20 -0.0320 -0.4623
#>   0.5635 -0.0665
#>  -0.1044 -0.6041
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.0624 -0.3118  0.3913  0.3075  0.2516 -0.1045 -0.4820 -0.4790  0.3643
#>  -0.6862  0.1883  0.2425 -0.3786  0.8622 -0.4180 -0.6181 -0.2603  0.5368
#>  -0.5399 -0.4458 -0.0879  0.1567  0.5844 -0.3896 -0.7701 -0.2906  0.0414
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.8745 -0.2281  0.4285  0.0921  0.3920  0.1820 -0.3632 -0.3739 -0.2685
#>  -0.0728 -0.3209  0.2151  0.2962  0.0249 -0.0689 -0.0885  0.2896  0.4296
#>   0.4196  0.6836 -0.2605 -0.4522 -0.5252 -0.0960 -0.0456 -0.3794 -0.2980
#> 
#> Columns 10 to 18 -0.0172 -0.4920 -0.5409  0.6231 -0.5803 -0.5945  0.0276  0.2922  0.6764
#>  -0.5863 -0.3136  0.0880 -0.0462 -0.0164 -0.0103 -0.1950  0.0811  0.5019
#>  -0.5749 -0.0651 -0.0769 -0.6868  0.1224  0.1987 -0.4740  0.2365  0.8673
#> 
#> Columns 19 to 20  0.4668  0.7866
#>   0.1417  0.4683
#>   0.1177 -0.1797
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2647  0.4164  0.3282  0.3088 -0.3074  0.3320 -0.4600 -0.2597 -0.3959
#>  -0.4633 -0.1524  0.0935 -0.3391  0.2001 -0.4000 -0.2188  0.0076  0.0882
#>  -0.3778  0.3452  0.5208 -0.0449  0.5942  0.0666 -0.5776 -0.5305  0.4801
#> 
#> Columns 10 to 18 -0.0780  0.0390  0.1823  0.4655  0.1559  0.3164 -0.0411 -0.1693  0.2721
#>  -0.1537 -0.0195  0.3371  0.5208  0.3966 -0.0556 -0.0679 -0.2449  0.1014
#>   0.0331  0.2690  0.7243  0.7230 -0.0543  0.2105  0.1387 -0.1445  0.4098
#> 
#> Columns 19 to 20  0.6355  0.2567
#>   0.4250  0.3140
#>   0.5232  0.6234
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
