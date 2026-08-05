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
#> Columns 1 to 9 -0.0666 -0.7613  0.5128 -0.2330  0.1848  0.5186 -0.6982  0.3679 -0.3466
#>  -0.4825 -0.4787 -0.2460 -0.7604  0.6448  0.3741 -0.0291  0.2056 -0.4136
#>  -0.1317  0.3357  0.4480 -0.5631  0.7912  0.7334 -0.2137 -0.1759 -0.7018
#> 
#> Columns 10 to 18 -0.2293  0.4505 -0.7473  0.3817 -0.6308 -0.6251  0.5437 -0.4579  0.5769
#>  -0.0941  0.8979 -0.8850  0.8671 -0.3602 -0.0876  0.1714  0.7670 -0.4554
#>  -0.6076  0.6095 -0.3281  0.8168 -0.2257 -0.8679 -0.6655 -0.0133  0.1606
#> 
#> Columns 19 to 20  0.3615  0.6253
#>  -0.6170 -0.4632
#>  -0.0164 -0.6321
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.3361 -0.2929 -0.5680  0.2137 -0.7000 -0.4175  0.0957 -0.0624  0.1456
#>   0.2524 -0.0130 -0.3556  0.4501  0.1951  0.3627 -0.2436 -0.2083 -0.5011
#>  -0.0363  0.1953 -0.5730  0.1938 -0.3421  0.2767 -0.1672 -0.1469  0.0340
#> 
#> Columns 10 to 18  0.0159 -0.1560 -0.2018 -0.5527 -0.7107 -0.1726  0.1502  0.4967  0.5247
#>  -0.3035 -0.2031 -0.6336 -0.2395 -0.3125 -0.1601 -0.3958  0.0944 -0.3338
#>  -0.2348  0.2022 -0.5302  0.1117 -0.1192  0.4893 -0.4702  0.0545  0.4216
#> 
#> Columns 19 to 20 -0.0394  0.2712
#>  -0.4441  0.2831
#>   0.0911  0.3698
#> 
#> (3,.,.) = 
#> Columns 1 to 9  0.7144 -0.1391 -0.0904  0.3153  0.0527  0.2134 -0.4428 -0.1096  0.6765
#>  -0.2737  0.4631 -0.5530 -0.0692  0.0195  0.5749  0.4158 -0.3356  0.2698
#>   0.6130 -0.2292 -0.2343  0.4528 -0.0310  0.4009 -0.4498 -0.0197  0.4788
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.0830 -0.1735 -0.1922 -0.5199  0.0420 -0.2236  0.2867 -0.0770 -0.4180
#>   0.1913  0.7084 -0.1551 -0.6816 -0.3410  0.2450  0.0745 -0.2257  0.0458
#>   0.5318  0.7013  0.6527  0.4304 -0.3820 -0.5673  0.5359 -0.4126  0.1083
#> 
#> Columns 10 to 18  0.5032  0.2083 -0.3843 -0.2090  0.2719 -0.4865  0.5514 -0.6009  0.1341
#>   0.2086 -0.1602 -0.3122 -0.0652  0.4770  0.7627  0.0453 -0.3647  0.5334
#>   0.2924 -0.2322  0.0485 -0.0900  0.5895  0.0894  0.4762 -0.3619 -0.2329
#> 
#> Columns 19 to 20 -0.3695  0.2497
#>  -0.2134  0.5223
#>   0.3388  0.1294
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2067 -0.1040 -0.4066  0.1581  0.0933  0.2402  0.0882 -0.0559  0.2695
#>  -0.0636  0.2880 -0.6109  0.3895 -0.6372  0.3469  0.2648 -0.1767  0.0379
#>   0.3562 -0.2499 -0.2966  0.1141 -0.4769  0.6220 -0.5936  0.0796  0.2697
#> 
#> Columns 10 to 18 -0.3036  0.0617  0.0898  0.2004  0.1838 -0.0128 -0.4661  0.2531 -0.1584
#>  -0.2075  0.2014 -0.3393  0.1411  0.0617 -0.1411  0.4115  0.6036  0.4519
#>  -0.0087  0.4247 -0.3518  0.2128 -0.0942 -0.0720 -0.0399  0.2686  0.0797
#> 
#> Columns 19 to 20 -0.1945 -0.0372
#>   0.3778 -0.1427
#>  -0.0105 -0.0827
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
