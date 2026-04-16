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
#> Columns 1 to 9  0.2351 -0.1568  0.8924 -0.1823 -0.8786 -0.3675  0.6199 -0.5705 -0.2191
#>   0.7794 -0.9839  0.9053  0.0945  0.5045  0.2437 -0.7301  0.7876 -0.7521
#>   0.5804 -0.1311 -0.3004  0.9343  0.4788  0.6104  0.2461 -0.5039  0.4772
#> 
#> Columns 10 to 18  0.7185  0.6789  0.1775  0.3643  0.3479 -0.2901  0.6829  0.6147  0.1258
#>   0.6038  0.4749 -0.6973  0.5274  0.0434  0.7066 -0.9338 -0.9777  0.6771
#>  -0.0086 -0.6918  0.2083  0.5409 -0.1857  0.3520  0.2608 -0.8197  0.3957
#> 
#> Columns 19 to 20  0.6267 -0.1568
#>   0.1912  0.7958
#>   0.9711 -0.3411
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.0240  0.2176  0.0486 -0.0801  0.0703 -0.0886  0.3387 -0.4136  0.6481
#>  -0.3278 -0.4790  0.3777  0.0655 -0.1506 -0.2532 -0.7203 -0.3356 -0.1211
#>  -0.4979  0.6547  0.3514 -0.0065 -0.6117 -0.2908 -0.2651 -0.3713 -0.5806
#> 
#> Columns 10 to 18  0.1195  0.4894 -0.4043  0.4383  0.0672  0.4650  0.0561 -0.1602 -0.3404
#>   0.1225 -0.2961 -0.0890  0.3945  0.0610  0.5863  0.0230 -0.3276 -0.2092
#>   0.0563  0.3346  0.5111  0.0922  0.7097  0.2922 -0.0473 -0.0931  0.2662
#> 
#> Columns 19 to 20  0.1528 -0.2461
#>   0.1894  0.0239
#>  -0.1813 -0.3547
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.3159 -0.1233 -0.4680  0.2117 -0.3363 -0.2684  0.4010 -0.2298  0.1284
#>   0.4649 -0.2260  0.0247  0.3983 -0.3273 -0.4147  0.4572 -0.5785  0.0669
#>   0.4348  0.0401  0.7006 -0.2027 -0.1848 -0.4144  0.3772 -0.6123  0.3511
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.0627 -0.1775  0.5394  0.4626 -0.4427 -0.5073  0.7691  0.0650 -0.7512
#>  -0.0701  0.4360  0.7070  0.3271 -0.3872 -0.0729  0.6620  0.4534 -0.2019
#>  -0.5411 -0.1480  0.1283 -0.2907  0.1043 -0.5960 -0.1072 -0.4940 -0.1732
#> 
#> Columns 10 to 18 -0.3954 -0.5330 -0.0140  0.0881  0.2361  0.7595  0.9143 -0.5564  0.7001
#>  -0.2135 -0.6812 -0.2270  0.7223 -0.4865 -0.0451 -0.0748 -0.5950 -0.2036
#>  -0.2105 -0.3972 -0.0236  0.0697  0.6490  0.1665 -0.0764 -0.6752  0.5245
#> 
#> Columns 19 to 20  0.4547 -0.1936
#>  -0.2767  0.2486
#>   0.8433 -0.0287
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2781  0.0927 -0.5110 -0.5750 -0.0796  0.3789  0.5292 -0.0218  0.2502
#>   0.0117 -0.0164 -0.4487  0.1870 -0.2213  0.0319  0.3568 -0.2532  0.2009
#>   0.0889  0.2756 -0.3986 -0.2949 -0.3856  0.1042  0.2639 -0.1850  0.2481
#> 
#> Columns 10 to 18 -0.0747  0.2181  0.0533 -0.1157  0.3560  0.5441  0.4971 -0.0495 -0.0528
#>  -0.2235 -0.0898 -0.4229  0.0897  0.0073  0.0999  0.3019 -0.1911 -0.1243
#>   0.5453 -0.1318 -0.0753 -0.1247 -0.1973  0.4349  0.3869  0.0166 -0.0692
#> 
#> Columns 19 to 20  0.0770 -0.5426
#>   0.6583 -0.2426
#>   0.1443 -0.6834
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
