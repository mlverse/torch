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
#> Columns 1 to 9  0.2438  0.7193  0.0140  0.3342  0.0138  0.7824  0.5767  0.0686  0.5405
#>  -0.8752 -0.4928 -0.6687  0.6277 -0.2059  0.6293  0.2370 -0.1428  0.0338
#>  -0.7933 -0.5885  0.2580 -0.0723  0.0026 -0.7159  0.4532 -0.6916  0.0446
#> 
#> Columns 10 to 18  0.5467  0.6300 -0.1006 -0.4220  0.1907  0.1373  0.8078 -0.7966  0.2831
#>   0.0354  0.6388 -0.7436  0.0545  0.7659 -0.4876  0.8348 -0.6680 -0.5505
#>   0.3629  0.2942 -0.0433  0.1705 -0.4070  0.5253 -0.9319 -0.5534  0.6799
#> 
#> Columns 19 to 20  0.6072  0.3338
#>   0.9022  0.5650
#>   0.5635  0.8595
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.0150 -0.3703  0.0053  0.4453 -0.4463  0.1192 -0.3548  0.0985  0.1569
#>  -0.0513 -0.3846  0.3728  0.6306 -0.8179  0.1247  0.2430 -0.2062 -0.3927
#>   0.4579  0.3926 -0.3575  0.0426 -0.2187 -0.5144 -0.3470  0.4673 -0.6622
#> 
#> Columns 10 to 18 -0.2925 -0.1666 -0.0960 -0.2756  0.0478  0.1409  0.5809 -0.3167  0.1343
#>  -0.1996  0.1946  0.1924 -0.5258  0.0902  0.2444  0.1821 -0.2311 -0.1393
#>   0.4765  0.3449  0.3020 -0.1097  0.1848 -0.3088  0.3700  0.4597  0.1603
#> 
#> Columns 19 to 20  0.3440  0.4909
#>   0.3900  0.6590
#>   0.4306  0.7676
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.0808 -0.2572 -0.1184  0.2320 -0.5005 -0.0035 -0.1186 -0.2326 -0.2705
#>   0.0167 -0.5634 -0.2365  0.0708 -0.5404 -0.0096 -0.2567 -0.3429 -0.3923
#>  -0.3952 -0.6003 -0.0198  0.4296  0.0094  0.5757 -0.1001 -0.1938  0.4644
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.2942  0.1504  0.1714  0.2111 -0.4366 -0.2257  0.4051 -0.7733  0.0219
#>   0.2192 -0.1078  0.3573 -0.7436 -0.0826 -0.0336 -0.6824  0.3627  0.4728
#>  -0.0957  0.1727  0.0971 -0.1940 -0.4950  0.2422  0.3867  0.7398  0.3281
#> 
#> Columns 10 to 18 -0.6470 -0.0138  0.1419  0.2057  0.1844  0.2289  0.6600  0.0176  0.5376
#>   0.4506  0.1156  0.3627 -0.3358  0.1252 -0.6370 -0.2724 -0.1424  0.5413
#>   0.4241  0.0838  0.1400 -0.2094  0.3022 -0.1411  0.5118 -0.0642  0.6014
#> 
#> Columns 19 to 20 -0.0565  0.0722
#>   0.6073 -0.4180
#>  -0.6827  0.4748
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2027 -0.0298 -0.3578  0.3539 -0.4284  0.5992  0.3104  0.2611  0.4115
#>   0.1294  0.3346  0.5891  0.2127 -0.3593 -0.2914  0.0598 -0.0363  0.1181
#>  -0.1216 -0.4544 -0.0362  0.5467 -0.4400  0.0557 -0.1648  0.1244  0.3613
#> 
#> Columns 10 to 18 -0.4444 -0.3258  0.3687  0.0638  0.4245  0.1582  0.2123  0.1991  0.1010
#>   0.4021  0.3975 -0.3108 -0.0716  0.5227  0.2025  0.4606  0.1518  0.0757
#>  -0.0993 -0.1098 -0.0088 -0.0066  0.0715  0.3529  0.3165 -0.0515 -0.0437
#> 
#> Columns 19 to 20  0.3864  0.4167
#>  -0.0739  0.7302
#>   0.4982  0.6484
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
