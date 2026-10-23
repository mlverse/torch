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
#> Columns 1 to 9  0.0720 -0.3395 -0.2776 -0.0050  0.7412  0.8558 -0.7829 -0.6055 -0.4498
#>  -0.8839 -0.6728  0.3289  0.5586 -0.2335  0.9193 -0.5343  0.1800 -0.8326
#>  -0.5487  0.7846  0.1516  0.0377  0.4252  0.3287 -0.4865  0.0346  0.5172
#> 
#> Columns 10 to 18 -0.1388  0.2986  0.1581 -0.8166 -0.2872  0.6846 -0.4068 -0.1038 -0.3220
#>   0.8945 -0.1151 -0.6631  0.1051 -0.3293  0.5983  0.3355 -0.4403 -0.3006
#>  -0.5869 -0.3531  0.0367 -0.4394  0.4112  0.8741 -0.4951 -0.0362  0.6806
#> 
#> Columns 19 to 20  0.3484 -0.0567
#>  -0.1107  0.3410
#>  -0.0941 -0.3124
#> 
#> (2,.,.) = 
#> Columns 1 to 9  0.2505 -0.1301 -0.2097  0.0501 -0.4253  0.6685 -0.3363 -0.1493 -0.1593
#>  -0.0189 -0.1328 -0.7547  0.5610 -0.4073  0.5557 -0.4880  0.3712 -0.0028
#>  -0.3041 -0.1330  0.3692  0.1348  0.3639  0.6133 -0.6363  0.0819  0.0681
#> 
#> Columns 10 to 18 -0.6615 -0.0288 -0.5298 -0.2296 -0.0980  0.4456 -0.4484 -0.4157  0.2591
#>  -0.1008 -0.2854  0.0137 -0.4599 -0.2719  0.4895 -0.7352 -0.0293  0.1886
#>   0.2210  0.2726  0.2403 -0.4342  0.2354  0.5549  0.3396 -0.0530  0.1211
#> 
#> Columns 19 to 20  0.5905  0.0824
#>  -0.0394 -0.2430
#>   0.0158 -0.4985
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.2370  0.2795 -0.3802  0.1192 -0.0066  0.6644 -0.5905  0.2916 -0.1516
#>  -0.1551  0.0300 -0.5444 -0.2045  0.1753  0.5750 -0.6970  0.0121 -0.0431
#>   0.1142 -0.0468 -0.0982 -0.2874 -0.2951  0.1535 -0.3467  0.0185  0.0383
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9 -0.1053  0.6001  0.2641  0.0009 -0.3023 -0.6301  0.3729 -0.6152 -0.8906
#>  -0.3418 -0.2913  0.5180  0.4827 -0.3345  0.0240 -0.3078  0.5513  0.3907
#>   0.6787 -0.4566  0.0051  0.4788 -0.0686 -0.5085 -0.0846  0.0012 -0.3850
#> 
#> Columns 10 to 18 -0.4460  0.3680 -0.4170 -0.1184 -0.6439  0.2151 -0.0927  0.1482 -0.3563
#>  -0.2015  0.8209 -0.0937  0.5276 -0.2710 -0.5994 -0.3000  0.8159 -0.0616
#>  -0.7262  0.7756  0.2259  0.4307  0.6512 -0.2687 -0.5653 -0.2470 -0.1299
#> 
#> Columns 19 to 20 -0.8065  0.1528
#>  -0.6448  0.0893
#>  -0.7183  0.1415
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.5110  0.0452 -0.2167  0.1619  0.3787  0.1317 -0.6210  0.1812 -0.3091
#>  -0.3071 -0.0208  0.0778 -0.3895 -0.4550  0.4516 -0.5173  0.1574  0.0243
#>  -0.1879 -0.0285 -0.2953 -0.4366  0.0202  0.2423 -0.4693 -0.2329  0.3096
#> 
#> Columns 10 to 18 -0.1553  0.2384  0.0605 -0.0844 -0.1849  0.1943  0.2258 -0.1314 -0.5175
#>  -0.5680 -0.0767 -0.5465  0.0400  0.3508  0.2456 -0.5044 -0.2764 -0.5120
#>  -0.4046 -0.2224  0.2744 -0.0790 -0.3852  0.3083 -0.4524  0.0218 -0.4108
#> 
#> Columns 19 to 20 -0.1942 -0.4659
#>   0.0741  0.2782
#>   0.2072 -0.5152
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
