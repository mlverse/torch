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
#> Columns 1 to 9 -0.9285 -0.0262  0.1378 -0.2114 -0.1693 -0.4232  0.5184 -0.8063  0.2803
#>   0.2771 -0.1011  0.4748 -0.4485 -0.2887  0.1070  0.4972 -0.3434  0.3747
#>   0.1875  0.2268 -0.6980 -0.2641 -0.6682 -0.8331 -0.7582 -0.6949 -0.2308
#> 
#> Columns 10 to 18  0.2017 -0.0303 -0.3914 -0.6104  0.0509  0.4799 -0.7017  0.7693 -0.2854
#>  -0.4258 -0.0260 -0.5592 -0.7603  0.0014  0.5609 -0.1230  0.8169 -0.6365
#>   0.7886 -0.1787  0.9827  0.4376  0.4101  0.7900  0.5275 -0.3830  0.6856
#> 
#> Columns 19 to 20  0.2636 -0.5336
#>  -0.5269  0.7784
#>  -0.6021 -0.0220
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.5445 -0.1932 -0.1239  0.0300  0.5582 -0.4961  0.5175 -0.7058 -0.1586
#>   0.0783  0.3874  0.0149 -0.3278 -0.3727 -0.5074  0.2111 -0.2398 -0.2725
#>  -0.6182 -0.0284 -0.0512  0.1937  0.1571  0.5265 -0.1646 -0.7067  0.7989
#> 
#> Columns 10 to 18 -0.1631  0.1756 -0.3509 -0.1848  0.5184  0.6286 -0.4209 -0.3751 -0.0957
#>  -0.3466  0.3728 -0.3985  0.2412  0.5542  0.3150 -0.2380  0.0684 -0.6391
#>   0.0671 -0.6875  0.7052  0.3524 -0.5806  0.6327  0.1272  0.6373 -0.3201
#> 
#> Columns 19 to 20 -0.7504 -0.3104
#>  -0.3961 -0.2188
#>  -0.6950  0.7375
#> 
#> (3,.,.) = 
#> Columns 1 to 9 -0.0856  0.0653 -0.0481  0.5414  0.2001 -0.4433  0.1945 -0.1744 -0.3499
#>  -0.3744 -0.3134 -0.3565 -0.0783 -0.3264 -0.4964  0.4853 -0.3181  0.1993
#>  -0.8821 -0.2967  0.3845 -0.4302 -0.1206 -0.0578  0.5618  0.0777  0.2288
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,3,20} ][ grad_fn = <StackBackward0> ]
#> 
#> [[2]]
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 9  0.5972  0.4727 -0.2917 -0.4413 -0.2198 -0.7594  0.2342  0.6338  0.0072
#>  -0.6548 -0.7841  0.7254  0.3190 -0.1956 -0.3053  0.7197 -0.7817 -0.1762
#>  -0.1342  0.0976  0.5339  0.1214 -0.2650 -0.4884  0.6410  0.1876 -0.4991
#> 
#> Columns 10 to 18 -0.2632 -0.4806  0.3578 -0.1882 -0.4021 -0.0647  0.3766  0.7422  0.0208
#>   0.4409  0.4816 -0.6773  0.0794 -0.0860  0.4702  0.8688  0.0891 -0.3768
#>   0.1584 -0.2230 -0.1352 -0.5964 -0.2422  0.1734  0.4064  0.5330 -0.3203
#> 
#> Columns 19 to 20  0.7755 -0.1456
#>   0.1420  0.0454
#>   0.4032  0.5687
#> 
#> (2,.,.) = 
#> Columns 1 to 9 -0.5352 -0.2987  0.1786 -0.2773 -0.1373  0.2963  0.5401  0.1191  0.0545
#>  -0.4876 -0.2484  0.0554 -0.1354  0.0528 -0.3952  0.6200 -0.1072  0.2014
#>  -0.6848 -0.1732  0.0671 -0.0940  0.0693  0.2056  0.2680 -0.2722  0.4838
#> 
#> Columns 10 to 18  0.0695 -0.6270  0.2303  0.0517 -0.0575  0.0774  0.0268  0.1299  0.0176
#>   0.0967 -0.1277 -0.5080  0.6467  0.5335  0.3864  0.2406 -0.3960 -0.3403
#>   0.0078 -0.2337 -0.6142 -0.0092  0.1151  0.2360  0.2728 -0.5551 -0.1936
#> 
#> Columns 19 to 20 -0.6250  0.3548
#>  -0.4956 -0.0135
#>  -0.5519  0.4463
#> [ CPUFloatType{2,3,20} ][ grad_fn = <StackBackward0> ]
#> 
```
