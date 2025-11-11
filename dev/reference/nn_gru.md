# Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

For each element in the input sequence, each layer computes the
following function:

## Usage

``` r
nn_gru(
  input_size,
  hidden_size,
  num_layers = 1,
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
  stacking two GRUs together to form a `stacked GRU`, with the second
  GRU taking in outputs of the first GRU and computing the final
  results. Default: 1

- bias:

  If `FALSE`, then the layer does not use bias weights `b_ih` and
  `b_hh`. Default: `TRUE`

- batch_first:

  If `TRUE`, then the input and output tensors are provided as (batch,
  seq, feature). Default: `FALSE`

- dropout:

  If non-zero, introduces a `Dropout` layer on the outputs of each GRU
  layer except the last layer, with dropout probability equal to
  `dropout`. Default: 0

- bidirectional:

  If `TRUE`, becomes a bidirectional GRU. Default: `FALSE`

- ...:

  currently unused.

## Details

\$\$ \begin{array}{ll} r_t = \sigma(W\_{ir} x_t + b\_{ir} + W\_{hr}
h\_{(t-1)} + b\_{hr}) \\ z_t = \sigma(W\_{iz} x_t + b\_{iz} + W\_{hz}
h\_{(t-1)} + b\_{hz}) \\ n_t = \tanh(W\_{in} x_t + b\_{in} + r_t
(W\_{hn} h\_{(t-1)}+ b\_{hn})) \\ h_t = (1 - z_t) n_t + z_t h\_{(t-1)}
\end{array} \$\$

where \\h_t\\ is the hidden state at time `t`, \\x_t\\ is the input at
time `t`, \\h\_{(t-1)}\\ is the hidden state of the previous layer at
time `t-1` or the initial hidden state at time `0`, and \\r_t\\,
\\z_t\\, \\n_t\\ are the reset, update, and new gates, respectively.
\\\sigma\\ is the sigmoid function.

## Note

All the weights and biases are initialized from \\\mathcal{U}(-\sqrt{k},
\sqrt{k})\\ where \\k = \frac{1}{\mbox{hidden\\size}}\\

## Inputs

Inputs: input, h_0

- **input** of shape `(seq_len, batch, input_size)`: tensor containing
  the features of the input sequence. The input can also be a packed
  variable length sequence. See
  [`nn_utils_rnn_pack_padded_sequence()`](https://torch.mlverse.org/docs/dev/reference/nn_utils_rnn_pack_padded_sequence.md)
  for details.

- **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`:
  tensor containing the initial hidden state for each element in the
  batch. Defaults to zero if not provided.

## Outputs

Outputs: output, h_n

- **output** of shape `(seq_len, batch, num_directions * hidden_size)`:
  tensor containing the output features h_t from the last layer of the
  GRU, for each t. If a `PackedSequence` has been given as the input,
  the output will also be a packed sequence. For the unpacked case, the
  directions can be separated using
  `output$view(c(seq_len, batch, num_directions, hidden_size))`, with
  forward and backward being direction `0` and `1` respectively.
  Similarly, the directions can be separated in the packed case.

- **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`:
  tensor containing the hidden state for `t = seq_len` Like *output*,
  the layers can be separated using
  `h_n$view(num_layers, num_directions, batch, hidden_size)`.

## Attributes

- `weight_ih_l[k]` : the learnable input-hidden weights of the
  \\\mbox{k}^{th}\\ layer (W_ir\|W_iz\|W_in), of shape
  `(3*hidden_size x input_size)`

- `weight_hh_l[k]` : the learnable hidden-hidden weights of the
  \\\mbox{k}^{th}\\ layer (W_hr\|W_hz\|W_hn), of shape
  `(3*hidden_size x hidden_size)`

- `bias_ih_l[k]` : the learnable input-hidden bias of the
  \\\mbox{k}^{th}\\ layer (b_ir\|b_iz\|b_in), of shape `(3*hidden_size)`

- `bias_hh_l[k]` : the learnable hidden-hidden bias of the
  \\\mbox{k}^{th}\\ layer (b_hr\|b_hz\|b_hn), of shape `(3*hidden_size)`

## Examples

``` r
if (torch_is_installed()) {

rnn <- nn_gru(10, 20, 2)
input <- torch_randn(5, 3, 10)
h0 <- torch_randn(2, 3, 20)
output <- rnn(input, h0)
}
```
