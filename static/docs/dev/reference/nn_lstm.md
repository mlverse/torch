# Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

For each element in the input sequence, each layer computes the
following function:

## Usage

``` r
nn_lstm(
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
  stacking two LSTMs together to form a `stacked LSTM`, with the second
  LSTM taking in outputs of the first LSTM and computing the final
  results. Default: 1

- bias:

  If `FALSE`, then the layer does not use bias weights `b_ih` and
  `b_hh`. Default: `TRUE`

- batch_first:

  If `TRUE`, then the input and output tensors are provided as (batch,
  seq, feature). Default: `FALSE`

- dropout:

  If non-zero, introduces a `Dropout` layer on the outputs of each LSTM
  layer except the last layer, with dropout probability equal to
  `dropout`. Default: 0

- bidirectional:

  If `TRUE`, becomes a bidirectional LSTM. Default: `FALSE`

- ...:

  currently unused.

## Details

\$\$ \begin{array}{ll} \\ i_t = \sigma(W\_{ii} x_t + b\_{ii} + W\_{hi}
h\_{(t-1)} + b\_{hi}) \\ f_t = \sigma(W\_{if} x_t + b\_{if} + W\_{hf}
h\_{(t-1)} + b\_{hf}) \\ g_t = \tanh(W\_{ig} x_t + b\_{ig} + W\_{hg}
h\_{(t-1)} + b\_{hg}) \\ o_t = \sigma(W\_{io} x_t + b\_{io} + W\_{ho}
h\_{(t-1)} + b\_{ho}) \\ c_t = f_t c\_{(t-1)} + i_t g_t \\ h_t = o_t
\tanh(c_t) \\ \end{array} \$\$

where \\h_t\\ is the hidden state at time `t`, \\c_t\\ is the cell state
at time `t`, \\x_t\\ is the input at time `t`, \\h\_{(t-1)}\\ is the
hidden state of the previous layer at time `t-1` or the initial hidden
state at time `0`, and \\i_t\\, \\f_t\\, \\g_t\\, \\o_t\\ are the input,
forget, cell, and output gates, respectively. \\\sigma\\ is the sigmoid
function.

## Note

All the weights and biases are initialized from \\\mathcal{U}(-\sqrt{k},
\sqrt{k})\\ where \\k = \frac{1}{\mbox{hidden\\size}}\\

## Inputs

Inputs: input, (h_0, c_0)

- **input** of shape `(seq_len, batch, input_size)`: tensor containing
  the features of the input sequence. The input can also be a packed
  variable length sequence. See
  [`nn_utils_rnn_pack_padded_sequence()`](https://torch.mlverse.org/docs/dev/reference/nn_utils_rnn_pack_padded_sequence.md)
  or
  [`nn_utils_rnn_pack_sequence()`](https://torch.mlverse.org/docs/dev/reference/nn_utils_rnn_pack_sequence.md)
  for details.

- **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`:
  tensor containing the initial hidden state for each element in the
  batch.

- **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`:
  tensor containing the initial cell state for each element in the
  batch.

If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to
zero.

## Outputs

Outputs: output, (h_n, c_n)

- **output** of shape `(seq_len, batch, num_directions * hidden_size)`:
  tensor containing the output features `(h_t)` from the last layer of
  the LSTM, for each t. If a `torch_nn.utils.rnn.PackedSequence` has
  been given as the input, the output will also be a packed sequence.
  For the unpacked case, the directions can be separated using
  `output$view(c(seq_len, batch, num_directions, hidden_size))`, with
  forward and backward being direction `0` and `1` respectively.
  Similarly, the directions can be separated in the packed case.

- **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`:
  tensor containing the hidden state for `t = seq_len`. Like *output*,
  the layers can be separated using
  `h_n$view(c(num_layers, num_directions, batch, hidden_size))` and
  similarly for *c_n*.

- **c_n** (num_layers \* num_directions, batch, hidden_size): tensor
  containing the cell state for `t = seq_len`

## Attributes

- `weight_ih_l[k]` : the learnable input-hidden weights of the
  \\\mbox{k}^{th}\\ layer `(W_ii|W_if|W_ig|W_io)`, of shape
  `(4*hidden_size x input_size)`

- `weight_hh_l[k]` : the learnable hidden-hidden weights of the
  \\\mbox{k}^{th}\\ layer `(W_hi|W_hf|W_hg|W_ho)`, of shape
  `(4*hidden_size x hidden_size)`

- `bias_ih_l[k]` : the learnable input-hidden bias of the
  \\\mbox{k}^{th}\\ layer `(b_ii|b_if|b_ig|b_io)`, of shape
  `(4*hidden_size)`

- `bias_hh_l[k]` : the learnable hidden-hidden bias of the
  \\\mbox{k}^{th}\\ layer `(b_hi|b_hf|b_hg|b_ho)`, of shape
  `(4*hidden_size)`

## Examples

``` r
if (torch_is_installed()) {
rnn <- nn_lstm(10, 20, 2)
input <- torch_randn(5, 3, 10)
h0 <- torch_randn(2, 3, 20)
c0 <- torch_randn(2, 3, 20)
output <- rnn(input, list(h0, c0))
}
```
