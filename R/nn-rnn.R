#' @include nn.R
NULL

nn_apply_permutation <- function(tensor, permutation, dim = 2) {
  tensor$index_select(dim, permutation)
}

rnn_impls_ <- list(
  RNN_RELU = torch_rnn_relu,
  RNN_TANH = torch_rnn_tanh,
  LSTM = torch_lstm,
  GRU = torch_gru
)

nn_rnn_base <- nn_module(
  "nn_rnn_base",
  initialize = function(mode, input_size, hidden_size, num_layers = 1, bias = TRUE,
                        batch_first = FALSE, dropout = 0., bidirectional = FALSE) {
    self$mode <- mode
    self$input_size <- input_size
    self$hidden_size <- hidden_size
    self$num_layers <- num_layers
    self$bias <- bias
    self$batch_first <- batch_first
    self$dropout <- dropout
    self$bidirectional <- bidirectional
    self$proj_size <- 0 # TODO: add support for proj_size.

    if (bidirectional) {
      num_directions <- 2
    } else {
      num_directions <- 1
    }

    if (dropout > 0 && num_layers == 1) {
      warn(
        "dropout option adds dropout after all but last ",
        "recurrent layer, so non-zero dropout expects ",
        "num_layers greater than 1, but got dropout={dropout} and ",
        "num_layers={num_layers}"
      )
    }

    if (mode == "LSTM") {
      gate_size <- 4 * hidden_size
    } else if (mode == "GRU") {
      gate_size <- 3 * hidden_size
    } else if (mode == "RNN_TANH") {
      gate_size <- hidden_size
    } else if (mode == "RNN_RELU") {
      gate_size <- hidden_size
    } else {
      value_error("Unrecognized RNN mode: {mode}")
    }

    self$flat_weights_names_ <- list()
    self$all_weights_ <- list()

    for (layer in seq_len(num_layers)) {
      for (direction in seq_len(num_directions)) {
        if (layer == 1) {
          layer_input_size <- input_size
        } else {
          layer_input_size <- hidden_size * num_directions
        }


        w_ih <- nn_parameter(torch_empty(gate_size, layer_input_size))
        w_hh <- nn_parameter(torch_empty(gate_size, hidden_size))
        b_ih <- nn_parameter(torch_empty(gate_size))

        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        b_hh <- nn_parameter(torch_empty(gate_size))
        layer_params <- list(w_ih, w_hh, b_ih, b_hh)

        if (direction == 2) {
          suffix <- "_reverse"
        } else {
          suffix <- ""
        }

        param_names <- c(
          glue::glue("weight_ih_l{layer}{suffix}"),
          glue::glue("weight_hh_l{layer}{suffix}")
        )
        if (bias) {
          param_names <- c(param_names, c(
            glue::glue("bias_ih_l{layer}{suffix}"),
            glue::glue("bias_hh_l{layer}{suffix}")
          ))
        }

        for (i in seq_along(param_names)) {
          self[[param_names[i]]] <- layer_params[[i]]
        }

        self$flat_weight_names_ <- c(self$flat_weight_names_, param_names)
        self$all_weights_ <- c(self$all_weights_, param_names)
      }
    }

    self$flat_weights_ <- lapply(
      self$flat_weight_names_,
      function(wn) {
        self[[wn]]
      }
    )

    self$flatten_parameters()
    self$reset_parameters()
  },
  .apply = function(fn) {
    ret <- super$.apply(fn)
    # Resets _flat_weights
    # Note: be v. careful before removing this, as 3rd party device types
    # likely rely on this behavior to properly .to() modules like LSTM.
    self$flat_weights_ <- lapply(
      self$flat_weight_names_,
      function(wn) {
        self[[wn]]
      }
    )
    # Flattens params (on CUDA)
    self$flatten_parameters()
    ret
  },
  flatten_parameters = function() {
    # Short-circuits if _flat_weights is only partially instantiated
    if (length(self$flat_weights_) != length(self$flat_weight_names_)) {
      return()
    }

    for (w in self$flat_weights_) {
      if (!is_torch_tensor(w)) {
        return()
      }
    }

    # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
    # or the tensors in _flat_weights are of different dtypes
    first_fw <- self$flat_weights_[[1]]
    dtype <- first_fw$dtype

    for (fw in self$flat_weights_) {
      if (!is_torch_tensor(fw) || !(fw$dtype == dtype) || !fw$is_cuda ||
        !torch_cudnn_is_acceptable(fw)) {
        return()
      }
    }

    # If any parameters alias, we fall back to the slower, copying code path. This is
    # a sufficient check, because overlapping parameter buffers that don't completely
    # alias would break the assumptions of the uniqueness check in
    # Module.named_parameters().
    unique_data_ptrs <- unique(sapply(self$flat_weights_, function(x) x$storage()$data_ptr()))
    if (length(unique_data_ptrs) != length(self$flat_weights_)) {
      return()
    }

    with_no_grad({
      if (cpp_torch_namespace__use_cudnn_rnn_flatten_weight()) {
        num_weights <- if (self$bias) 4 else 2
        torch__cudnn_rnn_flatten_weight(
          weight_arr = self$flat_weights_, weight_stride0 = num_weights,
          input_size = self$input_size, mode = rnn.get_cudnn_mode(self$mode),
          hidden_size = self$hidden_size, num_layers = self$num_layers,
          batch_first = self$batch_first, bidirectional = as.logical(self$bidirectional),
          proj_size = self$proj_size
        )
      }
    })
  },
  reset_parameters = function() {
    stdv <- 1 / sqrt(self$hidden_size)
    for (weight in self$parameters) {
      nn_init_uniform_(weight, -stdv, stdv)
    }
  },
  permute_hidden = function(hx, permutation) {
    if (is.null(permutation)) {
      hx
    } else {
      nn_apply_permutation(hx, permutation)
    }
  },
  forward = function(input, hx = NULL) {
    is_packed <- is_packed_sequence(input)

    if (is_packed) {
      batch_sizes <- input$batch_sizes
      sorted_indices <- input$sorted_indices
      unsorted_indices <- input$unsorted_indices
      max_batch_size <- as_array(batch_sizes[1]$to(dtype = torch_int()))
      input <- input$data
    } else {
      batch_sizes <- NULL
      if (self$batch_first) {
        max_batch_size <- input$size(1)
      } else {
        max_batch_size <- input$size(2)
      }

      sorted_indices <- NULL
      unsorted_indices <- NULL
    }



    if (is.null(hx)) {
      num_directions <- ifelse(self$bidirectional, 2, 1)
      hx <- torch_zeros(self$num_layers * num_directions,
        max_batch_size, self$hidden_size,
        dtype = input$dtype, device = input$device
      )

      if (self$mode == "LSTM") {
        hx <- list(hx, hx)
      }
    } else {
      hx <- self$permute_hidden(hx, sorted_indices)
    }

    impl_ <- rnn_impls_[[self$mode]]

    if (is.null(batch_sizes)) {
      result <- impl_(
        input = input, hx = hx,
        params = self$flat_weights_, has_biases = self$bias,
        num_layers = self$num_layers, dropout = self$dropout,
        train = self$training,
        bidirectional = self$bidirectional,
        batch_first = self$batch_first
      )
    } else {
      result <- impl_(
        data = input, hx = hx, batch_sizes = batch_sizes,
        params = self$flat_weights_, has_biases = self$bias,
        num_layers = self$num_layers, dropout = self$dropout,
        train = self$training,
        bidirectional = self$bidirectional
      )
    }

    output <- result[[1]]
    hidden <- result[-1]

    if (length(hidden) == 1) {
      hidden <- hidden[[1]]
    }

    if (is_packed) {
      output <- new_packed_sequence(
        output, batch_sizes, sorted_indices,
        unsorted_indices
      )
    }

    list(output, self$permute_hidden(hidden, unsorted_indices))
  }
)

rnn.get_cudnn_mode <- function(mode) {
  if (mode == "RNN_RELU") {
    0L
  } else if (mode == "RNN_TANH") {
    1L
  } else if (mode == "LSTM") {
    2L
  } else if (mode == "GRU") {
    3L
  } else {
    not_implemented_error("No cudnn backend for mode '{mode}'")
  }
}

#' RNN module
#'
#' Applies a multi-layer Elman RNN with \eqn{\tanh} or \eqn{\mbox{ReLU}} non-linearity
#' to an input sequence.
#'
#' For each element in the input sequence, each layer computes the following
#' function:
#'
#' \deqn{
#' h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
#' }
#'
#' where \eqn{h_t} is the hidden state at time `t`, \eqn{x_t} is
#' the input at time `t`, and \eqn{h_{(t-1)}} is the hidden state of the
#' previous layer at time `t-1` or the initial hidden state at time `0`.
#' If `nonlinearity` is `'relu'`, then \eqn{\mbox{ReLU}} is used instead of
#' \eqn{\tanh}.
#'
#' @param input_size The number of expected features in the input `x`
#' @param hidden_size The number of features in the hidden state `h`
#' @param num_layers Number of recurrent layers. E.g., setting `num_layers=2`
#'   would mean stacking two RNNs together to form a `stacked RNN`,
#'   with the second RNN taking in outputs of the first RNN and
#'   computing the final results. Default: 1
#' @param nonlinearity The non-linearity to use. Can be either `'tanh'` or
#'   `'relu'`. Default: `'tanh'`
#' @param bias If `FALSE`, then the layer does not use bias weights `b_ih` and
#'   `b_hh`. Default: `TRUE`
#' @param batch_first If `TRUE`, then the input and output tensors are provided
#'   as `(batch, seq, feature)`. Default: `FALSE`
#' @param dropout If non-zero, introduces a `Dropout` layer on the outputs of each
#'   RNN layer except the last layer, with dropout probability equal to
#'   `dropout`. Default: 0
#' @param bidirectional If `TRUE`, becomes a bidirectional RNN. Default: `FALSE`
#' @param ... other arguments that can be passed to the super class.
#'
#' @section Inputs:
#'
#' - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
#' of the input sequence. The input can also be a packed variable length
#' sequence.
#' - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#' containing the initial hidden state for each element in the batch.
#' Defaults to zero if not provided. If the RNN is bidirectional,
#' num_directions should be 2, else it should be 1.
#'
#'
#'
#' @section Outputs:
#'
#' - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
#' containing the output features (`h_t`) from the last layer of the RNN,
#' for each `t`.  If a :class:`nn_packed_sequence` has
#' been given as the input, the output will also be a packed sequence.
#' For the unpacked case, the directions can be separated
#' using `output$view(seq_len, batch, num_directions, hidden_size)`,
#' with forward and backward being direction `0` and `1` respectively.
#' Similarly, the directions can be separated in the packed case.
#'
#' - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#' containing the hidden state for `t = seq_len`.
#' Like *output*, the layers can be separated using
#' `h_n$view(num_layers, num_directions, batch, hidden_size)`.
#'
#' @section Shape:
#'
#' - Input1: \eqn{(L, N, H_{in})} tensor containing input features where
#'  \eqn{H_{in}=\mbox{input\_size}} and `L` represents a sequence length.
#' - Input2: \eqn{(S, N, H_{out})} tensor
#'   containing the initial hidden state for each element in the batch.
#'   \eqn{H_{out}=\mbox{hidden\_size}}
#'   Defaults to zero if not provided. where \eqn{S=\mbox{num\_layers} * \mbox{num\_directions}}
#'   If the RNN is bidirectional, num_directions should be 2, else it should be 1.
#' - Output1: \eqn{(L, N, H_{all})} where \eqn{H_{all}=\mbox{num\_directions} * \mbox{hidden\_size}}
#' - Output2: \eqn{(S, N, H_{out})} tensor containing the next hidden state
#'   for each element in the batch
#'
#' @section Attributes:
#' - `weight_ih_l[k]`: the learnable input-hidden weights of the k-th layer,
#'   of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
#'   `(hidden_size, num_directions * hidden_size)`
#' - `weight_hh_l[k]`: the learnable hidden-hidden weights of the k-th layer,
#'   of shape `(hidden_size, hidden_size)`
#' - `bias_ih_l[k]`: the learnable input-hidden bias of the k-th layer,
#'   of shape `(hidden_size)`
#' - `bias_hh_l[k]`: the learnable hidden-hidden bias of the k-th layer,
#'   of shape `(hidden_size)`
#'
#' @section Note:
#'
#' All the weights and biases are initialized from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})}
#' where \eqn{k = \frac{1}{\mbox{hidden\_size}}}
#'
#' @examples
#' rnn <- nn_rnn(10, 20, 2)
#' input <- torch_randn(5, 3, 10)
#' h0 <- torch_randn(2, 3, 20)
#' rnn(input, h0)
#' @export
nn_rnn <- nn_module(
  "nn_rnn",
  inherit = nn_rnn_base,
  initialize = function(input_size, hidden_size, num_layers = 1, nonlinearity = NULL,
                        bias = TRUE, batch_first = FALSE, dropout = 0.,
                        bidirectional = FALSE, ...) {
    args <- list(...)

    if (is.null(nonlinearity)) {
      self$nonlinearity <- "tanh"
    } else {
      self$nonlinearity <- nonlinearity
    }

    if (self$nonlinearity == "tanh") {
      mode <- "RNN_TANH"
    } else if (self$nonlinearity == "relu") {
      mode <- "RNN_RELU"
    } else {
      value_error("Unknown nonlinearity '{self$nonlinearity}'")
    }

    super$initialize(mode,
      input_size = input_size, hidden_size = hidden_size,
      num_layers = num_layers, bias = bias,
      batch_first = batch_first, dropout = dropout,
      bidirectional = bidirectional, ...
    )
  }
)

#' Applies a multi-layer long short-term memory (LSTM) RNN to an input
#' sequence.
#'
#' For each element in the input sequence, each layer computes the following
#' function:
#'
#' \deqn{
#' \begin{array}{ll} \\
#' i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
#' f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
#' g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
#' o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
#' c_t = f_t c_{(t-1)} + i_t g_t \\
#' h_t = o_t \tanh(c_t) \\
#' \end{array}
#' }
#'
#' where \eqn{h_t} is the hidden state at time `t`, \eqn{c_t} is the cell
#' state at time `t`, \eqn{x_t} is the input at time `t`, \eqn{h_{(t-1)}}
#' is the hidden state of the previous layer at time `t-1` or the initial hidden
#' state at time `0`, and \eqn{i_t}, \eqn{f_t}, \eqn{g_t},
#' \eqn{o_t} are the input, forget, cell, and output gates, respectively.
#' \eqn{\sigma} is the sigmoid function.
#'
#' @param input_size The number of expected features in the input `x`
#' @param hidden_size The number of features in the hidden state `h`
#' @param num_layers Number of recurrent layers. E.g., setting `num_layers=2`
#'   would mean stacking two LSTMs together to form a `stacked LSTM`,
#'   with the second LSTM taking in outputs of the first LSTM and
#'   computing the final results. Default: 1
#' @param bias If `FALSE`, then the layer does not use bias weights `b_ih` and `b_hh`.
#'   Default: `TRUE`
#' @param batch_first If `TRUE`, then the input and output tensors are provided
#'   as (batch, seq, feature). Default: `FALSE`
#' @param dropout If non-zero, introduces a `Dropout` layer on the outputs of each
#'   LSTM layer except the last layer, with dropout probability equal to
#'   `dropout`. Default: 0
#' @param bidirectional If `TRUE`, becomes a bidirectional LSTM. Default: `FALSE`
#' @param ... currently unused.
#'
#' @section Inputs:
#'
#' Inputs: input, (h_0, c_0)
#'
#' - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
#'   of the input sequence.
#'   The input can also be a packed variable length sequence.
#'   See [nn_utils_rnn_pack_padded_sequence()] or
#'   [nn_utils_rnn_pack_sequence()] for details.
#'
#' - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#'   containing the initial hidden state for each element in the batch.
#' - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#'   containing the initial cell state for each element in the batch.
#'
#' If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
#'
#' @section Outputs:
#'
#' Outputs: output, (h_n, c_n)
#'
#' - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
#'   containing the output features `(h_t)` from the last layer of the LSTM,
#'   for each t. If a `torch_nn.utils.rnn.PackedSequence` has been
#'   given as the input, the output will also be a packed sequence.
#'   For the unpacked case, the directions can be separated
#'   using `output$view(c(seq_len, batch, num_directions, hidden_size))`,
#'   with forward and backward being direction `0` and `1` respectively.
#'   Similarly, the directions can be separated in the packed case.
#' - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#'   containing the hidden state for `t = seq_len`.
#'   Like *output*, the layers can be separated using
#'   `h_n$view(c(num_layers, num_directions, batch, hidden_size))` and similarly for *c_n*.
#' - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
#'   containing the cell state for `t = seq_len`
#'
#' @section Attributes:
#'
#' * `weight_ih_l[k]` : the learnable input-hidden weights of the \eqn{\mbox{k}^{th}} layer
#'   `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size x input_size)`
#' * `weight_hh_l[k]` : the learnable hidden-hidden weights of the \eqn{\mbox{k}^{th}} layer
#'   `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size x hidden_size)`
#' * `bias_ih_l[k]` : the learnable input-hidden bias of the \eqn{\mbox{k}^{th}} layer
#'   `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
#' * `bias_hh_l[k]` : the learnable hidden-hidden bias of the \eqn{\mbox{k}^{th}} layer
#'   `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
#'
#' @note
#' All the weights and biases are initialized from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})}
#' where \eqn{k = \frac{1}{\mbox{hidden\_size}}}
#'
#' @examples
#' rnn <- nn_lstm(10, 20, 2)
#' input <- torch_randn(5, 3, 10)
#' h0 <- torch_randn(2, 3, 20)
#' c0 <- torch_randn(2, 3, 20)
#' output <- rnn(input, list(h0, c0))
#' @export
nn_lstm <- nn_module(
  "nn_lstm",
  inherit = nn_rnn_base,
  initialize = function(input_size, hidden_size, num_layers = 1,
                        bias = TRUE, batch_first = FALSE, dropout = 0.,
                        bidirectional = FALSE, ...) {
    super$initialize(
      "LSTM",
      input_size = input_size, hidden_size = hidden_size,
      num_layers = num_layers, bias = bias,
      batch_first = batch_first, dropout = dropout,
      bidirectional = bidirectional, ...
    )
  },
  permute_hidden = function(hx, permutation) {
    if (is.null(permutation)) {
      hx
    } else {
      lapply(hx, nn_apply_permutation, permutation)
    }
  }
)

#' Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
#'
#' For each element in the input sequence, each layer computes the following
#' function:
#'
#' \deqn{
#' \begin{array}{ll}
#' r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
#' z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
#' n_t = \tanh(W_{in} x_t + b_{in} + r_t (W_{hn} h_{(t-1)}+ b_{hn})) \\
#' h_t = (1 - z_t) n_t + z_t h_{(t-1)}
#' \end{array}
#' }
#'
#' where \eqn{h_t} is the hidden state at time `t`, \eqn{x_t} is the input
#' at time `t`, \eqn{h_{(t-1)}} is the hidden state of the previous layer
#' at time `t-1` or the initial hidden state at time `0`, and \eqn{r_t},
#' \eqn{z_t}, \eqn{n_t} are the reset, update, and new gates, respectively.
#' \eqn{\sigma} is the sigmoid function.
#'
#'
#' @param input_size The number of expected features in the input `x`
#' @param hidden_size The number of features in the hidden state `h`
#' @param num_layers Number of recurrent layers. E.g., setting `num_layers=2`
#'   would mean stacking two GRUs together to form a `stacked GRU`,
#'   with the second GRU taking in outputs of the first GRU and
#'   computing the final results. Default: 1
#' @param bias If `FALSE`, then the layer does not use bias weights `b_ih` and `b_hh`.
#'   Default: `TRUE`
#' @param batch_first If `TRUE`, then the input and output tensors are provided
#'   as (batch, seq, feature). Default: `FALSE`
#' @param dropout If non-zero, introduces a `Dropout` layer on the outputs of each
#'   GRU layer except the last layer, with dropout probability equal to
#'   `dropout`. Default: 0
#' @param bidirectional If `TRUE`, becomes a bidirectional GRU. Default: `FALSE`
#' @param ... currently unused.
#'
#' @section Inputs:
#'
#' Inputs: input, h_0
#'
#' - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
#'   of the input sequence. The input can also be a packed variable length
#'   sequence. See [nn_utils_rnn_pack_padded_sequence()]
#'   for details.
#' - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#'   containing the initial hidden state for each element in the batch.
#'   Defaults to zero if not provided.
#'
#' @section Outputs:
#'
#' Outputs: output, h_n
#'
#' - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
#'   containing the output features h_t from the last layer of the GRU,
#'   for each t. If a `PackedSequence` has been
#'   given as the input, the output will also be a packed sequence.
#'   For the unpacked case, the directions can be separated
#'   using `output$view(c(seq_len, batch, num_directions, hidden_size))`,
#'   with forward and backward being direction `0` and `1` respectively.
#'   Similarly, the directions can be separated in the packed case.
#' - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#'   containing the hidden state for `t = seq_len`
#'   Like *output*, the layers can be separated using
#'   `h_n$view(num_layers, num_directions, batch, hidden_size)`.
#'
#' @section Attributes:
#' - `weight_ih_l[k]` : the learnable input-hidden weights of the \eqn{\mbox{k}^{th}} layer
#'   (W_ir|W_iz|W_in), of shape `(3*hidden_size x input_size)`
#' - `weight_hh_l[k]` : the learnable hidden-hidden weights of the \eqn{\mbox{k}^{th}} layer
#'   (W_hr|W_hz|W_hn), of shape `(3*hidden_size x hidden_size)`
#' - `bias_ih_l[k]` : the learnable input-hidden bias of the \eqn{\mbox{k}^{th}} layer
#'   (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
#' - `bias_hh_l[k]` : the learnable hidden-hidden bias of the \eqn{\mbox{k}^{th}} layer
#'   (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`
#'
#' @note
#'
#' All the weights and biases are initialized from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})}
#' where \eqn{k = \frac{1}{\mbox{hidden\_size}}}
#'
#' @examples
#'
#' rnn <- nn_gru(10, 20, 2)
#' input <- torch_randn(5, 3, 10)
#' h0 <- torch_randn(2, 3, 20)
#' output <- rnn(input, h0)
#' @export
nn_gru <- nn_module(
  "nn_gru",
  inherit = nn_rnn_base,
  initialize = function(input_size, hidden_size, num_layers = 1,
                        bias = TRUE, batch_first = FALSE, dropout = 0.,
                        bidirectional = FALSE, ...) {
    super$initialize(
      "GRU",
      input_size = input_size, hidden_size = hidden_size,
      num_layers = num_layers, bias = bias,
      batch_first = batch_first, dropout = dropout,
      bidirectional = bidirectional, ...
    )
  }
)
