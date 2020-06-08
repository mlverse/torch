#' @include nn.R
NULL

nn_apply_permutation <- function(tensor, permutation, dim = 1) {
  tensor$index_select(dim, permutation)
}

rnn_impls_ <- list(
  RNN_RELU = torch_rnn_relu,
  RNN_TANH = torch_rnn_tanh
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
    
    if (bidirectional)
      num_directions <- 2
    else
      num_directions <- 1
    
    if (dropout > 0 && num_layers == 1) {
      warn("dropout option adds dropout after all but last ",
           "recurrent layer, so non-zero dropout expects ",
           "num_layers greater than 1, but got dropout={dropout} and ",
           "num_layers={num_layers}")
    }
    
    if (mode == "LSTM")
      gate_size <- 4 * hidden_size
    else if (mode == "GRU")
      gate_size <- 3 * hidden_size
    else if (mode == "RNN_TANH")
      gate_size <- hidden_size
    else if (mode == "RNN_RELU")
      gate_size <- hidden_size
    else
      value_error("Unrecognized RNN mode: {mode}")
    
    self$flat_weights_names_ <- list()
    self$all_weights_ <- list()
    
    for (layer in seq_len(num_layers)) {
      for (direction in seq_len(num_directions)) {
       
        if (layer == 1) 
          layer_input_size <- input_size
        else
          layer_input_size <- hidden_size * num_directions
        
        
        w_ih <- nn_parameter(torch_empty(gate_size, layer_input_size))
        w_hh <- nn_parameter(torch_empty(gate_size, hidden_size))
        b_ih <- nn_parameter(torch_empty(gate_size))
        
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        b_hh <- nn_parameter(torch_empty(gate_size))
        layer_params <- list(w_ih, w_hh, b_ih, b_hh)
        
        if (direction == 2)
          suffix <- "_reverse"
        else
          suffix <- ""
        
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
  flatten_parameters = function() {
    
  },
  reset_parameters = function() {
    stdv <- 1 / sqrt(self$hidden_size)
    for (weight in self$parameters) {
      nn_init_uniform_(weight, -stdv, stdv)
    }
  },
  permute_hidden = function(hx, permutation) {
    if (is.null(permutation))
      hx
    else
      nn_apply_permutation(hx, permutation)
  },
  forward = function(input, hx = NULL) {
    
    batch_sizes <- NULL
    if (self$batch_first)
      max_batch_size <- input$size(0)
    else
      max_batch_size <- input$size(1)
    
    sorted_indices <- NULL
    unsorted_indices <- NULL
    
    if (is.null(hx)) {
      num_directions <- ifelse(self$bidirectional, 2, 1)
      hx <- torch_zeros(self$num_layers * num_directions,
                        max_batch_size, self.hidden_size,
                        dtype=input$dtype(), device=input$device())
      
    } else {
      hx <- self$permute_hidden(hx, sorted_indices)  
    }
    
    impl_ <- rnn_impls_[[self$mode]]
    
    if (is.null(batch_sizes)) {
      result <- impl_(input = input, hx  = hx, 
                      params = self$flat_weights_, has_biases = self$bias,
                      num_layers = self$num_layers, dropout = self$dropout, 
                      train = self$training,
                      bidirectional = self$bidirectional,
                      batch_first = self$batch_first
                      )
    } else {
      result <- impl_(input = input, batch_sizes = batch_sizes, hx  = hx, 
                      params = self$flat_weights_, has_biases = self$bias,
                      num_layers = self$num_layers, dropout = self$dropout, 
                      train = self$training, bidirectional = self$bidirectional, 
                      batch_first = self$batch_first)
    }
        
    output <- result[[1]]
    hidden <- result[[2]]
    
    list(output, self$permute_hidden(hidden, unsorted_indices))
  }
)

#' RNN module
#' 
#' @examples
#' rnn <- nn_rnn(10, 20, 2)
#' input <- torch_randn(5, 3, 10)
#' h0 <- torch_randn(2, 3, 20)
#' rnn(input, h0)
#'
#' @export
nn_rnn <- nn_module(
  "nn_rnn",
  inherit = nn_rnn_base,
  initialize = function(...) {
    args <- list(...)
    
    self$nonlinearity <- args$nonlinearity
    args$nonlinearity <- NULL
    
    if (self$nonlinearity == "tanh" || is.null(args$nonlinearity))
      mode <- "RNN_TANH"
    else if (self$nonlinearity == "relu")
      mode <- "RNN_RELU"
    else
      value_error("Unknown nonlinearity '{self$nonlinearity}'")
    
    args$mode <- mode
    
    do.call(super$initialize, args)
  }
)