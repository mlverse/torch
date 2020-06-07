#' @include nn.R
NULL

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
        b_ih <- nn_parameter(toorch_empty(gate_size))
        
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        b_hh <- nn_parameter(torch_empty(gate_size))
        layer_params <- list(w_ih, w_hh, b_ih, b_hh)
        
        if (direction == 2)
          suffix <- "_reverse"
        else
          suffix <- ""
        
        param_names <- glue::glue(c(
          "weight_ih_l{layer}{suffix}",
          "weight_hh_l{layer}{suffix}"
          ))
        if (bias) {
          param_names <- c(param_names, glue::glue(c(
            "bias_ih_l{layer}{suffix}",
            "bias_hh_l{layer}{suffix}"
          )))
        }
        
        for (i in seq_along(param_names)) {
          self[[param_names[i]]] <- layer_params[[i]]
        }
        
        self$flat_weight_names_ <- c(self$flat_weight_names_, param_names)
        self$all_weights_ <- c(self$all_weights_, param_names)
          
      }
    }
    
    self$flat_weights_ <- lapply(
      self$flat_weights_names_,
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
  }
)