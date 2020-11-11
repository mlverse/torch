context("nn-rnn")

test_that("rnn nonlinearity", {
  rnn <- nn_rnn(1, 10)
  expect_equal(rnn$nonlinearity, "tanh")
  
  rnn <- nn_rnn(1, 10, nonlinearity = "relu")
  expect_equal(rnn$nonlinearity, "relu")
  
  expect_error(
    rnn <- nn_rnn(1, 10, nonlinearity = "garbage"), 
    class = "value_error"
  )
})

test_that("rnn dropout", {
  
  for (p in c(0., .276, .731, 1)) {
    for (train in c(TRUE, FALSE)) {
      
      rnn <- nn_rnn(10, 1000, 2, bias=FALSE, dropout=p, nonlinearity='relu')
      
      with_no_grad({
        
        rnn$weight_ih_l1$fill_(1)
        rnn$weight_hh_l1$fill_(1)
        rnn$weight_ih_l2$fill_(1)
        rnn$weight_hh_l2$fill_(1)
        
      })
      
      if (train)
        rnn$train()
      else
        rnn$eval()
      
      input <- torch_ones(1, 1, 10)
      hx <- torch_zeros(2, 1, 1000)
      
      out <- rnn(input, hx)
      output <- out[[1]]
      hy <- out[[2]]
      
      
      expect_equal_to_tensor(output$min(), output$max(), tolerance = 1e-2)
      
      output_val <- output[1,1,1]
      
      if (p == 0 || !train) {
        expect_equal_to_r(output_val, 10000)
      } else if (p == 1) {
        expect_equal_to_r(output_val, 0)
      } else {
        expect_equal_to_r(output_val > 8000, TRUE)
        expect_equal_to_r(output_val < 12000, TRUE)
      }
      
      expect_equal_to_tensor(hy[1,,]$min(), hy[1,,]$max(), tolerance = 1e-2)
      expect_equal_to_tensor(hy[2,,]$min(), hy[2,,]$max(), tolerance = 1e-2)
      expect_equal_to_r(hy[1,1,1], 10)
      expect_equal_to_tensor(hy[2,1,1], output_val, tolerance = 1e-2)
      
    }
  }
})

test_that("rnn packed sequence", {
  
  x <- torch_tensor(rbind(
    c(1, 2, 0, 0),
    c(1, 2, 3, 0),
    c(1, 2, 3, 4)
  ), dtype = torch_float())
  x <- x[,,newaxis]
  lens <- torch_tensor(c(2,3,4), dtype = torch_long())
  
  p <- nn_utils_rnn_pack_padded_sequence(x, lens, batch_first = TRUE, 
                                         enforce_sorted = FALSE)
  
  rnn <- nn_rnn(1, 4, nonlinearity = "relu")
  out <- rnn(p)
  
  unpack <- nn_utils_rnn_pad_packed_sequence(out[[1]])
  expect_tensor_shape(unpack[[1]], c(4, 3, 4))
  expect_equal_to_r(unpack[[2]]$to(dtype = torch_int()), c(2, 3, 4))
})

test_that("lstm", {
  
  lstm <- nn_lstm(10, 5)
  expect_equal(lstm$mode, "LSTM")
  
  input <- torch_ones(1, 1, 10)
  o <- lstm(input)
  expect_length(o, 2)
  
  expect_tensor_shape(o[[1]], c(1,1,5))
  expect_tensor_shape(o[[2]][[1]], c(1,1,5))
  expect_tensor_shape(o[[2]][[2]], c(1,1,5))
  
  expect_tensor_shape(lstm$weight_ih_l1, c(20, 10))
  expect_tensor_shape(lstm$weight_hh_l1, c(20, 5))
  expect_tensor_shape(lstm$bias_ih_l1, c(20))
  expect_tensor_shape(lstm$bias_hh_l1, c(20))
  
  expect_length(lstm$parameters, 4)
  
  with_no_grad({
    lstm$weight_ih_l1$fill_(1)
    lstm$weight_hh_l1$fill_(1)
    lstm$bias_ih_l1$fill_(1)
    lstm$bias_hh_l1$fill_(1)
  })
  
  z <- lstm(input)
  
  expect_equal_to_tensor(z[[1]], torch_ones(1,1,5) * 0.7615868, tolerance = 1e-5)
  expect_equal_to_tensor(z[[2]][[1]], torch_ones(1,1,5) * 0.7615868, tolerance = 1e-5)
  expect_equal_to_tensor(z[[2]][[2]], torch_ones(1,1,5), tolerance = 1e-5)
  
  lstm <- nn_lstm(10, 5, bias = FALSE)
  
  expect_tensor_shape(lstm$weight_ih_l1, c(20, 10))
  expect_tensor_shape(lstm$weight_hh_l1, c(20, 5))
  expect_null(lstm$bias_ih_l1)
  expect_null(lstm$bias_hh_l1, NULL)
  
  with_no_grad({
    lstm$weight_ih_l1$fill_(1)
    lstm$weight_hh_l1$fill_(1)
  })
  
  z <- lstm(input)
  
  expect_equal_to_tensor(z[[1]], torch_ones(1,1,5) * 0.7615405, tolerance = 1e-5)
  expect_equal_to_tensor(z[[2]][[1]], torch_ones(1,1,5) * 0.7615405, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[2]], torch_ones(1,1,5), tolerance = 1e-4)
  
  lstm <- nn_lstm(10, 5, num_layers = 2)
  expect_length(lstm$parameters, 8)
  lstm <- nn_lstm(10, 5, num_layers = 3)
  expect_length(lstm$parameters, 12)
  
  with_no_grad({
    for (p in lstm$parameters)
      p$fill_(1)
  })
  
  z <- lstm(input)
  expect_equal_to_tensor(z[[1]], torch_ones(1,1,5) * 0.7580, tolerance = 1e-4)
  
  expect_equal_to_tensor(z[[2]][[1]][1,,], torch_ones(1,5) * 0.7616, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[1]][2,,], torch_ones(1,5) * 0.7580, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[1]][3,,], torch_ones(1,5) * 0.7580, tolerance = 1e-4)
  
  expect_equal_to_tensor(z[[2]][[2]][1,,], torch_ones(1,5) , tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[2]][2,,], torch_ones(1,5) * 0.9970 , tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[2]][3,,], torch_ones(1,5) * 0.9969, tolerance = 1e-4)
  
})
