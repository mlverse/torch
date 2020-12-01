test_that("multiplication works", {
  fn <- function(inputs) {
    stack <- Stack$new(ptr = inputs)
    print(stack$to_r())
    t <- stack$at(1)
    o <- torch_relu(t)
    stack <- Stack$new()
    stack$push_back(o)
    stack$ptr
  }
  
  input <- Stack$new()
  input$push_back(torch_tensor(c(-1,0,1)))
  
  o <- cpp_trace_function(fn, input$ptr, .compilation_unit)
  o <- cpp_call_traced_fn(o, input$ptr)
  o <- Stack$new(ptr = o)$to_r()[[1]]
  
  expect_equal_to_tensor(o, torch_relu(torch_tensor(c(-1,0,1))))
})
