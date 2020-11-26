test_that("multiplication works", {
  fn <- function(inputs) {
    stack <- Stack$new(ptr = inputs)
    t <- stack$at(1)
    o <- torch_relu(t)
    stack <- Stack$new()
    stack$push_back(o)
    stack$ptr
  }
  
  input <- Stack$new()
  input$push_back(torch_tensor(c(-1,0,1)))
  
  o <- cpp_trace_function(fn, input$ptr, .compilation_unit)
  expect_equal(o, 0)
})
