#' @include tensor.R

Tensor$set("public", "is_quantized", function() {
  cpp_Tensor_is_quantized(self$ptr)
})
