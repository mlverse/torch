
torch___and__ <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('__and__', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch___lshift__ <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('__lshift__', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch___or__ <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('__or__', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch___rshift__ <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('__rshift__', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch___xor__ <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('__xor__', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__adaptive_avg_pool2d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_adaptive_avg_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__adaptive_avg_pool2d_backward <- function(grad_output, self) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor")
nd_args <- c("grad_output", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_adaptive_avg_pool2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__addr <- function(self, vec1, vec2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "vec1", "vec2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", vec1 = "Tensor", vec2 = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "vec1", "vec2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_addr', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__addr_ <- function(self, vec1, vec2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "vec1", "vec2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", vec1 = "Tensor", vec2 = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "vec1", "vec2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_addr_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__addr_out <- function(out, self, vec1, vec2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "vec1", "vec2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", vec1 = "Tensor", vec2 = "Tensor", 
    beta = "Scalar", alpha = "Scalar")
nd_args <- c("out", "self", "vec1", "vec2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_addr_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__baddbmm_mkl_ <- function(self, batch1, batch2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "batch1", "batch2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", batch1 = "Tensor", batch2 = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "batch1", "batch2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_baddbmm_mkl_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__batch_norm_impl_index <- function(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "running_mean", "running_var", "training", "momentum", "eps", "cudnn_enabled"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", running_mean = "Tensor", 
    running_var = "Tensor", training = "bool", momentum = "double", 
    eps = "double", cudnn_enabled = "bool")
nd_args <- c("input", "weight", "bias", "running_mean", "running_var", "training", 
"momentum", "eps", "cudnn_enabled")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_batch_norm_impl_index', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__batch_norm_impl_index_backward <- function(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace) {
  
args <- rlang::env_get_list(nms = c("impl_index", "input", "grad_output", "weight", "running_mean", "running_var", "save_mean", "save_var_transform", "train", "eps", "output_mask", "reservedSpace"))
args <- Filter(Negate(is.name), args)
expected_types <- list(impl_index = "int64_t", input = "Tensor", grad_output = "Tensor", 
    weight = "Tensor", running_mean = "Tensor", running_var = "Tensor", 
    save_mean = "Tensor", save_var_transform = "Tensor", train = "bool", 
    eps = "double", output_mask = "std::array<bool,3>", reservedSpace = "Tensor")
nd_args <- c("impl_index", "input", "grad_output", "weight", "running_mean", 
"running_var", "save_mean", "save_var_transform", "train", "eps", 
"output_mask", "reservedSpace")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_batch_norm_impl_index_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cast_Byte <- function(self, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", non_blocking = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cast_Byte', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cast_Char <- function(self, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", non_blocking = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cast_Char', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cast_Double <- function(self, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", non_blocking = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cast_Double', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cast_Float <- function(self, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", non_blocking = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cast_Float', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cast_Half <- function(self, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", non_blocking = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cast_Half', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cast_Int <- function(self, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", non_blocking = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cast_Int', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cast_Long <- function(self, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", non_blocking = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cast_Long', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cast_Short <- function(self, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", non_blocking = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cast_Short', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cat <- function(tensors, dim = 0) {
  
args <- rlang::env_get_list(nms = c("tensors", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(tensors = "TensorList", dim = "int64_t")
nd_args <- "tensors"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cat', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cat_out <- function(out, tensors, dim = 0) {
  
args <- rlang::env_get_list(nms = c("out", "tensors", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", tensors = "TensorList", dim = "int64_t")
nd_args <- c("out", "tensors")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cat_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cdist_backward <- function(grad, x1, x2, p, cdist) {
  
args <- rlang::env_get_list(nms = c("grad", "x1", "x2", "p", "cdist"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", x1 = "Tensor", x2 = "Tensor", p = "double", 
    cdist = "Tensor")
nd_args <- c("grad", "x1", "x2", "p", "cdist")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cdist_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cholesky_helper <- function(self, upper) {
  
args <- rlang::env_get_list(nms = c("self", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", upper = "bool")
nd_args <- c("self", "upper")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cholesky_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cholesky_solve_helper <- function(self, A, upper) {
  
args <- rlang::env_get_list(nms = c("self", "A", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", A = "Tensor", upper = "bool")
nd_args <- c("self", "A", "upper")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cholesky_solve_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__convolution <- function(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "dilation", "transposed", "output_padding", "groups", "benchmark", "deterministic", "cudnn_enabled"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", transposed = "bool", 
    output_padding = "IntArrayRef", groups = "int64_t", benchmark = "bool", 
    deterministic = "bool", cudnn_enabled = "bool")
nd_args <- c("input", "weight", "bias", "stride", "padding", "dilation", 
"transposed", "output_padding", "groups", "benchmark", "deterministic", 
"cudnn_enabled")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_convolution', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__convolution_double_backward <- function(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask) {
  
args <- rlang::env_get_list(nms = c("ggI", "ggW", "ggb", "gO", "weight", "self", "stride", "padding", "dilation", "transposed", "output_padding", "groups", "benchmark", "deterministic", "cudnn_enabled", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(ggI = "Tensor", ggW = "Tensor", ggb = "Tensor", gO = "Tensor", 
    weight = "Tensor", self = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", transposed = "bool", 
    output_padding = "IntArrayRef", groups = "int64_t", benchmark = "bool", 
    deterministic = "bool", cudnn_enabled = "bool", output_mask = "std::array<bool,3>")
nd_args <- c("ggI", "ggW", "ggb", "gO", "weight", "self", "stride", "padding", 
"dilation", "transposed", "output_padding", "groups", "benchmark", 
"deterministic", "cudnn_enabled", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_convolution_double_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__convolution_nogroup <- function(input, weight, bias, stride, padding, dilation, transposed, output_padding) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "dilation", "transposed", "output_padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", transposed = "bool", 
    output_padding = "IntArrayRef")
nd_args <- c("input", "weight", "bias", "stride", "padding", "dilation", 
"transposed", "output_padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_convolution_nogroup', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__copy_from <- function(self, dst, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dst", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dst = "Tensor", non_blocking = "bool")
nd_args <- c("self", "dst")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_copy_from', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__ctc_loss <- function(log_probs, targets, input_lengths, target_lengths, blank = 0, zero_infinity = FALSE) {
  
args <- rlang::env_get_list(nms = c("log_probs", "targets", "input_lengths", "target_lengths", "blank", "zero_infinity"))
args <- Filter(Negate(is.name), args)
expected_types <- list(log_probs = "Tensor", targets = "Tensor", input_lengths = "IntArrayRef", 
    target_lengths = "IntArrayRef", blank = "int64_t", zero_infinity = "bool")
nd_args <- c("log_probs", "targets", "input_lengths", "target_lengths")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_ctc_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__ctc_loss_backward <- function(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity = FALSE) {
  
args <- rlang::env_get_list(nms = c("grad", "log_probs", "targets", "input_lengths", "target_lengths", "neg_log_likelihood", "log_alpha", "blank", "zero_infinity"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", log_probs = "Tensor", targets = "Tensor", 
    input_lengths = "IntArrayRef", target_lengths = "IntArrayRef", 
    neg_log_likelihood = "Tensor", log_alpha = "Tensor", blank = "int64_t", 
    zero_infinity = "bool")
nd_args <- c("grad", "log_probs", "targets", "input_lengths", "target_lengths", 
"neg_log_likelihood", "log_alpha", "blank")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_ctc_loss_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cudnn_ctc_loss <- function(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity) {
  
args <- rlang::env_get_list(nms = c("log_probs", "targets", "input_lengths", "target_lengths", "blank", "deterministic", "zero_infinity"))
args <- Filter(Negate(is.name), args)
expected_types <- list(log_probs = "Tensor", targets = "Tensor", input_lengths = "IntArrayRef", 
    target_lengths = "IntArrayRef", blank = "int64_t", deterministic = "bool", 
    zero_infinity = "bool")
nd_args <- c("log_probs", "targets", "input_lengths", "target_lengths", 
"blank", "deterministic", "zero_infinity")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cudnn_ctc_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cudnn_init_dropout_state <- function(dropout, train, dropout_seed, options) {
  
args <- rlang::env_get_list(nms = c("dropout", "train", "dropout_seed", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(dropout = "double", train = "bool", dropout_seed = "int64_t", 
    options = "TensorOptions")
nd_args <- c("dropout", "train", "dropout_seed", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cudnn_init_dropout_state', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cudnn_rnn <- function(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "weight_stride0", "weight_buf", "hx", "cx", "mode", "hidden_size", "num_layers", "batch_first", "dropout", "train", "bidirectional", "batch_sizes", "dropout_state"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "TensorList", weight_stride0 = "int64_t", 
    weight_buf = "Tensor", hx = "Tensor", cx = "Tensor", mode = "int64_t", 
    hidden_size = "int64_t", num_layers = "int64_t", batch_first = "bool", 
    dropout = "double", train = "bool", bidirectional = "bool", 
    batch_sizes = "IntArrayRef", dropout_state = "Tensor")
nd_args <- c("input", "weight", "weight_stride0", "weight_buf", "hx", "cx", 
"mode", "hidden_size", "num_layers", "batch_first", "dropout", 
"train", "bidirectional", "batch_sizes", "dropout_state")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cudnn_rnn', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cudnn_rnn_backward <- function(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "weight_stride0", "weight_buf", "hx", "cx", "output", "grad_output", "grad_hy", "grad_cy", "mode", "hidden_size", "num_layers", "batch_first", "dropout", "train", "bidirectional", "batch_sizes", "dropout_state", "reserve", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "TensorList", weight_stride0 = "int64_t", 
    weight_buf = "Tensor", hx = "Tensor", cx = "Tensor", output = "Tensor", 
    grad_output = "Tensor", grad_hy = "Tensor", grad_cy = "Tensor", 
    mode = "int64_t", hidden_size = "int64_t", num_layers = "int64_t", 
    batch_first = "bool", dropout = "double", train = "bool", 
    bidirectional = "bool", batch_sizes = "IntArrayRef", dropout_state = "Tensor", 
    reserve = "Tensor", output_mask = "std::array<bool,4>")
nd_args <- c("input", "weight", "weight_stride0", "weight_buf", "hx", "cx", 
"output", "grad_output", "grad_hy", "grad_cy", "mode", "hidden_size", 
"num_layers", "batch_first", "dropout", "train", "bidirectional", 
"batch_sizes", "dropout_state", "reserve", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cudnn_rnn_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cudnn_rnn_flatten_weight <- function(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional) {
  
args <- rlang::env_get_list(nms = c("weight_arr", "weight_stride0", "input_size", "mode", "hidden_size", "num_layers", "batch_first", "bidirectional"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight_arr = "TensorList", weight_stride0 = "int64_t", input_size = "int64_t", 
    mode = "int64_t", hidden_size = "int64_t", num_layers = "int64_t", 
    batch_first = "bool", bidirectional = "bool")
nd_args <- c("weight_arr", "weight_stride0", "input_size", "mode", "hidden_size", 
"num_layers", "batch_first", "bidirectional")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cudnn_rnn_flatten_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cufft_clear_plan_cache <- function(device_index) {
  
args <- rlang::env_get_list(nms = c("device_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(device_index = "int64_t")
nd_args <- "device_index"
return_types <- c('void')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cufft_clear_plan_cache', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cufft_get_plan_cache_max_size <- function(device_index) {
  
args <- rlang::env_get_list(nms = c("device_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(device_index = "int64_t")
nd_args <- "device_index"
return_types <- c('int64_t')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cufft_get_plan_cache_max_size', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cufft_get_plan_cache_size <- function(device_index) {
  
args <- rlang::env_get_list(nms = c("device_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(device_index = "int64_t")
nd_args <- "device_index"
return_types <- c('int64_t')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cufft_get_plan_cache_size', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cufft_set_plan_cache_max_size <- function(device_index, max_size) {
  
args <- rlang::env_get_list(nms = c("device_index", "max_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(device_index = "int64_t", max_size = "int64_t")
nd_args <- c("device_index", "max_size")
return_types <- c('void')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cufft_set_plan_cache_max_size', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cumprod <- function(self, dim) {
  
args <- rlang::env_get_list(nms = c("self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cumprod', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cumprod_out <- function(out, self, dim) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = "int64_t")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cumprod_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cumsum <- function(self, dim) {
  
args <- rlang::env_get_list(nms = c("self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cumsum', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__cumsum_out <- function(out, self, dim) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = "int64_t")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_cumsum_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__debug_has_internal_overlap <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('int64_t')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_debug_has_internal_overlap', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__dim_arange <- function(like, dim) {
  
args <- rlang::env_get_list(nms = c("like", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(like = "Tensor", dim = "int64_t")
nd_args <- c("like", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_dim_arange', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__dirichlet_grad <- function(x, alpha, total) {
  
args <- rlang::env_get_list(nms = c("x", "alpha", "total"))
args <- Filter(Negate(is.name), args)
expected_types <- list(x = "Tensor", alpha = "Tensor", total = "Tensor")
nd_args <- c("x", "alpha", "total")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_dirichlet_grad', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__embedding_bag <- function(weight, indices, offsets, scale_grad_by_freq = FALSE, mode = 0, sparse = FALSE, per_sample_weights = list()) {
  
args <- rlang::env_get_list(nms = c("weight", "indices", "offsets", "scale_grad_by_freq", "mode", "sparse", "per_sample_weights"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight = "Tensor", indices = "Tensor", offsets = "Tensor", 
    scale_grad_by_freq = "bool", mode = "int64_t", sparse = "bool", 
    per_sample_weights = "Tensor")
nd_args <- c("weight", "indices", "offsets")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_embedding_bag', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__embedding_bag_backward <- function(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights) {
  
args <- rlang::env_get_list(nms = c("grad", "indices", "offsets", "offset2bag", "bag_size", "maximum_indices", "num_weights", "scale_grad_by_freq", "mode", "sparse", "per_sample_weights"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", indices = "Tensor", offsets = "Tensor", 
    offset2bag = "Tensor", bag_size = "Tensor", maximum_indices = "Tensor", 
    num_weights = "int64_t", scale_grad_by_freq = "bool", mode = "int64_t", 
    sparse = "bool", per_sample_weights = "Tensor")
nd_args <- c("grad", "indices", "offsets", "offset2bag", "bag_size", "maximum_indices", 
"num_weights", "scale_grad_by_freq", "mode", "sparse", "per_sample_weights"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_embedding_bag_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__embedding_bag_dense_backward <- function(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights) {
  
args <- rlang::env_get_list(nms = c("grad", "indices", "offsets", "offset2bag", "bag_size", "maximum_indices", "num_weights", "scale_grad_by_freq", "mode", "per_sample_weights"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", indices = "Tensor", offsets = "Tensor", 
    offset2bag = "Tensor", bag_size = "Tensor", maximum_indices = "Tensor", 
    num_weights = "int64_t", scale_grad_by_freq = "bool", mode = "int64_t", 
    per_sample_weights = "Tensor")
nd_args <- c("grad", "indices", "offsets", "offset2bag", "bag_size", "maximum_indices", 
"num_weights", "scale_grad_by_freq", "mode", "per_sample_weights"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_embedding_bag_dense_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__embedding_bag_per_sample_weights_backward <- function(grad, weight, indices, offsets, offset2bag, mode) {
  
args <- rlang::env_get_list(nms = c("grad", "weight", "indices", "offsets", "offset2bag", "mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", weight = "Tensor", indices = "Tensor", 
    offsets = "Tensor", offset2bag = "Tensor", mode = "int64_t")
nd_args <- c("grad", "weight", "indices", "offsets", "offset2bag", "mode"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_embedding_bag_per_sample_weights_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__embedding_bag_sparse_backward <- function(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights) {
  
args <- rlang::env_get_list(nms = c("grad", "indices", "offsets", "offset2bag", "bag_size", "num_weights", "scale_grad_by_freq", "mode", "per_sample_weights"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", indices = "Tensor", offsets = "Tensor", 
    offset2bag = "Tensor", bag_size = "Tensor", num_weights = "int64_t", 
    scale_grad_by_freq = "bool", mode = "int64_t", per_sample_weights = "Tensor")
nd_args <- c("grad", "indices", "offsets", "offset2bag", "bag_size", "num_weights", 
"scale_grad_by_freq", "mode", "per_sample_weights")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_embedding_bag_sparse_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__empty_affine_quantized <- function(size, options = list(), scale = 1, zero_point = 0, memory_format = torch_contiguous_format()) {
  
args <- rlang::env_get_list(nms = c("size", "options", "scale", "zero_point", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", options = "TensorOptions", scale = "double", 
    zero_point = "int64_t", memory_format = "MemoryFormat")
nd_args <- "size"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_empty_affine_quantized', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__empty_per_channel_affine_quantized <- function(size, scales, zero_points, axis, options = list(), memory_format = torch_contiguous_format()) {
  
args <- rlang::env_get_list(nms = c("size", "scales", "zero_points", "axis", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", scales = "Tensor", zero_points = "Tensor", 
    axis = "int64_t", options = "TensorOptions", memory_format = "MemoryFormat")
nd_args <- c("size", "scales", "zero_points", "axis")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_empty_per_channel_affine_quantized', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__fft_with_size <- function(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes) {
  
args <- rlang::env_get_list(nms = c("self", "signal_ndim", "complex_input", "complex_output", "inverse", "checked_signal_sizes", "normalized", "onesided", "output_sizes"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", signal_ndim = "int64_t", complex_input = "bool", 
    complex_output = "bool", inverse = "bool", checked_signal_sizes = "IntArrayRef", 
    normalized = "bool", onesided = "bool", output_sizes = "IntArrayRef")
nd_args <- c("self", "signal_ndim", "complex_input", "complex_output", "inverse", 
"checked_signal_sizes", "normalized", "onesided", "output_sizes"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_fft_with_size', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__fused_dropout <- function(self, p, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "p", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "double", generator = "Generator *")
nd_args <- c("self", "p")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_fused_dropout', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__gather_sparse_backward <- function(self, dim, index, grad) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index", "grad"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", index = "Tensor", grad = "Tensor")
nd_args <- c("self", "dim", "index", "grad")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_gather_sparse_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__has_compatible_shallow_copy_type <- function(self, from) {
  
args <- rlang::env_get_list(nms = c("self", "from"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", from = "Tensor")
nd_args <- c("self", "from")
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_has_compatible_shallow_copy_type', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__index_copy_ <- function(self, dim, index, source) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index", "source"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", index = "Tensor", source = "Tensor")
nd_args <- c("self", "dim", "index", "source")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_index_copy_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__index_put_impl_ <- function(self, indices, values, accumulate = FALSE, unsafe = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "indices", "values", "accumulate", "unsafe"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", indices = "TensorList", values = "Tensor", 
    accumulate = "bool", unsafe = "bool")
nd_args <- c("self", "indices", "values")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_index_put_impl_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__inverse_helper <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_inverse_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__local_scalar_dense <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Scalar')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_local_scalar_dense', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__log_softmax <- function(self, dim, half_to_float) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "half_to_float"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", half_to_float = "bool")
nd_args <- c("self", "dim", "half_to_float")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_log_softmax', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__log_softmax_backward_data <- function(grad_output, output, dim, self) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output", "dim", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output = "Tensor", dim = "int64_t", 
    self = "Tensor")
nd_args <- c("grad_output", "output", "dim", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_log_softmax_backward_data', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__lu_solve_helper <- function(self, LU_data, LU_pivots) {
  
args <- rlang::env_get_list(nms = c("self", "LU_data", "LU_pivots"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", LU_data = "Tensor", LU_pivots = "Tensor")
nd_args <- c("self", "LU_data", "LU_pivots")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_lu_solve_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__lu_with_info <- function(self, pivot = TRUE, check_errors = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "pivot", "check_errors"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", pivot = "bool", check_errors = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_lu_with_info', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__make_per_channel_quantized_tensor <- function(self, scale, zero_point, axis) {
  
args <- rlang::env_get_list(nms = c("self", "scale", "zero_point", "axis"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", scale = "Tensor", zero_point = "Tensor", 
    axis = "int64_t")
nd_args <- c("self", "scale", "zero_point", "axis")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_make_per_channel_quantized_tensor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__make_per_tensor_quantized_tensor <- function(self, scale, zero_point) {
  
args <- rlang::env_get_list(nms = c("self", "scale", "zero_point"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", scale = "double", zero_point = "int64_t")
nd_args <- c("self", "scale", "zero_point")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_make_per_tensor_quantized_tensor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__masked_scale <- function(self, mask, scale) {
  
args <- rlang::env_get_list(nms = c("self", "mask", "scale"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mask = "Tensor", scale = "double")
nd_args <- c("self", "mask", "scale")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_masked_scale', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__max <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_max', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__max_out <- function(max, max_indices, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("max", "max_indices", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(max = "Tensor", max_indices = "Tensor", self = "Tensor", 
    dim = "int64_t", keepdim = "bool")
nd_args <- c("max", "max_indices", "self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_max_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__min <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_min', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__min_out <- function(min, min_indices, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("min", "min_indices", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(min = "Tensor", min_indices = "Tensor", self = "Tensor", 
    dim = "int64_t", keepdim = "bool")
nd_args <- c("min", "min_indices", "self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_min_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__mkldnn_reshape <- function(self, shape) {
  
args <- rlang::env_get_list(nms = c("self", "shape"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", shape = "IntArrayRef")
nd_args <- c("self", "shape")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_mkldnn_reshape', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__mkldnn_transpose <- function(self, dim0, dim1) {
  
args <- rlang::env_get_list(nms = c("self", "dim0", "dim1"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim0 = "int64_t", dim1 = "int64_t")
nd_args <- c("self", "dim0", "dim1")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_mkldnn_transpose', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__mkldnn_transpose_ <- function(self, dim0, dim1) {
  
args <- rlang::env_get_list(nms = c("self", "dim0", "dim1"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim0 = "int64_t", dim1 = "int64_t")
nd_args <- c("self", "dim0", "dim1")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_mkldnn_transpose_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__mode <- function(self, dim = -1, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", keepdim = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_mode', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__mode_out <- function(values, indices, self, dim = -1, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("values", "indices", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(values = "Tensor", indices = "Tensor", self = "Tensor", 
    dim = "int64_t", keepdim = "bool")
nd_args <- c("values", "indices", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_mode_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__multinomial_alias_draw <- function(J, q, num_samples, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("J", "q", "num_samples", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(J = "Tensor", q = "Tensor", num_samples = "int64_t", generator = "Generator *")
nd_args <- c("J", "q", "num_samples")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_multinomial_alias_draw', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__multinomial_alias_setup <- function(probs) {
  
args <- rlang::env_get_list(nms = c("probs"))
args <- Filter(Negate(is.name), args)
expected_types <- list(probs = "Tensor")
nd_args <- "probs"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_multinomial_alias_setup', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__nnpack_available <- function() {
  
args <- list()
args <- Filter(Negate(is.name), args)
expected_types <- structure(list(), .Names = character(0))
nd_args <- character(0)
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_nnpack_available', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__nnpack_spatial_convolution <- function(input, weight, bias, padding, stride = 1) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", padding = "IntArrayRef", 
    stride = "IntArrayRef")
nd_args <- c("input", "weight", "bias", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_nnpack_spatial_convolution', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__nnpack_spatial_convolution_backward <- function(input, grad_output, weight, padding, output_mask) {
  
args <- rlang::env_get_list(nms = c("input", "grad_output", "weight", "padding", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", output_mask = "std::array<bool,3>")
nd_args <- c("input", "grad_output", "weight", "padding", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_nnpack_spatial_convolution_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__nnpack_spatial_convolution_backward_input <- function(input, grad_output, weight, padding) {
  
args <- rlang::env_get_list(nms = c("input", "grad_output", "weight", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef")
nd_args <- c("input", "grad_output", "weight", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_nnpack_spatial_convolution_backward_input', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__nnpack_spatial_convolution_backward_weight <- function(input, weightsize, grad_output, padding) {
  
args <- rlang::env_get_list(nms = c("input", "weightsize", "grad_output", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weightsize = "IntArrayRef", grad_output = "Tensor", 
    padding = "IntArrayRef")
nd_args <- c("input", "weightsize", "grad_output", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_nnpack_spatial_convolution_backward_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__pack_padded_sequence <- function(input, lengths, batch_first) {
  
args <- rlang::env_get_list(nms = c("input", "lengths", "batch_first"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", lengths = "Tensor", batch_first = "bool")
nd_args <- c("input", "lengths", "batch_first")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_pack_padded_sequence', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__pack_padded_sequence_backward <- function(grad, input_size, batch_sizes, batch_first) {
  
args <- rlang::env_get_list(nms = c("grad", "input_size", "batch_sizes", "batch_first"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", input_size = "IntArrayRef", batch_sizes = "Tensor", 
    batch_first = "bool")
nd_args <- c("grad", "input_size", "batch_sizes", "batch_first")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_pack_padded_sequence_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__pad_packed_sequence <- function(data, batch_sizes, batch_first, padding_value, total_length) {
  
args <- rlang::env_get_list(nms = c("data", "batch_sizes", "batch_first", "padding_value", "total_length"))
args <- Filter(Negate(is.name), args)
expected_types <- list(data = "Tensor", batch_sizes = "Tensor", batch_first = "bool", 
    padding_value = "Scalar", total_length = "int64_t")
nd_args <- c("data", "batch_sizes", "batch_first", "padding_value", "total_length"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_pad_packed_sequence', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__pdist_backward <- function(grad, self, p, pdist) {
  
args <- rlang::env_get_list(nms = c("grad", "self", "p", "pdist"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", self = "Tensor", p = "double", pdist = "Tensor")
nd_args <- c("grad", "self", "p", "pdist")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_pdist_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__pdist_forward <- function(self, p = 2) {
  
args <- rlang::env_get_list(nms = c("self", "p"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "double")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_pdist_forward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__qr_helper <- function(self, some) {
  
args <- rlang::env_get_list(nms = c("self", "some"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", some = "bool")
nd_args <- c("self", "some")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_qr_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__reshape_from_tensor <- function(self, shape) {
  
args <- rlang::env_get_list(nms = c("self", "shape"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", shape = "Tensor")
nd_args <- c("self", "shape")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_reshape_from_tensor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__s_where <- function(condition, self, other) {
  
args <- rlang::env_get_list(nms = c("condition", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(condition = "Tensor", self = "Tensor", other = "Tensor")
nd_args <- c("condition", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_s_where', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sample_dirichlet <- function(self, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", generator = "Generator *")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sample_dirichlet', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__shape_as_tensor <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_shape_as_tensor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sobol_engine_draw <- function(quasi, n, sobolstate, dimension, num_generated, dtype) {
  
args <- rlang::env_get_list(nms = c("quasi", "n", "sobolstate", "dimension", "num_generated", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(quasi = "Tensor", n = "int64_t", sobolstate = "Tensor", 
    dimension = "int64_t", num_generated = "int64_t", dtype = "ScalarType")
nd_args <- c("quasi", "n", "sobolstate", "dimension", "num_generated", "dtype"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sobol_engine_draw', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sobol_engine_ff_ <- function(self, n, sobolstate, dimension, num_generated) {
  
args <- rlang::env_get_list(nms = c("self", "n", "sobolstate", "dimension", "num_generated"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", n = "int64_t", sobolstate = "Tensor", dimension = "int64_t", 
    num_generated = "int64_t")
nd_args <- c("self", "n", "sobolstate", "dimension", "num_generated")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sobol_engine_ff_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sobol_engine_initialize_state_ <- function(self, dimension) {
  
args <- rlang::env_get_list(nms = c("self", "dimension"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dimension = "int64_t")
nd_args <- c("self", "dimension")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sobol_engine_initialize_state_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sobol_engine_scramble_ <- function(self, ltm, dimension) {
  
args <- rlang::env_get_list(nms = c("self", "ltm", "dimension"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", ltm = "Tensor", dimension = "int64_t")
nd_args <- c("self", "ltm", "dimension")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sobol_engine_scramble_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__softmax <- function(self, dim, half_to_float) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "half_to_float"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", half_to_float = "bool")
nd_args <- c("self", "dim", "half_to_float")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_softmax', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__softmax_backward_data <- function(grad_output, output, dim, self) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output", "dim", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output = "Tensor", dim = "int64_t", 
    self = "Tensor")
nd_args <- c("grad_output", "output", "dim", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_softmax_backward_data', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__solve_helper <- function(self, A) {
  
args <- rlang::env_get_list(nms = c("self", "A"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", A = "Tensor")
nd_args <- c("self", "A")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_solve_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sparse_addmm <- function(self, sparse, dense, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "sparse", "dense", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", sparse = "Tensor", dense = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "sparse", "dense")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sparse_addmm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sparse_coo_tensor_unsafe <- function(indices, values, size, options = list()) {
  
args <- rlang::env_get_list(nms = c("indices", "values", "size", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(indices = "Tensor", values = "Tensor", size = "IntArrayRef", 
    options = "TensorOptions")
nd_args <- c("indices", "values", "size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sparse_coo_tensor_unsafe', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sparse_coo_tensor_with_dims <- function(sparse_dim, dense_dim, size, options) {
  
args <- rlang::env_get_list(nms = c("sparse_dim", "dense_dim", "size", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(sparse_dim = "int64_t", dense_dim = "int64_t", size = "IntArrayRef", 
    options = "TensorOptions")
nd_args <- c("sparse_dim", "dense_dim", "size", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sparse_coo_tensor_with_dims', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sparse_coo_tensor_with_dims_and_tensors <- function(sparse_dim, dense_dim, size, indices, values, options) {
  
args <- rlang::env_get_list(nms = c("sparse_dim", "dense_dim", "size", "indices", "values", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(sparse_dim = "int64_t", dense_dim = "int64_t", size = "IntArrayRef", 
    indices = "Tensor", values = "Tensor", options = "TensorOptions")
nd_args <- c("sparse_dim", "dense_dim", "size", "indices", "values", "options"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sparse_coo_tensor_with_dims_and_tensors', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sparse_mm <- function(sparse, dense) {
  
args <- rlang::env_get_list(nms = c("sparse", "dense"))
args <- Filter(Negate(is.name), args)
expected_types <- list(sparse = "Tensor", dense = "Tensor")
nd_args <- c("sparse", "dense")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sparse_mm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sparse_sum <- function(self, dim, dtype) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "IntArrayRef", dtype = "ScalarType")
nd_args <- c("self", "dim", "dtype")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sparse_sum', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__sparse_sum_backward <- function(grad, self, dim) {
  
args <- rlang::env_get_list(nms = c("grad", "self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", self = "Tensor", dim = "IntArrayRef")
nd_args <- c("grad", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_sparse_sum_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__standard_gamma <- function(self, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", generator = "Generator *")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_standard_gamma', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__standard_gamma_grad <- function(self, output) {
  
args <- rlang::env_get_list(nms = c("self", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output = "Tensor")
nd_args <- c("self", "output")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_standard_gamma_grad', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__std <- function(self, unbiased = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "unbiased"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", unbiased = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_std', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__svd_helper <- function(self, some, compute_uv) {
  
args <- rlang::env_get_list(nms = c("self", "some", "compute_uv"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", some = "bool", compute_uv = "bool")
nd_args <- c("self", "some", "compute_uv")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_svd_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__symeig_helper <- function(self, eigenvectors, upper) {
  
args <- rlang::env_get_list(nms = c("self", "eigenvectors", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", eigenvectors = "bool", upper = "bool")
nd_args <- c("self", "eigenvectors", "upper")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_symeig_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__test_optional_float <- function(self, scale = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "scale"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", scale = "double")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_test_optional_float', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__thnn_differentiable_gru_cell_backward <- function(grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias) {
  
args <- rlang::env_get_list(nms = c("grad_hy", "input_gates", "hidden_gates", "hx", "input_bias", "hidden_bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_hy = "Tensor", input_gates = "Tensor", hidden_gates = "Tensor", 
    hx = "Tensor", input_bias = "Tensor", hidden_bias = "Tensor")
nd_args <- c("grad_hy", "input_gates", "hidden_gates", "hx", "input_bias", 
"hidden_bias")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_thnn_differentiable_gru_cell_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__thnn_differentiable_lstm_cell_backward <- function(grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy) {
  
args <- rlang::env_get_list(nms = c("grad_hy", "grad_cy", "input_gates", "hidden_gates", "input_bias", "hidden_bias", "cx", "cy"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_hy = "Tensor", grad_cy = "Tensor", input_gates = "Tensor", 
    hidden_gates = "Tensor", input_bias = "Tensor", hidden_bias = "Tensor", 
    cx = "Tensor", cy = "Tensor")
nd_args <- c("grad_hy", "grad_cy", "input_gates", "hidden_gates", "input_bias", 
"hidden_bias", "cx", "cy")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_thnn_differentiable_lstm_cell_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__thnn_fused_gru_cell <- function(input_gates, hidden_gates, hx, input_bias = list(), hidden_bias = list()) {
  
args <- rlang::env_get_list(nms = c("input_gates", "hidden_gates", "hx", "input_bias", "hidden_bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input_gates = "Tensor", hidden_gates = "Tensor", hx = "Tensor", 
    input_bias = "Tensor", hidden_bias = "Tensor")
nd_args <- c("input_gates", "hidden_gates", "hx")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_thnn_fused_gru_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__thnn_fused_gru_cell_backward <- function(grad_hy, workspace, has_bias) {
  
args <- rlang::env_get_list(nms = c("grad_hy", "workspace", "has_bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_hy = "Tensor", workspace = "Tensor", has_bias = "bool")
nd_args <- c("grad_hy", "workspace", "has_bias")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_thnn_fused_gru_cell_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__thnn_fused_lstm_cell <- function(input_gates, hidden_gates, cx, input_bias = list(), hidden_bias = list()) {
  
args <- rlang::env_get_list(nms = c("input_gates", "hidden_gates", "cx", "input_bias", "hidden_bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input_gates = "Tensor", hidden_gates = "Tensor", cx = "Tensor", 
    input_bias = "Tensor", hidden_bias = "Tensor")
nd_args <- c("input_gates", "hidden_gates", "cx")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_thnn_fused_lstm_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__thnn_fused_lstm_cell_backward <- function(grad_hy, grad_cy, cx, cy, workspace, has_bias) {
  
args <- rlang::env_get_list(nms = c("grad_hy", "grad_cy", "cx", "cy", "workspace", "has_bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_hy = "Tensor", grad_cy = "Tensor", cx = "Tensor", cy = "Tensor", 
    workspace = "Tensor", has_bias = "bool")
nd_args <- c("grad_hy", "grad_cy", "cx", "cy", "workspace", "has_bias")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_thnn_fused_lstm_cell_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__triangular_solve_helper <- function(self, A, upper, transpose, unitriangular) {
  
args <- rlang::env_get_list(nms = c("self", "A", "upper", "transpose", "unitriangular"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", A = "Tensor", upper = "bool", transpose = "bool", 
    unitriangular = "bool")
nd_args <- c("self", "A", "upper", "transpose", "unitriangular")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_triangular_solve_helper', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__trilinear <- function(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim = 1) {
  
args <- rlang::env_get_list(nms = c("i1", "i2", "i3", "expand1", "expand2", "expand3", "sumdim", "unroll_dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(i1 = "Tensor", i2 = "Tensor", i3 = "Tensor", expand1 = "IntArrayRef", 
    expand2 = "IntArrayRef", expand3 = "IntArrayRef", sumdim = "IntArrayRef", 
    unroll_dim = "int64_t")
nd_args <- c("i1", "i2", "i3", "expand1", "expand2", "expand3", "sumdim"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_trilinear', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__unique <- function(self, sorted = TRUE, return_inverse = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "sorted", "return_inverse"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", sorted = "bool", return_inverse = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_unique', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__unique2 <- function(self, sorted = TRUE, return_inverse = FALSE, return_counts = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "sorted", "return_inverse", "return_counts"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", sorted = "bool", return_inverse = "bool", 
    return_counts = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_unique2', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__unsafe_view <- function(self, size) {
  
args <- rlang::env_get_list(nms = c("self", "size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", size = "IntArrayRef")
nd_args <- c("self", "size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_unsafe_view', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__use_cudnn_ctc_loss <- function(log_probs, targets, input_lengths, target_lengths, blank) {
  
args <- rlang::env_get_list(nms = c("log_probs", "targets", "input_lengths", "target_lengths", "blank"))
args <- Filter(Negate(is.name), args)
expected_types <- list(log_probs = "Tensor", targets = "Tensor", input_lengths = "IntArrayRef", 
    target_lengths = "IntArrayRef", blank = "int64_t")
nd_args <- c("log_probs", "targets", "input_lengths", "target_lengths", 
"blank")
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_use_cudnn_ctc_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__var <- function(self, unbiased = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "unbiased"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", unbiased = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_var', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__weight_norm <- function(v, g, dim = 0) {
  
args <- rlang::env_get_list(nms = c("v", "g", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(v = "Tensor", g = "Tensor", dim = "int64_t")
nd_args <- c("v", "g")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_weight_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__weight_norm_cuda_interface <- function(v, g, dim = 0) {
  
args <- rlang::env_get_list(nms = c("v", "g", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(v = "Tensor", g = "Tensor", dim = "int64_t")
nd_args <- c("v", "g")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_weight_norm_cuda_interface', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__weight_norm_cuda_interface_backward <- function(grad_w, saved_v, saved_g, saved_norms, dim) {
  
args <- rlang::env_get_list(nms = c("grad_w", "saved_v", "saved_g", "saved_norms", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_w = "Tensor", saved_v = "Tensor", saved_g = "Tensor", 
    saved_norms = "Tensor", dim = "int64_t")
nd_args <- c("grad_w", "saved_v", "saved_g", "saved_norms", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_weight_norm_cuda_interface_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch__weight_norm_differentiable_backward <- function(grad_w, saved_v, saved_g, saved_norms, dim) {
  
args <- rlang::env_get_list(nms = c("grad_w", "saved_v", "saved_g", "saved_norms", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_w = "Tensor", saved_v = "Tensor", saved_g = "Tensor", 
    saved_norms = "Tensor", dim = "int64_t")
nd_args <- c("grad_w", "saved_v", "saved_g", "saved_norms", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('_weight_norm_differentiable_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_abs <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('abs', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_abs_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('abs_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_abs_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('abs_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_acos <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('acos', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_acos_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('acos_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_acos_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('acos_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_avg_pool1d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_avg_pool1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_avg_pool2d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_avg_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_avg_pool2d_out <- function(out, self, output_size) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("out", "self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_avg_pool2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_avg_pool3d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_avg_pool3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_avg_pool3d_backward <- function(grad_output, self) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor")
nd_args <- c("grad_output", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_avg_pool3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_avg_pool3d_backward_out <- function(grad_input, grad_output, self) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor")
nd_args <- c("grad_input", "grad_output", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_avg_pool3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_avg_pool3d_out <- function(out, self, output_size) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("out", "self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_avg_pool3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool1d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool2d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool2d_backward <- function(grad_output, self, indices) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", indices = "Tensor")
nd_args <- c("grad_output", "self", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool2d_backward_out <- function(grad_input, grad_output, self, indices) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    indices = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool2d_out <- function(out, indices, self, output_size) {
  
args <- rlang::env_get_list(nms = c("out", "indices", "self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", indices = "Tensor", self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("out", "indices", "self", "output_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool3d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool3d_backward <- function(grad_output, self, indices) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", indices = "Tensor")
nd_args <- c("grad_output", "self", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool3d_backward_out <- function(grad_input, grad_output, self, indices) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    indices = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_adaptive_max_pool3d_out <- function(out, indices, self, output_size) {
  
args <- rlang::env_get_list(nms = c("out", "indices", "self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", indices = "Tensor", self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("out", "indices", "self", "output_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('adaptive_max_pool3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_add <- function(self, other, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "other", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Tensor", "Scalar"), alpha = "Scalar")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('add', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_add_out <- function(out, self, other, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = "Tensor", alpha = "Scalar")
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('add_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addbmm <- function(self, batch1, batch2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "batch1", "batch2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", batch1 = "Tensor", batch2 = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "batch1", "batch2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addbmm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addbmm_out <- function(out, self, batch1, batch2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "batch1", "batch2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", batch1 = "Tensor", batch2 = "Tensor", 
    beta = "Scalar", alpha = "Scalar")
nd_args <- c("out", "self", "batch1", "batch2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addbmm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addcdiv <- function(self, tensor1, tensor2, value = 1) {
  
args <- rlang::env_get_list(nms = c("self", "tensor1", "tensor2", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", tensor1 = "Tensor", tensor2 = "Tensor", 
    value = "Scalar")
nd_args <- c("self", "tensor1", "tensor2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addcdiv', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addcdiv_out <- function(out, self, tensor1, tensor2, value = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "tensor1", "tensor2", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", tensor1 = "Tensor", tensor2 = "Tensor", 
    value = "Scalar")
nd_args <- c("out", "self", "tensor1", "tensor2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addcdiv_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addcmul <- function(self, tensor1, tensor2, value = 1) {
  
args <- rlang::env_get_list(nms = c("self", "tensor1", "tensor2", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", tensor1 = "Tensor", tensor2 = "Tensor", 
    value = "Scalar")
nd_args <- c("self", "tensor1", "tensor2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addcmul', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addcmul_out <- function(out, self, tensor1, tensor2, value = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "tensor1", "tensor2", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", tensor1 = "Tensor", tensor2 = "Tensor", 
    value = "Scalar")
nd_args <- c("out", "self", "tensor1", "tensor2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addcmul_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addmm <- function(self, mat1, mat2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "mat1", "mat2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mat1 = "Tensor", mat2 = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "mat1", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addmm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addmm_out <- function(out, self, mat1, mat2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "mat1", "mat2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", mat1 = "Tensor", mat2 = "Tensor", 
    beta = "Scalar", alpha = "Scalar")
nd_args <- c("out", "self", "mat1", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addmm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addmv <- function(self, mat, vec, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "mat", "vec", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mat = "Tensor", vec = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "mat", "vec")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addmv', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addmv_ <- function(self, mat, vec, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "mat", "vec", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mat = "Tensor", vec = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "mat", "vec")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addmv_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addmv_out <- function(out, self, mat, vec, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "mat", "vec", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", mat = "Tensor", vec = "Tensor", 
    beta = "Scalar", alpha = "Scalar")
nd_args <- c("out", "self", "mat", "vec")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addmv_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addr <- function(self, vec1, vec2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "vec1", "vec2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", vec1 = "Tensor", vec2 = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "vec1", "vec2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addr', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_addr_out <- function(out, self, vec1, vec2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "vec1", "vec2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", vec1 = "Tensor", vec2 = "Tensor", 
    beta = "Scalar", alpha = "Scalar")
nd_args <- c("out", "self", "vec1", "vec2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('addr_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_affine_grid_generator <- function(theta, size, align_corners) {
  
args <- rlang::env_get_list(nms = c("theta", "size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(theta = "Tensor", size = "IntArrayRef", align_corners = "bool")
nd_args <- c("theta", "size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('affine_grid_generator', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_affine_grid_generator_backward <- function(grad, size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad", "size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", size = "IntArrayRef", align_corners = "bool")
nd_args <- c("grad", "size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('affine_grid_generator_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_alias <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('alias', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_align_tensors <- function(tensors) {
  
args <- rlang::env_get_list(nms = c("tensors"))
args <- Filter(Negate(is.name), args)
expected_types <- list(tensors = "TensorList")
nd_args <- "tensors"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('align_tensors', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_all <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('all', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_all_out <- function(out, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("int64_t", "Dimname"
), keepdim = "bool")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('all_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_allclose <- function(self, other, rtol = 0.000010, atol = 0.000000, equal_nan = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "other", "rtol", "atol", "equal_nan"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor", rtol = "double", atol = "double", 
    equal_nan = "bool")
nd_args <- c("self", "other")
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('allclose', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_alpha_dropout <- function(input, p, train) {
  
args <- rlang::env_get_list(nms = c("input", "p", "train"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", p = "double", train = "bool")
nd_args <- c("input", "p", "train")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('alpha_dropout', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_alpha_dropout_ <- function(self, p, train) {
  
args <- rlang::env_get_list(nms = c("self", "p", "train"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "double", train = "bool")
nd_args <- c("self", "p", "train")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('alpha_dropout_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_angle <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('angle', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_angle_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('angle_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_any <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('any', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_any_out <- function(out, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("int64_t", "Dimname"
), keepdim = "bool")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('any_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_arange <- function(start, end, step, options = list()) {
  
args <- rlang::env_get_list(nms = c("start", "end", "step", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(start = "Scalar", end = "Scalar", step = "Scalar", options = "TensorOptions")
nd_args <- c("start", "end", "step")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('arange', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_arange_out <- function(out, start, end, step = 1) {
  
args <- rlang::env_get_list(nms = c("out", "start", "end", "step"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", start = "Scalar", end = "Scalar", step = "Scalar")
nd_args <- c("out", "start", "end")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('arange_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_argmax <- function(self, dim = NULL, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", keepdim = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('argmax', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_argmin <- function(self, dim = NULL, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", keepdim = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('argmin', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_argsort <- function(self, dim = -1, descending = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "descending"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), descending = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('argsort', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_as_strided <- function(self, size, stride, storage_offset = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "size", "stride", "storage_offset"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", size = "IntArrayRef", stride = "IntArrayRef", 
    storage_offset = "int64_t")
nd_args <- c("self", "size", "stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('as_strided', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_as_strided_ <- function(self, size, stride, storage_offset = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "size", "stride", "storage_offset"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", size = "IntArrayRef", stride = "IntArrayRef", 
    storage_offset = "int64_t")
nd_args <- c("self", "size", "stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('as_strided_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_asin <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('asin', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_asin_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('asin_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_asin_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('asin_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_atan <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('atan', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_atan_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('atan_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_atan_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('atan_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_atan2 <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('atan2', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_atan2_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = "Tensor")
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('atan2_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool1d <- function(self, kernel_size, stride = list(), padding = 0, ceil_mode = FALSE, count_include_pad = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", ceil_mode = "bool", count_include_pad = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool2d <- function(self, kernel_size, stride = list(), padding = 0, ceil_mode = FALSE, count_include_pad = TRUE, divisor_override = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", ceil_mode = "bool", count_include_pad = "bool", 
    divisor_override = "int64_t")
nd_args <- c("self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool2d_backward <- function(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", ceil_mode = "bool", 
    count_include_pad = "bool", divisor_override = "int64_t")
nd_args <- c("grad_output", "self", "kernel_size", "stride", "padding", 
"ceil_mode", "count_include_pad", "divisor_override")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool2d_backward_out <- function(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    ceil_mode = "bool", count_include_pad = "bool", divisor_override = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "kernel_size", "stride", 
"padding", "ceil_mode", "count_include_pad", "divisor_override"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool2d_out <- function(out, self, kernel_size, stride = list(), padding = 0, ceil_mode = FALSE, count_include_pad = TRUE, divisor_override = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", ceil_mode = "bool", 
    count_include_pad = "bool", divisor_override = "int64_t")
nd_args <- c("out", "self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool3d <- function(self, kernel_size, stride = list(), padding = 0, ceil_mode = FALSE, count_include_pad = TRUE, divisor_override = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", ceil_mode = "bool", count_include_pad = "bool", 
    divisor_override = "int64_t")
nd_args <- c("self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool3d_backward <- function(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", ceil_mode = "bool", 
    count_include_pad = "bool", divisor_override = "int64_t")
nd_args <- c("grad_output", "self", "kernel_size", "stride", "padding", 
"ceil_mode", "count_include_pad", "divisor_override")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool3d_backward_out <- function(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    ceil_mode = "bool", count_include_pad = "bool", divisor_override = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "kernel_size", "stride", 
"padding", "ceil_mode", "count_include_pad", "divisor_override"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_avg_pool3d_out <- function(out, self, kernel_size, stride = list(), padding = 0, ceil_mode = FALSE, count_include_pad = TRUE, divisor_override = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", ceil_mode = "bool", 
    count_include_pad = "bool", divisor_override = "int64_t")
nd_args <- c("out", "self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('avg_pool3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_baddbmm <- function(self, batch1, batch2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "batch1", "batch2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", batch1 = "Tensor", batch2 = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "batch1", "batch2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('baddbmm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_baddbmm_out <- function(out, self, batch1, batch2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "batch1", "batch2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", batch1 = "Tensor", batch2 = "Tensor", 
    beta = "Scalar", alpha = "Scalar")
nd_args <- c("out", "self", "batch1", "batch2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('baddbmm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bartlett_window <- function(window_length, periodic, options = list()) {
  
args <- rlang::env_get_list(nms = c("window_length", "periodic", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(window_length = "int64_t", periodic = "bool", options = "TensorOptions")
nd_args <- c("window_length", "periodic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bartlett_window', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm <- function(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "running_mean", "running_var", "training", "momentum", "eps", "cudnn_enabled"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", running_mean = "Tensor", 
    running_var = "Tensor", training = "bool", momentum = "double", 
    eps = "double", cudnn_enabled = "bool")
nd_args <- c("input", "weight", "bias", "running_mean", "running_var", "training", 
"momentum", "eps", "cudnn_enabled")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm_backward_elemt <- function(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu) {
  
args <- rlang::env_get_list(nms = c("grad_out", "input", "mean", "invstd", "weight", "mean_dy", "mean_dy_xmu"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_out = "Tensor", input = "Tensor", mean = "Tensor", 
    invstd = "Tensor", weight = "Tensor", mean_dy = "Tensor", 
    mean_dy_xmu = "Tensor")
nd_args <- c("grad_out", "input", "mean", "invstd", "weight", "mean_dy", 
"mean_dy_xmu")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm_backward_elemt', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm_backward_reduce <- function(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g) {
  
args <- rlang::env_get_list(nms = c("grad_out", "input", "mean", "invstd", "weight", "input_g", "weight_g", "bias_g"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_out = "Tensor", input = "Tensor", mean = "Tensor", 
    invstd = "Tensor", weight = "Tensor", input_g = "bool", weight_g = "bool", 
    bias_g = "bool")
nd_args <- c("grad_out", "input", "mean", "invstd", "weight", "input_g", 
"weight_g", "bias_g")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm_backward_reduce', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm_elemt <- function(input, weight, bias, mean, invstd, eps) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "mean", "invstd", "eps"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", mean = "Tensor", 
    invstd = "Tensor", eps = "double")
nd_args <- c("input", "weight", "bias", "mean", "invstd", "eps")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm_elemt', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm_elemt_out <- function(out, input, weight, bias, mean, invstd, eps) {
  
args <- rlang::env_get_list(nms = c("out", "input", "weight", "bias", "mean", "invstd", "eps"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", input = "Tensor", weight = "Tensor", bias = "Tensor", 
    mean = "Tensor", invstd = "Tensor", eps = "double")
nd_args <- c("out", "input", "weight", "bias", "mean", "invstd", "eps")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm_elemt_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm_gather_stats <- function(input, mean, invstd, running_mean, running_var, momentum, eps, count) {
  
args <- rlang::env_get_list(nms = c("input", "mean", "invstd", "running_mean", "running_var", "momentum", "eps", "count"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", mean = "Tensor", invstd = "Tensor", running_mean = "Tensor", 
    running_var = "Tensor", momentum = "double", eps = "double", 
    count = "int64_t")
nd_args <- c("input", "mean", "invstd", "running_mean", "running_var", "momentum", 
"eps", "count")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm_gather_stats', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm_gather_stats_with_counts <- function(input, mean, invstd, running_mean, running_var, momentum, eps, counts) {
  
args <- rlang::env_get_list(nms = c("input", "mean", "invstd", "running_mean", "running_var", "momentum", "eps", "counts"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", mean = "Tensor", invstd = "Tensor", running_mean = "Tensor", 
    running_var = "Tensor", momentum = "double", eps = "double", 
    counts = "IntArrayRef")
nd_args <- c("input", "mean", "invstd", "running_mean", "running_var", "momentum", 
"eps", "counts")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm_gather_stats_with_counts', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm_stats <- function(input, eps) {
  
args <- rlang::env_get_list(nms = c("input", "eps"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", eps = "double")
nd_args <- c("input", "eps")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm_stats', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_batch_norm_update_stats <- function(input, running_mean, running_var, momentum) {
  
args <- rlang::env_get_list(nms = c("input", "running_mean", "running_var", "momentum"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", running_mean = "Tensor", running_var = "Tensor", 
    momentum = "double")
nd_args <- c("input", "running_mean", "running_var", "momentum")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('batch_norm_update_stats', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bernoulli <- function(self, p, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "p", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "double", generator = "Generator *")
nd_args <- c("self", "p")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bernoulli', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bernoulli_out <- function(out, self, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", generator = "Generator *")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bernoulli_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bilinear <- function(input1, input2, weight, bias) {
  
args <- rlang::env_get_list(nms = c("input1", "input2", "weight", "bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input1 = "Tensor", input2 = "Tensor", weight = "Tensor", 
    bias = "Tensor")
nd_args <- c("input1", "input2", "weight", "bias")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bilinear', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_binary_cross_entropy <- function(self, target, weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", weight = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('binary_cross_entropy', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_binary_cross_entropy_backward <- function(grad_output, self, target, weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    weight = "Tensor", reduction = "int64_t")
nd_args <- c("grad_output", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('binary_cross_entropy_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_binary_cross_entropy_backward_out <- function(grad_input, grad_output, self, target, weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", weight = "Tensor", reduction = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('binary_cross_entropy_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_binary_cross_entropy_out <- function(out, self, target, weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", weight = "Tensor", 
    reduction = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('binary_cross_entropy_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_binary_cross_entropy_with_logits <- function(self, target, weight = list(), pos_weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "weight", "pos_weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", weight = "Tensor", pos_weight = "Tensor", 
    reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('binary_cross_entropy_with_logits', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_binary_cross_entropy_with_logits_backward <- function(grad_output, self, target, weight = list(), pos_weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "weight", "pos_weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    weight = "Tensor", pos_weight = "Tensor", reduction = "int64_t")
nd_args <- c("grad_output", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('binary_cross_entropy_with_logits_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bincount <- function(self, weights = list(), minlength = 0) {
  
args <- rlang::env_get_list(nms = c("self", "weights", "minlength"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weights = "Tensor", minlength = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bincount', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bitwise_not <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bitwise_not', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bitwise_not_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bitwise_not_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bitwise_xor <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bitwise_xor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bitwise_xor_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Tensor", "Scalar"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bitwise_xor_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_blackman_window <- function(window_length, periodic, options = list()) {
  
args <- rlang::env_get_list(nms = c("window_length", "periodic", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(window_length = "int64_t", periodic = "bool", options = "TensorOptions")
nd_args <- c("window_length", "periodic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('blackman_window', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bmm <- function(self, mat2) {
  
args <- rlang::env_get_list(nms = c("self", "mat2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mat2 = "Tensor")
nd_args <- c("self", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bmm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_bmm_out <- function(out, self, mat2) {
  
args <- rlang::env_get_list(nms = c("out", "self", "mat2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", mat2 = "Tensor")
nd_args <- c("out", "self", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('bmm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_broadcast_tensors <- function(tensors) {
  
args <- rlang::env_get_list(nms = c("tensors"))
args <- Filter(Negate(is.name), args)
expected_types <- list(tensors = "TensorList")
nd_args <- "tensors"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('broadcast_tensors', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_can_cast <- function(from, to) {
  
args <- rlang::env_get_list(nms = c("from", "to"))
args <- Filter(Negate(is.name), args)
expected_types <- list(from = "ScalarType", to = "ScalarType")
nd_args <- c("from", "to")
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('can_cast', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cartesian_prod <- function(tensors) {
  
args <- rlang::env_get_list(nms = c("tensors"))
args <- Filter(Negate(is.name), args)
expected_types <- list(tensors = "TensorList")
nd_args <- "tensors"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cartesian_prod', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cat <- function(tensors, dim = 0) {
  
args <- rlang::env_get_list(nms = c("tensors", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(tensors = "TensorList", dim = c("int64_t", "Dimname"))
nd_args <- "tensors"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cat', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cat_out <- function(out, tensors, dim = 0) {
  
args <- rlang::env_get_list(nms = c("out", "tensors", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", tensors = "TensorList", dim = c("int64_t", 
"Dimname"))
nd_args <- c("out", "tensors")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cat_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cdist <- function(x1, x2, p = 2, compute_mode = NULL) {
  
args <- rlang::env_get_list(nms = c("x1", "x2", "p", "compute_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(x1 = "Tensor", x2 = "Tensor", p = "double", compute_mode = "int64_t")
nd_args <- c("x1", "x2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cdist', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ceil <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ceil', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ceil_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ceil_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ceil_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ceil_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_celu <- function(self, alpha = 1.000000) {
  
args <- rlang::env_get_list(nms = c("self", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", alpha = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('celu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_celu_ <- function(self, alpha = 1.000000) {
  
args <- rlang::env_get_list(nms = c("self", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", alpha = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('celu_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_chain_matmul <- function(matrices) {
  
args <- rlang::env_get_list(nms = c("matrices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(matrices = "TensorList")
nd_args <- "matrices"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('chain_matmul', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cholesky <- function(self, upper = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", upper = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cholesky', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cholesky_inverse <- function(self, upper = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", upper = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cholesky_inverse', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cholesky_inverse_out <- function(out, self, upper = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", upper = "bool")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cholesky_inverse_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cholesky_out <- function(out, self, upper = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", upper = "bool")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cholesky_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cholesky_solve <- function(self, input2, upper = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "input2", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", input2 = "Tensor", upper = "bool")
nd_args <- c("self", "input2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cholesky_solve', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cholesky_solve_out <- function(out, self, input2, upper = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "input2", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", input2 = "Tensor", upper = "bool")
nd_args <- c("out", "self", "input2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cholesky_solve_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_chunk <- function(self, chunks, dim = 0) {
  
args <- rlang::env_get_list(nms = c("self", "chunks", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", chunks = "int64_t", dim = "int64_t")
nd_args <- c("self", "chunks")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('chunk', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp <- function(self, min = NULL, max = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "min", "max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", min = "Scalar", max = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp_ <- function(self, min = NULL, max = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "min", "max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", min = "Scalar", max = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp_max <- function(self, max) {
  
args <- rlang::env_get_list(nms = c("self", "max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", max = "Scalar")
nd_args <- c("self", "max")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp_max', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp_max_ <- function(self, max) {
  
args <- rlang::env_get_list(nms = c("self", "max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", max = "Scalar")
nd_args <- c("self", "max")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp_max_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp_max_out <- function(out, self, max) {
  
args <- rlang::env_get_list(nms = c("out", "self", "max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", max = "Scalar")
nd_args <- c("out", "self", "max")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp_max_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp_min <- function(self, min) {
  
args <- rlang::env_get_list(nms = c("self", "min"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", min = "Scalar")
nd_args <- c("self", "min")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp_min', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp_min_ <- function(self, min) {
  
args <- rlang::env_get_list(nms = c("self", "min"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", min = "Scalar")
nd_args <- c("self", "min")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp_min_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp_min_out <- function(out, self, min) {
  
args <- rlang::env_get_list(nms = c("out", "self", "min"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", min = "Scalar")
nd_args <- c("out", "self", "min")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp_min_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clamp_out <- function(out, self, min = NULL, max = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "min", "max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", min = "Scalar", max = "Scalar")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clamp_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_clone <- function(self, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", memory_format = "MemoryFormat")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('clone', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_col2im <- function(self, output_size, kernel_size, dilation, padding, stride) {
  
args <- rlang::env_get_list(nms = c("self", "output_size", "kernel_size", "dilation", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef", kernel_size = "IntArrayRef", 
    dilation = "IntArrayRef", padding = "IntArrayRef", stride = "IntArrayRef")
nd_args <- c("self", "output_size", "kernel_size", "dilation", "padding", 
"stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('col2im', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_col2im_backward <- function(grad_output, kernel_size, dilation, padding, stride) {
  
args <- rlang::env_get_list(nms = c("grad_output", "kernel_size", "dilation", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", kernel_size = "IntArrayRef", dilation = "IntArrayRef", 
    padding = "IntArrayRef", stride = "IntArrayRef")
nd_args <- c("grad_output", "kernel_size", "dilation", "padding", "stride"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('col2im_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_col2im_backward_out <- function(grad_input, grad_output, kernel_size, dilation, padding, stride) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "kernel_size", "dilation", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", kernel_size = "IntArrayRef", 
    dilation = "IntArrayRef", padding = "IntArrayRef", stride = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "kernel_size", "dilation", "padding", 
"stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('col2im_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_col2im_out <- function(out, self, output_size, kernel_size, dilation, padding, stride) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size", "kernel_size", "dilation", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef", 
    kernel_size = "IntArrayRef", dilation = "IntArrayRef", padding = "IntArrayRef", 
    stride = "IntArrayRef")
nd_args <- c("out", "self", "output_size", "kernel_size", "dilation", "padding", 
"stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('col2im_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_combinations <- function(self, r = 2, with_replacement = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "r", "with_replacement"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", r = "int64_t", with_replacement = "bool")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('combinations', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conj <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conj', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conj_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conj_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_constant_pad_nd <- function(self, pad, value = 0) {
  
args <- rlang::env_get_list(nms = c("self", "pad", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", pad = "IntArrayRef", value = "Scalar")
nd_args <- c("self", "pad")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('constant_pad_nd', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conv_tbc <- function(self, weight, bias, pad = 0) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "bias", "pad"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", bias = "Tensor", pad = "int64_t")
nd_args <- c("self", "weight", "bias")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conv_tbc', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conv_tbc_backward <- function(self, input, weight, bias, pad) {
  
args <- rlang::env_get_list(nms = c("self", "input", "weight", "bias", "pad"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", input = "Tensor", weight = "Tensor", bias = "Tensor", 
    pad = "int64_t")
nd_args <- c("self", "input", "weight", "bias", "pad")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conv_tbc_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conv_transpose1d <- function(input, weight, bias = list(), stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", output_padding = "IntArrayRef", 
    groups = "int64_t", dilation = "IntArrayRef")
nd_args <- c("input", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conv_transpose1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conv_transpose2d <- function(input, weight, bias = list(), stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", output_padding = "IntArrayRef", 
    groups = "int64_t", dilation = "IntArrayRef")
nd_args <- c("input", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conv_transpose2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conv_transpose3d <- function(input, weight, bias = list(), stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", output_padding = "IntArrayRef", 
    groups = "int64_t", dilation = "IntArrayRef")
nd_args <- c("input", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conv_transpose3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conv1d <- function(input, weight, bias = list(), stride = 1, padding = 0, dilation = 1, groups = 1) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "dilation", "groups"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t")
nd_args <- c("input", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conv1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conv2d <- function(input, weight, bias = list(), stride = 1, padding = 0, dilation = 1, groups = 1) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "dilation", "groups"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t")
nd_args <- c("input", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conv2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_conv3d <- function(input, weight, bias = list(), stride = 1, padding = 0, dilation = 1, groups = 1) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "dilation", "groups"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t")
nd_args <- c("input", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('conv3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_convolution <- function(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "dilation", "transposed", "output_padding", "groups"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", transposed = "bool", 
    output_padding = "IntArrayRef", groups = "int64_t")
nd_args <- c("input", "weight", "bias", "stride", "padding", "dilation", 
"transposed", "output_padding", "groups")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('convolution', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_convolution_backward_overrideable <- function(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_output", "input", "weight", "stride", "padding", "dilation", "transposed", "output_padding", "groups", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", input = "Tensor", weight = "Tensor", 
    stride = "IntArrayRef", padding = "IntArrayRef", dilation = "IntArrayRef", 
    transposed = "bool", output_padding = "IntArrayRef", groups = "int64_t", 
    output_mask = "std::array<bool,3>")
nd_args <- c("grad_output", "input", "weight", "stride", "padding", "dilation", 
"transposed", "output_padding", "groups", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('convolution_backward_overrideable', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_convolution_overrideable <- function(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "stride", "padding", "dilation", "transposed", "output_padding", "groups"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", transposed = "bool", 
    output_padding = "IntArrayRef", groups = "int64_t")
nd_args <- c("input", "weight", "bias", "stride", "padding", "dilation", 
"transposed", "output_padding", "groups")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('convolution_overrideable', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_copy_sparse_to_sparse_ <- function(self, src, non_blocking = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "src", "non_blocking"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", src = "Tensor", non_blocking = "bool")
nd_args <- c("self", "src")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('copy_sparse_to_sparse_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cos <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cos', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cos_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cos_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cos_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cos_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cosh <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cosh', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cosh_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cosh_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cosh_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cosh_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cosine_embedding_loss <- function(input1, input2, target, margin = 0.000000, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("input1", "input2", "target", "margin", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input1 = "Tensor", input2 = "Tensor", target = "Tensor", 
    margin = "double", reduction = "int64_t")
nd_args <- c("input1", "input2", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cosine_embedding_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cosine_similarity <- function(x1, x2, dim = 1, eps = 0.000000) {
  
args <- rlang::env_get_list(nms = c("x1", "x2", "dim", "eps"))
args <- Filter(Negate(is.name), args)
expected_types <- list(x1 = "Tensor", x2 = "Tensor", dim = "int64_t", eps = "double")
nd_args <- c("x1", "x2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cosine_similarity', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cross <- function(self, other, dim = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "other", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor", dim = "int64_t")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cross', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cross_out <- function(out, self, other, dim = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = "Tensor", dim = "int64_t")
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cross_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ctc_loss <- function(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = torch_reduction_mean(), zero_infinity = FALSE) {
  
args <- rlang::env_get_list(nms = c("log_probs", "targets", "input_lengths", "target_lengths", "blank", "reduction", "zero_infinity"))
args <- Filter(Negate(is.name), args)
expected_types <- list(log_probs = "Tensor", targets = "Tensor", input_lengths = c("IntArrayRef", 
"Tensor"), target_lengths = c("IntArrayRef", "Tensor"), blank = "int64_t", 
    reduction = "int64_t", zero_infinity = "bool")
nd_args <- c("log_probs", "targets", "input_lengths", "target_lengths")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ctc_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_affine_grid_generator <- function(theta, False, C, H, W) {
  
args <- rlang::env_get_list(nms = c("theta", "False", "C", "H", "W"))
args <- Filter(Negate(is.name), args)
expected_types <- list(theta = "Tensor", False = "int64_t", C = "int64_t", H = "int64_t", 
    W = "int64_t")
nd_args <- c("theta", "False", "C", "H", "W")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_affine_grid_generator', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_affine_grid_generator_backward <- function(grad, False, C, H, W) {
  
args <- rlang::env_get_list(nms = c("grad", "False", "C", "H", "W"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", False = "int64_t", C = "int64_t", H = "int64_t", 
    W = "int64_t")
nd_args <- c("grad", "False", "C", "H", "W")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_affine_grid_generator_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_batch_norm <- function(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "running_mean", "running_var", "training", "exponential_average_factor", "epsilon"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", running_mean = "Tensor", 
    running_var = "Tensor", training = "bool", exponential_average_factor = "double", 
    epsilon = "double")
nd_args <- c("input", "weight", "bias", "running_mean", "running_var", "training", 
"exponential_average_factor", "epsilon")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_batch_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_batch_norm_backward <- function(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace) {
  
args <- rlang::env_get_list(nms = c("input", "grad_output", "weight", "running_mean", "running_var", "save_mean", "save_var", "epsilon", "reserveSpace"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    running_mean = "Tensor", running_var = "Tensor", save_mean = "Tensor", 
    save_var = "Tensor", epsilon = "double", reserveSpace = "Tensor")
nd_args <- c("input", "grad_output", "weight", "running_mean", "running_var", 
"save_mean", "save_var", "epsilon", "reserveSpace")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_batch_norm_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution <- function(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "bias", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", bias = "Tensor", padding = "IntArrayRef", 
    stride = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t", 
    benchmark = "bool", deterministic = "bool")
nd_args <- c("self", "weight", "bias", "padding", "stride", "dilation", 
"groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_backward <- function(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask) {
  
args <- rlang::env_get_list(nms = c("self", "grad_output", "weight", "padding", "stride", "dilation", "groups", "benchmark", "deterministic", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool", 
    output_mask = "std::array<bool,3>")
nd_args <- c("self", "grad_output", "weight", "padding", "stride", "dilation", 
"groups", "benchmark", "deterministic", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_backward_bias <- function(grad_output) {
  
args <- rlang::env_get_list(nms = c("grad_output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor")
nd_args <- "grad_output"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_backward_bias', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_backward_input <- function(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("self_size", "grad_output", "weight", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self_size = "IntArrayRef", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("self_size", "grad_output", "weight", "padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_backward_input', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_backward_weight <- function(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("weight_size", "grad_output", "self", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight_size = "IntArrayRef", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("weight_size", "grad_output", "self", "padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_backward_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_transpose <- function(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "bias", "padding", "output_padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", bias = "Tensor", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("self", "weight", "bias", "padding", "output_padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_transpose', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_transpose_backward <- function(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask) {
  
args <- rlang::env_get_list(nms = c("self", "grad_output", "weight", "padding", "output_padding", "stride", "dilation", "groups", "benchmark", "deterministic", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", output_padding = "IntArrayRef", 
    stride = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t", 
    benchmark = "bool", deterministic = "bool", output_mask = "std::array<bool,3>")
nd_args <- c("self", "grad_output", "weight", "padding", "output_padding", 
"stride", "dilation", "groups", "benchmark", "deterministic", 
"output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_transpose_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_transpose_backward_bias <- function(grad_output) {
  
args <- rlang::env_get_list(nms = c("grad_output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor")
nd_args <- "grad_output"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_transpose_backward_bias', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_transpose_backward_input <- function(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("grad_output", "weight", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", weight = "Tensor", padding = "IntArrayRef", 
    stride = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t", 
    benchmark = "bool", deterministic = "bool")
nd_args <- c("grad_output", "weight", "padding", "stride", "dilation", "groups", 
"benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_transpose_backward_input', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_convolution_transpose_backward_weight <- function(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("weight_size", "grad_output", "self", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight_size = "IntArrayRef", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("weight_size", "grad_output", "self", "padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_convolution_transpose_backward_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_grid_sampler <- function(self, grid) {
  
args <- rlang::env_get_list(nms = c("self", "grid"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", grid = "Tensor")
nd_args <- c("self", "grid")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_grid_sampler', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_grid_sampler_backward <- function(self, grid, grad_output) {
  
args <- rlang::env_get_list(nms = c("self", "grid", "grad_output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", grid = "Tensor", grad_output = "Tensor")
nd_args <- c("self", "grid", "grad_output")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_grid_sampler_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cudnn_is_acceptable <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cudnn_is_acceptable', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cumprod <- function(self, dim, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), dtype = "ScalarType")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cumprod', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cumprod_out <- function(out, self, dim, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("int64_t", "Dimname"
), dtype = "ScalarType")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cumprod_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cumsum <- function(self, dim, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), dtype = "ScalarType")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cumsum', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_cumsum_out <- function(out, self, dim, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("int64_t", "Dimname"
), dtype = "ScalarType")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('cumsum_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_dequantize <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('dequantize', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_det <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('det', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_detach <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('detach', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_detach_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('detach_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_diag <- function(self, diagonal = 0) {
  
args <- rlang::env_get_list(nms = c("self", "diagonal"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", diagonal = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('diag', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_diag_embed <- function(self, offset = 0, dim1 = -2, dim2 = -1) {
  
args <- rlang::env_get_list(nms = c("self", "offset", "dim1", "dim2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", offset = "int64_t", dim1 = "int64_t", dim2 = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('diag_embed', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_diag_out <- function(out, self, diagonal = 0) {
  
args <- rlang::env_get_list(nms = c("out", "self", "diagonal"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", diagonal = "int64_t")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('diag_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_diagflat <- function(self, offset = 0) {
  
args <- rlang::env_get_list(nms = c("self", "offset"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", offset = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('diagflat', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_diagonal <- function(self, offset = 0, dim1 = 0, dim2 = 1) {
  
args <- rlang::env_get_list(nms = c("self", "offset", "dim1", "dim2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", offset = "int64_t", dim1 = "int64_t", dim2 = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('diagonal', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_digamma <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('digamma', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_digamma_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('digamma_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_dist <- function(self, other, p = 2) {
  
args <- rlang::env_get_list(nms = c("self", "other", "p"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor", p = "Scalar")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('dist', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_div <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Tensor", "Scalar"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('div', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_div_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = "Tensor")
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('div_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_dot <- function(self, tensor) {
  
args <- rlang::env_get_list(nms = c("self", "tensor"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", tensor = "Tensor")
nd_args <- c("self", "tensor")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('dot', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_dot_out <- function(out, self, tensor) {
  
args <- rlang::env_get_list(nms = c("out", "self", "tensor"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", tensor = "Tensor")
nd_args <- c("out", "self", "tensor")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('dot_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_dropout <- function(input, p, train) {
  
args <- rlang::env_get_list(nms = c("input", "p", "train"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", p = "double", train = "bool")
nd_args <- c("input", "p", "train")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('dropout', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_dropout_ <- function(self, p, train) {
  
args <- rlang::env_get_list(nms = c("self", "p", "train"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "double", train = "bool")
nd_args <- c("self", "p", "train")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('dropout_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_eig <- function(self, eigenvectors = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "eigenvectors"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", eigenvectors = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('eig', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_eig_out <- function(e, v, self, eigenvectors = FALSE) {
  
args <- rlang::env_get_list(nms = c("e", "v", "self", "eigenvectors"))
args <- Filter(Negate(is.name), args)
expected_types <- list(e = "Tensor", v = "Tensor", self = "Tensor", eigenvectors = "bool")
nd_args <- c("e", "v", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('eig_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_einsum <- function(equation, tensors) {
  
args <- rlang::env_get_list(nms = c("equation", "tensors"))
args <- Filter(Negate(is.name), args)
expected_types <- list(equation = "std::string", tensors = "TensorList")
nd_args <- c("equation", "tensors")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('einsum', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_elu <- function(self, alpha = 1, scale = 1, input_scale = 1) {
  
args <- rlang::env_get_list(nms = c("self", "alpha", "scale", "input_scale"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", alpha = "Scalar", scale = "Scalar", input_scale = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('elu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_elu_ <- function(self, alpha = 1, scale = 1, input_scale = 1) {
  
args <- rlang::env_get_list(nms = c("self", "alpha", "scale", "input_scale"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", alpha = "Scalar", scale = "Scalar", input_scale = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('elu_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_elu_backward <- function(grad_output, alpha, scale, input_scale, output) {
  
args <- rlang::env_get_list(nms = c("grad_output", "alpha", "scale", "input_scale", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", alpha = "Scalar", scale = "Scalar", 
    input_scale = "Scalar", output = "Tensor")
nd_args <- c("grad_output", "alpha", "scale", "input_scale", "output")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('elu_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_elu_backward_out <- function(grad_input, grad_output, alpha, scale, input_scale, output) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "alpha", "scale", "input_scale", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", alpha = "Scalar", 
    scale = "Scalar", input_scale = "Scalar", output = "Tensor")
nd_args <- c("grad_input", "grad_output", "alpha", "scale", "input_scale", 
"output")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('elu_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_elu_out <- function(out, self, alpha = 1, scale = 1, input_scale = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "alpha", "scale", "input_scale"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", alpha = "Scalar", scale = "Scalar", 
    input_scale = "Scalar")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('elu_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_embedding <- function(weight, indices, padding_idx = -1, scale_grad_by_freq = FALSE, sparse = FALSE) {
  
args <- rlang::env_get_list(nms = c("weight", "indices", "padding_idx", "scale_grad_by_freq", "sparse"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight = "Tensor", indices = "Tensor", padding_idx = "int64_t", 
    scale_grad_by_freq = "bool", sparse = "bool")
nd_args <- c("weight", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('embedding', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_embedding_backward <- function(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse) {
  
args <- rlang::env_get_list(nms = c("grad", "indices", "num_weights", "padding_idx", "scale_grad_by_freq", "sparse"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", indices = "Tensor", num_weights = "int64_t", 
    padding_idx = "int64_t", scale_grad_by_freq = "bool", sparse = "bool")
nd_args <- c("grad", "indices", "num_weights", "padding_idx", "scale_grad_by_freq", 
"sparse")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('embedding_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_embedding_bag <- function(weight, indices, offsets, scale_grad_by_freq = FALSE, mode = 0, sparse = FALSE, per_sample_weights = list()) {
  
args <- rlang::env_get_list(nms = c("weight", "indices", "offsets", "scale_grad_by_freq", "mode", "sparse", "per_sample_weights"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight = "Tensor", indices = "Tensor", offsets = "Tensor", 
    scale_grad_by_freq = "bool", mode = "int64_t", sparse = "bool", 
    per_sample_weights = "Tensor")
nd_args <- c("weight", "indices", "offsets")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('embedding_bag', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_embedding_dense_backward <- function(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq) {
  
args <- rlang::env_get_list(nms = c("grad_output", "indices", "num_weights", "padding_idx", "scale_grad_by_freq"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", indices = "Tensor", num_weights = "int64_t", 
    padding_idx = "int64_t", scale_grad_by_freq = "bool")
nd_args <- c("grad_output", "indices", "num_weights", "padding_idx", "scale_grad_by_freq"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('embedding_dense_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_embedding_renorm_ <- function(self, indices, max_norm, norm_type) {
  
args <- rlang::env_get_list(nms = c("self", "indices", "max_norm", "norm_type"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", indices = "Tensor", max_norm = "double", 
    norm_type = "double")
nd_args <- c("self", "indices", "max_norm", "norm_type")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('embedding_renorm_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_embedding_sparse_backward <- function(grad, indices, num_weights, padding_idx, scale_grad_by_freq) {
  
args <- rlang::env_get_list(nms = c("grad", "indices", "num_weights", "padding_idx", "scale_grad_by_freq"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", indices = "Tensor", num_weights = "int64_t", 
    padding_idx = "int64_t", scale_grad_by_freq = "bool")
nd_args <- c("grad", "indices", "num_weights", "padding_idx", "scale_grad_by_freq"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('embedding_sparse_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_empty <- function(size, names, options = list(), memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("size", "names", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", names = "DimnameList", options = "TensorOptions", 
    memory_format = "MemoryFormat")
nd_args <- c("size", "names")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('empty', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_empty_like <- function(self, options, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", options = "TensorOptions", memory_format = "MemoryFormat")
nd_args <- c("self", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('empty_like', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_empty_out <- function(out, size, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "size", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", size = "IntArrayRef", memory_format = "MemoryFormat")
nd_args <- c("out", "size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('empty_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_empty_strided <- function(size, stride, options = list()) {
  
args <- rlang::env_get_list(nms = c("size", "stride", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", stride = "IntArrayRef", options = "TensorOptions")
nd_args <- c("size", "stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('empty_strided', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_eq <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('eq', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_eq_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Scalar", "Tensor"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('eq_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_equal <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor")
nd_args <- c("self", "other")
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('equal', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_erf <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('erf', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_erf_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('erf_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_erf_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('erf_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_erfc <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('erfc', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_erfc_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('erfc_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_erfc_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('erfc_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_erfinv <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('erfinv', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_erfinv_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('erfinv_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_exp <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('exp', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_exp_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('exp_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_exp_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('exp_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_expm1 <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('expm1', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_expm1_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('expm1_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_expm1_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('expm1_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_eye <- function(n, m, options = list()) {
  
args <- rlang::env_get_list(nms = c("n", "m", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(n = "int64_t", m = "int64_t", options = "TensorOptions")
nd_args <- c("n", "m")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('eye', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_eye_out <- function(out, n, m) {
  
args <- rlang::env_get_list(nms = c("out", "n", "m"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", n = "int64_t", m = "int64_t")
nd_args <- c("out", "n", "m")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('eye_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fake_quantize_per_channel_affine <- function(self, scale, zero_point, axis, quant_min, quant_max) {
  
args <- rlang::env_get_list(nms = c("self", "scale", "zero_point", "axis", "quant_min", "quant_max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", scale = "Tensor", zero_point = "Tensor", 
    axis = "int64_t", quant_min = "int64_t", quant_max = "int64_t")
nd_args <- c("self", "scale", "zero_point", "axis", "quant_min", "quant_max"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fake_quantize_per_channel_affine', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fake_quantize_per_channel_affine_backward <- function(grad, self, scale, zero_point, axis, quant_min, quant_max) {
  
args <- rlang::env_get_list(nms = c("grad", "self", "scale", "zero_point", "axis", "quant_min", "quant_max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", self = "Tensor", scale = "Tensor", zero_point = "Tensor", 
    axis = "int64_t", quant_min = "int64_t", quant_max = "int64_t")
nd_args <- c("grad", "self", "scale", "zero_point", "axis", "quant_min", 
"quant_max")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fake_quantize_per_channel_affine_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fake_quantize_per_tensor_affine <- function(self, scale, zero_point, quant_min, quant_max) {
  
args <- rlang::env_get_list(nms = c("self", "scale", "zero_point", "quant_min", "quant_max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", scale = "double", zero_point = "int64_t", 
    quant_min = "int64_t", quant_max = "int64_t")
nd_args <- c("self", "scale", "zero_point", "quant_min", "quant_max")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fake_quantize_per_tensor_affine', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fake_quantize_per_tensor_affine_backward <- function(grad, self, scale, zero_point, quant_min, quant_max) {
  
args <- rlang::env_get_list(nms = c("grad", "self", "scale", "zero_point", "quant_min", "quant_max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", self = "Tensor", scale = "double", zero_point = "int64_t", 
    quant_min = "int64_t", quant_max = "int64_t")
nd_args <- c("grad", "self", "scale", "zero_point", "quant_min", "quant_max"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fake_quantize_per_tensor_affine_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fbgemm_linear_fp16_weight <- function(input, packed_weight, bias) {
  
args <- rlang::env_get_list(nms = c("input", "packed_weight", "bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", packed_weight = "Tensor", bias = "Tensor")
nd_args <- c("input", "packed_weight", "bias")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fbgemm_linear_fp16_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fbgemm_linear_fp16_weight_fp32_activation <- function(input, packed_weight, bias) {
  
args <- rlang::env_get_list(nms = c("input", "packed_weight", "bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", packed_weight = "Tensor", bias = "Tensor")
nd_args <- c("input", "packed_weight", "bias")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fbgemm_linear_fp16_weight_fp32_activation', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fbgemm_linear_int8_weight <- function(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "packed", "col_offsets", "weight_scale", "weight_zero_point", "bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", packed = "Tensor", 
    col_offsets = "Tensor", weight_scale = "Scalar", weight_zero_point = "Scalar", 
    bias = "Tensor")
nd_args <- c("input", "weight", "packed", "col_offsets", "weight_scale", 
"weight_zero_point", "bias")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fbgemm_linear_int8_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fbgemm_linear_int8_weight_fp32_activation <- function(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "packed", "col_offsets", "weight_scale", "weight_zero_point", "bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", packed = "Tensor", 
    col_offsets = "Tensor", weight_scale = "Scalar", weight_zero_point = "Scalar", 
    bias = "Tensor")
nd_args <- c("input", "weight", "packed", "col_offsets", "weight_scale", 
"weight_zero_point", "bias")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fbgemm_linear_int8_weight_fp32_activation', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fbgemm_linear_quantize_weight <- function(input) {
  
args <- rlang::env_get_list(nms = c("input"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor")
nd_args <- "input"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fbgemm_linear_quantize_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fbgemm_pack_gemm_matrix_fp16 <- function(input) {
  
args <- rlang::env_get_list(nms = c("input"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor")
nd_args <- "input"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fbgemm_pack_gemm_matrix_fp16', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fbgemm_pack_quantized_matrix <- function(input, K, False) {
  
args <- rlang::env_get_list(nms = c("input", "K", "False"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", K = "int64_t", False = "int64_t")
nd_args <- c("input", "K", "False")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fbgemm_pack_quantized_matrix', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_feature_alpha_dropout <- function(input, p, train) {
  
args <- rlang::env_get_list(nms = c("input", "p", "train"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", p = "double", train = "bool")
nd_args <- c("input", "p", "train")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('feature_alpha_dropout', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_feature_alpha_dropout_ <- function(self, p, train) {
  
args <- rlang::env_get_list(nms = c("self", "p", "train"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "double", train = "bool")
nd_args <- c("self", "p", "train")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('feature_alpha_dropout_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_feature_dropout <- function(input, p, train) {
  
args <- rlang::env_get_list(nms = c("input", "p", "train"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", p = "double", train = "bool")
nd_args <- c("input", "p", "train")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('feature_dropout', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_feature_dropout_ <- function(self, p, train) {
  
args <- rlang::env_get_list(nms = c("self", "p", "train"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "double", train = "bool")
nd_args <- c("self", "p", "train")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('feature_dropout_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fft <- function(self, signal_ndim, normalized = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "signal_ndim", "normalized"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", signal_ndim = "int64_t", normalized = "bool")
nd_args <- c("self", "signal_ndim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fft', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fill_ <- function(self, value) {
  
args <- rlang::env_get_list(nms = c("self", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", value = c("Scalar", "Tensor"))
nd_args <- c("self", "value")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fill_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_flatten <- function(self, dims, start_dim = 0, end_dim = -1, out_dim) {
  
args <- rlang::env_get_list(nms = c("self", "dims", "start_dim", "end_dim", "out_dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dims = "DimnameList", start_dim = c("int64_t", 
"Dimname"), end_dim = c("int64_t", "Dimname"), out_dim = "Dimname")
nd_args <- c("self", "dims", "out_dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('flatten', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_flip <- function(self, dims) {
  
args <- rlang::env_get_list(nms = c("self", "dims"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dims = "IntArrayRef")
nd_args <- c("self", "dims")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('flip', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_floor <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('floor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_floor_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('floor_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_floor_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('floor_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fmod <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fmod', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fmod_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Scalar", "Tensor"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fmod_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_frac <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('frac', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_frac_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('frac_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_frac_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('frac_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fractional_max_pool2d <- function(self, kernel_size, output_size, random_samples) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "output_size", "random_samples"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", output_size = "IntArrayRef", 
    random_samples = "Tensor")
nd_args <- c("self", "kernel_size", "output_size", "random_samples")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fractional_max_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fractional_max_pool2d_backward <- function(grad_output, self, kernel_size, output_size, indices) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "kernel_size", "output_size", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    output_size = "IntArrayRef", indices = "Tensor")
nd_args <- c("grad_output", "self", "kernel_size", "output_size", "indices"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fractional_max_pool2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fractional_max_pool2d_backward_out <- function(grad_input, grad_output, self, kernel_size, output_size, indices) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "kernel_size", "output_size", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    kernel_size = "IntArrayRef", output_size = "IntArrayRef", 
    indices = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "kernel_size", "output_size", 
"indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fractional_max_pool2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fractional_max_pool2d_out <- function(output, indices, self, kernel_size, output_size, random_samples) {
  
args <- rlang::env_get_list(nms = c("output", "indices", "self", "kernel_size", "output_size", "random_samples"))
args <- Filter(Negate(is.name), args)
expected_types <- list(output = "Tensor", indices = "Tensor", self = "Tensor", 
    kernel_size = "IntArrayRef", output_size = "IntArrayRef", 
    random_samples = "Tensor")
nd_args <- c("output", "indices", "self", "kernel_size", "output_size", 
"random_samples")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fractional_max_pool2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fractional_max_pool3d <- function(self, kernel_size, output_size, random_samples) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "output_size", "random_samples"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", output_size = "IntArrayRef", 
    random_samples = "Tensor")
nd_args <- c("self", "kernel_size", "output_size", "random_samples")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fractional_max_pool3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fractional_max_pool3d_backward <- function(grad_output, self, kernel_size, output_size, indices) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "kernel_size", "output_size", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    output_size = "IntArrayRef", indices = "Tensor")
nd_args <- c("grad_output", "self", "kernel_size", "output_size", "indices"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fractional_max_pool3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fractional_max_pool3d_backward_out <- function(grad_input, grad_output, self, kernel_size, output_size, indices) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "kernel_size", "output_size", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    kernel_size = "IntArrayRef", output_size = "IntArrayRef", 
    indices = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "kernel_size", "output_size", 
"indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fractional_max_pool3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_fractional_max_pool3d_out <- function(output, indices, self, kernel_size, output_size, random_samples) {
  
args <- rlang::env_get_list(nms = c("output", "indices", "self", "kernel_size", "output_size", "random_samples"))
args <- Filter(Negate(is.name), args)
expected_types <- list(output = "Tensor", indices = "Tensor", self = "Tensor", 
    kernel_size = "IntArrayRef", output_size = "IntArrayRef", 
    random_samples = "Tensor")
nd_args <- c("output", "indices", "self", "kernel_size", "output_size", 
"random_samples")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('fractional_max_pool3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_frobenius_norm <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "IntArrayRef", keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('frobenius_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_frobenius_norm_out <- function(out, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = "IntArrayRef", keepdim = "bool")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('frobenius_norm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_from_file <- function(filename, shared = NULL, size = 0, options = list()) {
  
args <- rlang::env_get_list(nms = c("filename", "shared", "size", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(filename = "std::string", shared = "bool", size = "int64_t", 
    options = "TensorOptions")
nd_args <- "filename"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('from_file', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_full <- function(size, fill_value, names, options = list()) {
  
args <- rlang::env_get_list(nms = c("size", "fill_value", "names", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", fill_value = "Scalar", names = "DimnameList", 
    options = "TensorOptions")
nd_args <- c("size", "fill_value", "names")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('full', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_full_like <- function(self, fill_value, options, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "fill_value", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", fill_value = "Scalar", options = "TensorOptions", 
    memory_format = "MemoryFormat")
nd_args <- c("self", "fill_value", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('full_like', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_full_out <- function(out, size, fill_value) {
  
args <- rlang::env_get_list(nms = c("out", "size", "fill_value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", size = "IntArrayRef", fill_value = "Scalar")
nd_args <- c("out", "size", "fill_value")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('full_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_gather <- function(self, dim, index, sparse_grad = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index", "sparse_grad"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), index = "Tensor", 
    sparse_grad = "bool")
nd_args <- c("self", "dim", "index")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('gather', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_gather_out <- function(out, self, dim, index, sparse_grad = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "index", "sparse_grad"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("int64_t", "Dimname"
), index = "Tensor", sparse_grad = "bool")
nd_args <- c("out", "self", "dim", "index")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('gather_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ge <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ge', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ge_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Scalar", "Tensor"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ge_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_gelu <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('gelu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_gelu_backward <- function(grad, self) {
  
args <- rlang::env_get_list(nms = c("grad", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", self = "Tensor")
nd_args <- c("grad", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('gelu_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_geqrf <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('geqrf', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_geqrf_out <- function(a, tau, self) {
  
args <- rlang::env_get_list(nms = c("a", "tau", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(a = "Tensor", tau = "Tensor", self = "Tensor")
nd_args <- c("a", "tau", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('geqrf_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ger <- function(self, vec2) {
  
args <- rlang::env_get_list(nms = c("self", "vec2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", vec2 = "Tensor")
nd_args <- c("self", "vec2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ger', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ger_out <- function(out, self, vec2) {
  
args <- rlang::env_get_list(nms = c("out", "self", "vec2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", vec2 = "Tensor")
nd_args <- c("out", "self", "vec2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ger_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_glu <- function(self, dim = -1) {
  
args <- rlang::env_get_list(nms = c("self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('glu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_glu_backward <- function(grad_output, self, dim) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", dim = "int64_t")
nd_args <- c("grad_output", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('glu_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_glu_backward_out <- function(grad_input, grad_output, self, dim) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    dim = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('glu_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_glu_out <- function(out, self, dim = -1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = "int64_t")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('glu_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_grid_sampler <- function(input, grid, interpolation_mode, padding_mode, align_corners) {
  
args <- rlang::env_get_list(nms = c("input", "grid", "interpolation_mode", "padding_mode", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", grid = "Tensor", interpolation_mode = "int64_t", 
    padding_mode = "int64_t", align_corners = "bool")
nd_args <- c("input", "grid", "interpolation_mode", "padding_mode", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('grid_sampler', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_grid_sampler_2d <- function(input, grid, interpolation_mode, padding_mode, align_corners) {
  
args <- rlang::env_get_list(nms = c("input", "grid", "interpolation_mode", "padding_mode", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", grid = "Tensor", interpolation_mode = "int64_t", 
    padding_mode = "int64_t", align_corners = "bool")
nd_args <- c("input", "grid", "interpolation_mode", "padding_mode", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('grid_sampler_2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_grid_sampler_2d_backward <- function(grad_output, input, grid, interpolation_mode, padding_mode, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_output", "input", "grid", "interpolation_mode", "padding_mode", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", input = "Tensor", grid = "Tensor", 
    interpolation_mode = "int64_t", padding_mode = "int64_t", 
    align_corners = "bool")
nd_args <- c("grad_output", "input", "grid", "interpolation_mode", "padding_mode", 
"align_corners")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('grid_sampler_2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_grid_sampler_3d <- function(input, grid, interpolation_mode, padding_mode, align_corners) {
  
args <- rlang::env_get_list(nms = c("input", "grid", "interpolation_mode", "padding_mode", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", grid = "Tensor", interpolation_mode = "int64_t", 
    padding_mode = "int64_t", align_corners = "bool")
nd_args <- c("input", "grid", "interpolation_mode", "padding_mode", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('grid_sampler_3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_grid_sampler_3d_backward <- function(grad_output, input, grid, interpolation_mode, padding_mode, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_output", "input", "grid", "interpolation_mode", "padding_mode", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", input = "Tensor", grid = "Tensor", 
    interpolation_mode = "int64_t", padding_mode = "int64_t", 
    align_corners = "bool")
nd_args <- c("grad_output", "input", "grid", "interpolation_mode", "padding_mode", 
"align_corners")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('grid_sampler_3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_group_norm <- function(input, num_groups, weight = list(), bias = list(), eps = 0.000010, cudnn_enabled = TRUE) {
  
args <- rlang::env_get_list(nms = c("input", "num_groups", "weight", "bias", "eps", "cudnn_enabled"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", num_groups = "int64_t", weight = "Tensor", 
    bias = "Tensor", eps = "double", cudnn_enabled = "bool")
nd_args <- c("input", "num_groups")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('group_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_gru <- function(data, input, batch_sizes, hx, params, has_biases, num_layers, dropout, train, batch_first, bidirectional) {
  
args <- rlang::env_get_list(nms = c("data", "input", "batch_sizes", "hx", "params", "has_biases", "num_layers", "dropout", "train", "batch_first", "bidirectional"))
args <- Filter(Negate(is.name), args)
expected_types <- list(data = "Tensor", input = "Tensor", batch_sizes = "Tensor", 
    hx = "Tensor", params = "TensorList", has_biases = "bool", 
    num_layers = "int64_t", dropout = "double", train = "bool", 
    batch_first = "bool", bidirectional = "bool")
nd_args <- c("data", "input", "batch_sizes", "hx", "params", "has_biases", 
"num_layers", "dropout", "train", "batch_first", "bidirectional"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('gru', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_gru_cell <- function(input, hx, w_ih, w_hh, b_ih = list(), b_hh = list()) {
  
args <- rlang::env_get_list(nms = c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", hx = "Tensor", w_ih = "Tensor", w_hh = "Tensor", 
    b_ih = "Tensor", b_hh = "Tensor")
nd_args <- c("input", "hx", "w_ih", "w_hh")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('gru_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_gt <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('gt', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_gt_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Scalar", "Tensor"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('gt_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hamming_window <- function(window_length, periodic, alpha, beta, options = list()) {
  
args <- rlang::env_get_list(nms = c("window_length", "periodic", "alpha", "beta", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(window_length = "int64_t", periodic = "bool", alpha = "double", 
    beta = "double", options = "TensorOptions")
nd_args <- c("window_length", "periodic", "alpha", "beta")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hamming_window', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hann_window <- function(window_length, periodic, options = list()) {
  
args <- rlang::env_get_list(nms = c("window_length", "periodic", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(window_length = "int64_t", periodic = "bool", options = "TensorOptions")
nd_args <- c("window_length", "periodic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hann_window', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hardshrink <- function(self, lambd = 0.500000) {
  
args <- rlang::env_get_list(nms = c("self", "lambd"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", lambd = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hardshrink', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hardshrink_backward <- function(grad_out, self, lambd) {
  
args <- rlang::env_get_list(nms = c("grad_out", "self", "lambd"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_out = "Tensor", self = "Tensor", lambd = "Scalar")
nd_args <- c("grad_out", "self", "lambd")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hardshrink_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hardtanh <- function(self, min_val = -1, max_val = 1) {
  
args <- rlang::env_get_list(nms = c("self", "min_val", "max_val"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", min_val = "Scalar", max_val = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hardtanh', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hardtanh_ <- function(self, min_val = -1, max_val = 1) {
  
args <- rlang::env_get_list(nms = c("self", "min_val", "max_val"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", min_val = "Scalar", max_val = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hardtanh_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hardtanh_backward <- function(grad_output, self, min_val, max_val) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "min_val", "max_val"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", min_val = "Scalar", 
    max_val = "Scalar")
nd_args <- c("grad_output", "self", "min_val", "max_val")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hardtanh_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hardtanh_backward_out <- function(grad_input, grad_output, self, min_val, max_val) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "min_val", "max_val"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    min_val = "Scalar", max_val = "Scalar")
nd_args <- c("grad_input", "grad_output", "self", "min_val", "max_val")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hardtanh_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hardtanh_out <- function(out, self, min_val = -1, max_val = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "min_val", "max_val"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", min_val = "Scalar", max_val = "Scalar")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hardtanh_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hinge_embedding_loss <- function(self, target, margin = 1.000000, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "margin", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", margin = "double", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hinge_embedding_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_histc <- function(self, bins = 100, min = 0, max = 0) {
  
args <- rlang::env_get_list(nms = c("self", "bins", "min", "max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", bins = "int64_t", min = "Scalar", max = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('histc', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_histc_out <- function(out, self, bins = 100, min = 0, max = 0) {
  
args <- rlang::env_get_list(nms = c("out", "self", "bins", "min", "max"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", bins = "int64_t", min = "Scalar", 
    max = "Scalar")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('histc_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hspmm <- function(mat1, mat2) {
  
args <- rlang::env_get_list(nms = c("mat1", "mat2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(mat1 = "Tensor", mat2 = "Tensor")
nd_args <- c("mat1", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hspmm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_hspmm_out <- function(out, mat1, mat2) {
  
args <- rlang::env_get_list(nms = c("out", "mat1", "mat2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", mat1 = "Tensor", mat2 = "Tensor")
nd_args <- c("out", "mat1", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('hspmm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ifft <- function(self, signal_ndim, normalized = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "signal_ndim", "normalized"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", signal_ndim = "int64_t", normalized = "bool")
nd_args <- c("self", "signal_ndim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ifft', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_im2col <- function(self, kernel_size, dilation, padding, stride) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "dilation", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", dilation = "IntArrayRef", 
    padding = "IntArrayRef", stride = "IntArrayRef")
nd_args <- c("self", "kernel_size", "dilation", "padding", "stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('im2col', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_im2col_backward <- function(grad_output, input_size, kernel_size, dilation, padding, stride) {
  
args <- rlang::env_get_list(nms = c("grad_output", "input_size", "kernel_size", "dilation", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", input_size = "IntArrayRef", kernel_size = "IntArrayRef", 
    dilation = "IntArrayRef", padding = "IntArrayRef", stride = "IntArrayRef")
nd_args <- c("grad_output", "input_size", "kernel_size", "dilation", "padding", 
"stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('im2col_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_im2col_backward_out <- function(grad_input, grad_output, input_size, kernel_size, dilation, padding, stride) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "input_size", "kernel_size", "dilation", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", input_size = "IntArrayRef", 
    kernel_size = "IntArrayRef", dilation = "IntArrayRef", padding = "IntArrayRef", 
    stride = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "input_size", "kernel_size", "dilation", 
"padding", "stride")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('im2col_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_im2col_out <- function(out, self, kernel_size, dilation, padding, stride) {
  
args <- rlang::env_get_list(nms = c("out", "self", "kernel_size", "dilation", "padding", "stride"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    dilation = "IntArrayRef", padding = "IntArrayRef", stride = "IntArrayRef")
nd_args <- c("out", "self", "kernel_size", "dilation", "padding", "stride"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('im2col_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_imag <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('imag', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_imag_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('imag_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_index <- function(self, indices) {
  
args <- rlang::env_get_list(nms = c("self", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", indices = "TensorList")
nd_args <- c("self", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('index', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_index_add <- function(self, dim, index, source) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index", "source"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), index = "Tensor", 
    source = "Tensor")
nd_args <- c("self", "dim", "index", "source")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('index_add', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_index_copy <- function(self, dim, index, source) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index", "source"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), index = "Tensor", 
    source = "Tensor")
nd_args <- c("self", "dim", "index", "source")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('index_copy', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_index_fill <- function(self, dim, index, value) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), index = "Tensor", 
    value = c("Scalar", "Tensor"))
nd_args <- c("self", "dim", "index", "value")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('index_fill', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_index_put <- function(self, indices, values, accumulate = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "indices", "values", "accumulate"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", indices = "TensorList", values = "Tensor", 
    accumulate = "bool")
nd_args <- c("self", "indices", "values")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('index_put', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_index_put_ <- function(self, indices, values, accumulate = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "indices", "values", "accumulate"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", indices = "TensorList", values = "Tensor", 
    accumulate = "bool")
nd_args <- c("self", "indices", "values")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('index_put_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_index_select <- function(self, dim, index) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), index = "Tensor")
nd_args <- c("self", "dim", "index")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('index_select', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_index_select_out <- function(out, self, dim, index) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("int64_t", "Dimname"
), index = "Tensor")
nd_args <- c("out", "self", "dim", "index")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('index_select_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_instance_norm <- function(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "running_mean", "running_var", "use_input_stats", "momentum", "eps", "cudnn_enabled"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", running_mean = "Tensor", 
    running_var = "Tensor", use_input_stats = "bool", momentum = "double", 
    eps = "double", cudnn_enabled = "bool")
nd_args <- c("input", "weight", "bias", "running_mean", "running_var", "use_input_stats", 
"momentum", "eps", "cudnn_enabled")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('instance_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_int_repr <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('int_repr', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_inverse <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('inverse', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_inverse_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('inverse_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_irfft <- function(self, signal_ndim, normalized = FALSE, onesided = TRUE, signal_sizes = list()) {
  
args <- rlang::env_get_list(nms = c("self", "signal_ndim", "normalized", "onesided", "signal_sizes"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", signal_ndim = "int64_t", normalized = "bool", 
    onesided = "bool", signal_sizes = "IntArrayRef")
nd_args <- c("self", "signal_ndim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('irfft', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_is_complex <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('is_complex', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_is_distributed <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('is_distributed', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_is_floating_point <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('is_floating_point', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_is_nonzero <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('is_nonzero', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_is_same_size <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor")
nd_args <- c("self", "other")
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('is_same_size', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_is_signed <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('bool')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('is_signed', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_isclose <- function(self, other, rtol = 0.000010, atol = 0.000000, equal_nan = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "other", "rtol", "atol", "equal_nan"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor", rtol = "double", atol = "double", 
    equal_nan = "bool")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('isclose', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_isfinite <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('isfinite', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_isnan <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('isnan', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_kl_div <- function(self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('kl_div', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_kl_div_backward <- function(grad_output, self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    reduction = "int64_t")
nd_args <- c("grad_output", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('kl_div_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_kthvalue <- function(self, k, dim = -1, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "k", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", k = "int64_t", dim = c("int64_t", "Dimname"
), keepdim = "bool")
nd_args <- c("self", "k")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('kthvalue', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_kthvalue_out <- function(values, indices, self, k, dim = -1, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("values", "indices", "self", "k", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(values = "Tensor", indices = "Tensor", self = "Tensor", 
    k = "int64_t", dim = c("int64_t", "Dimname"), keepdim = "bool")
nd_args <- c("values", "indices", "self", "k")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('kthvalue_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_l1_loss <- function(self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('l1_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_l1_loss_backward <- function(grad_output, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    reduction = "int64_t")
nd_args <- c("grad_output", "self", "target", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('l1_loss_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_l1_loss_backward_out <- function(grad_input, grad_output, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", reduction = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "target", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('l1_loss_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_l1_loss_out <- function(out, self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('l1_loss_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_layer_norm <- function(input, normalized_shape, weight = list(), bias = list(), eps = 0.000010, cudnn_enable = TRUE) {
  
args <- rlang::env_get_list(nms = c("input", "normalized_shape", "weight", "bias", "eps", "cudnn_enable"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", normalized_shape = "IntArrayRef", weight = "Tensor", 
    bias = "Tensor", eps = "double", cudnn_enable = "bool")
nd_args <- c("input", "normalized_shape")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('layer_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_le <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('le', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_le_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Scalar", "Tensor"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('le_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_leaky_relu <- function(self, negative_slope = 0.010000) {
  
args <- rlang::env_get_list(nms = c("self", "negative_slope"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", negative_slope = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('leaky_relu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_leaky_relu_ <- function(self, negative_slope = 0.010000) {
  
args <- rlang::env_get_list(nms = c("self", "negative_slope"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", negative_slope = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('leaky_relu_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_leaky_relu_backward <- function(grad_output, self, negative_slope) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "negative_slope"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", negative_slope = "Scalar")
nd_args <- c("grad_output", "self", "negative_slope")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('leaky_relu_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_leaky_relu_backward_out <- function(grad_input, grad_output, self, negative_slope) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "negative_slope"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    negative_slope = "Scalar")
nd_args <- c("grad_input", "grad_output", "self", "negative_slope")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('leaky_relu_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_leaky_relu_out <- function(out, self, negative_slope = 0.010000) {
  
args <- rlang::env_get_list(nms = c("out", "self", "negative_slope"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", negative_slope = "Scalar")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('leaky_relu_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lerp <- function(self, end, weight) {
  
args <- rlang::env_get_list(nms = c("self", "end", "weight"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", end = "Tensor", weight = c("Scalar", "Tensor"
))
nd_args <- c("self", "end", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lerp', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lerp_out <- function(out, self, end, weight) {
  
args <- rlang::env_get_list(nms = c("out", "self", "end", "weight"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", end = "Tensor", weight = c("Scalar", 
"Tensor"))
nd_args <- c("out", "self", "end", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lerp_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lgamma <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lgamma', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lgamma_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lgamma_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_linear <- function(input, weight, bias = list()) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor")
nd_args <- c("input", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('linear', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_linspace <- function(start, end, steps = 100, options = list()) {
  
args <- rlang::env_get_list(nms = c("start", "end", "steps", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(start = "Scalar", end = "Scalar", steps = "int64_t", options = "TensorOptions")
nd_args <- c("start", "end")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('linspace', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_linspace_out <- function(out, start, end, steps = 100) {
  
args <- rlang::env_get_list(nms = c("out", "start", "end", "steps"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", start = "Scalar", end = "Scalar", steps = "int64_t")
nd_args <- c("out", "start", "end")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('linspace_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_sigmoid <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_sigmoid', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_sigmoid_backward <- function(grad_output, self, buffer) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "buffer"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", buffer = "Tensor")
nd_args <- c("grad_output", "self", "buffer")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_sigmoid_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_sigmoid_backward_out <- function(grad_input, grad_output, self, buffer) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "buffer"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    buffer = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "buffer")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_sigmoid_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_sigmoid_forward <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_sigmoid_forward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_sigmoid_forward_out <- function(output, buffer, self) {
  
args <- rlang::env_get_list(nms = c("output", "buffer", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(output = "Tensor", buffer = "Tensor", self = "Tensor")
nd_args <- c("output", "buffer", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_sigmoid_forward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_sigmoid_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_sigmoid_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log_softmax <- function(self, dim, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), dtype = "ScalarType")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log_softmax', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log10 <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log10', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log10_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log10_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log10_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log10_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log1p <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log1p', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log1p_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log1p_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log1p_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log1p_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log2 <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log2', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log2_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log2_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_log2_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('log2_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logdet <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logdet', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logical_not <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logical_not', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logical_not_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logical_not_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logical_xor <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logical_xor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logical_xor_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = "Tensor")
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logical_xor_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logspace <- function(start, end, steps = 100, base = 10.000000, options = list()) {
  
args <- rlang::env_get_list(nms = c("start", "end", "steps", "base", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(start = "Scalar", end = "Scalar", steps = "int64_t", base = "double", 
    options = "TensorOptions")
nd_args <- c("start", "end")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logspace', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logspace_out <- function(out, start, end, steps = 100, base = 10.000000) {
  
args <- rlang::env_get_list(nms = c("out", "start", "end", "steps", "base"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", start = "Scalar", end = "Scalar", steps = "int64_t", 
    base = "double")
nd_args <- c("out", "start", "end")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logspace_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logsumexp <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logsumexp', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_logsumexp_out <- function(out, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("IntArrayRef", 
"DimnameList"), keepdim = "bool")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('logsumexp_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lstm <- function(data, input, batch_sizes, hx, params, has_biases, num_layers, dropout, train, batch_first, bidirectional) {
  
args <- rlang::env_get_list(nms = c("data", "input", "batch_sizes", "hx", "params", "has_biases", "num_layers", "dropout", "train", "batch_first", "bidirectional"))
args <- Filter(Negate(is.name), args)
expected_types <- list(data = "Tensor", input = "Tensor", batch_sizes = "Tensor", 
    hx = "TensorList", params = "TensorList", has_biases = "bool", 
    num_layers = "int64_t", dropout = "double", train = "bool", 
    batch_first = "bool", bidirectional = "bool")
nd_args <- c("data", "input", "batch_sizes", "hx", "params", "has_biases", 
"num_layers", "dropout", "train", "batch_first", "bidirectional"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lstm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lstm_cell <- function(input, hx, w_ih, w_hh, b_ih = list(), b_hh = list()) {
  
args <- rlang::env_get_list(nms = c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", hx = "TensorList", w_ih = "Tensor", w_hh = "Tensor", 
    b_ih = "Tensor", b_hh = "Tensor")
nd_args <- c("input", "hx", "w_ih", "w_hh")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lstm_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lstsq <- function(self, A) {
  
args <- rlang::env_get_list(nms = c("self", "A"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", A = "Tensor")
nd_args <- c("self", "A")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lstsq', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lstsq_out <- function(X, qr, self, A) {
  
args <- rlang::env_get_list(nms = c("X", "qr", "self", "A"))
args <- Filter(Negate(is.name), args)
expected_types <- list(X = "Tensor", qr = "Tensor", self = "Tensor", A = "Tensor")
nd_args <- c("X", "qr", "self", "A")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lstsq_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lt <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lt', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lt_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Scalar", "Tensor"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lt_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lu_solve <- function(self, LU_data, LU_pivots) {
  
args <- rlang::env_get_list(nms = c("self", "LU_data", "LU_pivots"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", LU_data = "Tensor", LU_pivots = "Tensor")
nd_args <- c("self", "LU_data", "LU_pivots")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lu_solve', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_lu_solve_out <- function(out, self, LU_data, LU_pivots) {
  
args <- rlang::env_get_list(nms = c("out", "self", "LU_data", "LU_pivots"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", LU_data = "Tensor", LU_pivots = "Tensor")
nd_args <- c("out", "self", "LU_data", "LU_pivots")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('lu_solve_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_margin_ranking_loss <- function(input1, input2, target, margin = 0.000000, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("input1", "input2", "target", "margin", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input1 = "Tensor", input2 = "Tensor", target = "Tensor", 
    margin = "double", reduction = "int64_t")
nd_args <- c("input1", "input2", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('margin_ranking_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_masked_fill <- function(self, mask, value) {
  
args <- rlang::env_get_list(nms = c("self", "mask", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mask = "Tensor", value = c("Scalar", "Tensor"
))
nd_args <- c("self", "mask", "value")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('masked_fill', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_masked_scatter <- function(self, mask, source) {
  
args <- rlang::env_get_list(nms = c("self", "mask", "source"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mask = "Tensor", source = "Tensor")
nd_args <- c("self", "mask", "source")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('masked_scatter', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_masked_select <- function(self, mask) {
  
args <- rlang::env_get_list(nms = c("self", "mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mask = "Tensor")
nd_args <- c("self", "mask")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('masked_select', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_masked_select_out <- function(out, self, mask) {
  
args <- rlang::env_get_list(nms = c("out", "self", "mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", mask = "Tensor")
nd_args <- c("out", "self", "mask")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('masked_select_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_matmul <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('matmul', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_matmul_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = "Tensor")
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('matmul_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_matrix_power <- function(self, n) {
  
args <- rlang::env_get_list(nms = c("self", "n"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", n = "int64_t")
nd_args <- c("self", "n")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('matrix_power', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_matrix_rank <- function(self, tol, symmetric = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "tol", "symmetric"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", tol = "double", symmetric = "bool")
nd_args <- c("self", "tol")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('matrix_rank', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max <- function(self, dim, other, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "other", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), other = "Tensor", 
    keepdim = "bool")
nd_args <- c("self", "dim", "other")
return_types <- c('TensorList', 'Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_out <- function(max, out, max_values, other, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("max", "out", "max_values", "other", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(max = "Tensor", out = "Tensor", max_values = "Tensor", other = "Tensor", 
    self = "Tensor", dim = c("int64_t", "Dimname"), keepdim = "bool")
nd_args <- c("max", "out", "max_values", "other", "self", "dim")
return_types <- c('TensorList', 'Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool1d <- function(self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", ceil_mode = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool1d_with_indices <- function(self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", ceil_mode = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool1d_with_indices', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool2d <- function(self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", ceil_mode = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool2d_with_indices <- function(self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", ceil_mode = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool2d_with_indices', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool2d_with_indices_backward <- function(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", dilation = "IntArrayRef", 
    ceil_mode = "bool", indices = "Tensor")
nd_args <- c("grad_output", "self", "kernel_size", "stride", "padding", 
"dilation", "ceil_mode", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool2d_with_indices_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool2d_with_indices_backward_out <- function(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef", ceil_mode = "bool", indices = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "kernel_size", "stride", 
"padding", "dilation", "ceil_mode", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool2d_with_indices_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool2d_with_indices_out <- function(out, indices, self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "indices", "self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", indices = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", dilation = "IntArrayRef", 
    ceil_mode = "bool")
nd_args <- c("out", "indices", "self", "kernel_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool2d_with_indices_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool3d <- function(self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", ceil_mode = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool3d_with_indices <- function(self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", ceil_mode = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool3d_with_indices', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool3d_with_indices_backward <- function(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", dilation = "IntArrayRef", 
    ceil_mode = "bool", indices = "Tensor")
nd_args <- c("grad_output", "self", "kernel_size", "stride", "padding", 
"dilation", "ceil_mode", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool3d_with_indices_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool3d_with_indices_backward_out <- function(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "indices"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef", ceil_mode = "bool", indices = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "kernel_size", "stride", 
"padding", "dilation", "ceil_mode", "indices")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool3d_with_indices_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_pool3d_with_indices_out <- function(out, indices, self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "indices", "self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", indices = "Tensor", self = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", dilation = "IntArrayRef", 
    ceil_mode = "bool")
nd_args <- c("out", "indices", "self", "kernel_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_pool3d_with_indices_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_unpool2d <- function(self, indices, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "indices", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", indices = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "indices", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_unpool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_unpool2d_backward <- function(grad_output, self, indices, output_size) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "indices", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", indices = "Tensor", 
    output_size = "IntArrayRef")
nd_args <- c("grad_output", "self", "indices", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_unpool2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_unpool2d_backward_out <- function(grad_input, grad_output, self, indices, output_size) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "indices", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    indices = "Tensor", output_size = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "self", "indices", "output_size"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_unpool2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_unpool2d_out <- function(out, self, indices, output_size) {
  
args <- rlang::env_get_list(nms = c("out", "self", "indices", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", indices = "Tensor", output_size = "IntArrayRef")
nd_args <- c("out", "self", "indices", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_unpool2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_unpool3d <- function(self, indices, output_size, stride, padding) {
  
args <- rlang::env_get_list(nms = c("self", "indices", "output_size", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", indices = "Tensor", output_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("self", "indices", "output_size", "stride", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_unpool3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_unpool3d_backward <- function(grad_output, self, indices, output_size, stride, padding) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "indices", "output_size", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", indices = "Tensor", 
    output_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("grad_output", "self", "indices", "output_size", "stride", 
"padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_unpool3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_unpool3d_backward_out <- function(grad_input, grad_output, self, indices, output_size, stride, padding) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "indices", "output_size", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    indices = "Tensor", output_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "self", "indices", "output_size", 
"stride", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_unpool3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_unpool3d_out <- function(out, self, indices, output_size, stride, padding) {
  
args <- rlang::env_get_list(nms = c("out", "self", "indices", "output_size", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", indices = "Tensor", output_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("out", "self", "indices", "output_size", "stride", "padding"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_unpool3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_max_values <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('max_values', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mean <- function(self, dim, keepdim = FALSE, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    keepdim = "bool", dtype = "ScalarType")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mean', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mean_out <- function(out, self, dim, keepdim = FALSE, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "keepdim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("IntArrayRef", 
"DimnameList"), keepdim = "bool", dtype = "ScalarType")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mean_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_median <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('TensorList', 'Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('median', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_median_out <- function(values, indices, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("values", "indices", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(values = "Tensor", indices = "Tensor", self = "Tensor", 
    dim = c("int64_t", "Dimname"), keepdim = "bool")
nd_args <- c("values", "indices", "self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('median_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_meshgrid <- function(tensors) {
  
args <- rlang::env_get_list(nms = c("tensors"))
args <- Filter(Negate(is.name), args)
expected_types <- list(tensors = "TensorList")
nd_args <- "tensors"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('meshgrid', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_min <- function(self, dim, other, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "other", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), other = "Tensor", 
    keepdim = "bool")
nd_args <- c("self", "dim", "other")
return_types <- c('TensorList', 'Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('min', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_min_out <- function(min, out, min_indices, other, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("min", "out", "min_indices", "other", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(min = "Tensor", out = "Tensor", min_indices = "Tensor", 
    other = "Tensor", self = "Tensor", dim = c("int64_t", "Dimname"
    ), keepdim = "bool")
nd_args <- c("min", "out", "min_indices", "other", "self", "dim")
return_types <- c('TensorList', 'Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('min_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_min_values <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('min_values', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_batch_norm <- function(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "running_mean", "running_var", "training", "exponential_average_factor", "epsilon"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", running_mean = "Tensor", 
    running_var = "Tensor", training = "bool", exponential_average_factor = "double", 
    epsilon = "double")
nd_args <- c("input", "weight", "bias", "running_mean", "running_var", "training", 
"exponential_average_factor", "epsilon")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_batch_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_batch_norm_backward <- function(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon) {
  
args <- rlang::env_get_list(nms = c("input", "grad_output", "weight", "running_mean", "running_var", "save_mean", "save_var", "epsilon"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    running_mean = "Tensor", running_var = "Tensor", save_mean = "Tensor", 
    save_var = "Tensor", epsilon = "double")
nd_args <- c("input", "grad_output", "weight", "running_mean", "running_var", 
"save_mean", "save_var", "epsilon")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_batch_norm_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution <- function(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "bias", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", bias = "Tensor", padding = "IntArrayRef", 
    stride = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t", 
    benchmark = "bool", deterministic = "bool")
nd_args <- c("self", "weight", "bias", "padding", "stride", "dilation", 
"groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution_backward <- function(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask) {
  
args <- rlang::env_get_list(nms = c("self", "grad_output", "weight", "padding", "stride", "dilation", "groups", "benchmark", "deterministic", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool", 
    output_mask = "std::array<bool,3>")
nd_args <- c("self", "grad_output", "weight", "padding", "stride", "dilation", 
"groups", "benchmark", "deterministic", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution_backward_bias <- function(grad_output) {
  
args <- rlang::env_get_list(nms = c("grad_output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor")
nd_args <- "grad_output"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution_backward_bias', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution_backward_input <- function(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("self_size", "grad_output", "weight", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self_size = "IntArrayRef", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("self_size", "grad_output", "weight", "padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution_backward_input', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution_backward_weight <- function(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("weight_size", "grad_output", "self", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight_size = "IntArrayRef", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("weight_size", "grad_output", "self", "padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution_backward_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution_transpose <- function(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "bias", "padding", "output_padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", bias = "Tensor", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("self", "weight", "bias", "padding", "output_padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution_transpose', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution_transpose_backward <- function(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask) {
  
args <- rlang::env_get_list(nms = c("self", "grad_output", "weight", "padding", "output_padding", "stride", "dilation", "groups", "benchmark", "deterministic", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", output_padding = "IntArrayRef", 
    stride = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t", 
    benchmark = "bool", deterministic = "bool", output_mask = "std::array<bool,3>")
nd_args <- c("self", "grad_output", "weight", "padding", "output_padding", 
"stride", "dilation", "groups", "benchmark", "deterministic", 
"output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution_transpose_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution_transpose_backward_input <- function(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("grad_output", "weight", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", weight = "Tensor", padding = "IntArrayRef", 
    stride = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t", 
    benchmark = "bool", deterministic = "bool")
nd_args <- c("grad_output", "weight", "padding", "stride", "dilation", "groups", 
"benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution_transpose_backward_input', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_convolution_transpose_backward_weight <- function(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("weight_size", "grad_output", "self", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight_size = "IntArrayRef", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("weight_size", "grad_output", "self", "padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_convolution_transpose_backward_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_depthwise_convolution <- function(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "bias", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", bias = "Tensor", padding = "IntArrayRef", 
    stride = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t", 
    benchmark = "bool", deterministic = "bool")
nd_args <- c("self", "weight", "bias", "padding", "stride", "dilation", 
"groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_depthwise_convolution', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_depthwise_convolution_backward <- function(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask) {
  
args <- rlang::env_get_list(nms = c("self", "grad_output", "weight", "padding", "stride", "dilation", "groups", "benchmark", "deterministic", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool", 
    output_mask = "std::array<bool,3>")
nd_args <- c("self", "grad_output", "weight", "padding", "stride", "dilation", 
"groups", "benchmark", "deterministic", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_depthwise_convolution_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_depthwise_convolution_backward_input <- function(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("self_size", "grad_output", "weight", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self_size = "IntArrayRef", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("self_size", "grad_output", "weight", "padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_depthwise_convolution_backward_input', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_depthwise_convolution_backward_weight <- function(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic) {
  
args <- rlang::env_get_list(nms = c("weight_size", "grad_output", "self", "padding", "stride", "dilation", "groups", "benchmark", "deterministic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight_size = "IntArrayRef", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", benchmark = "bool", deterministic = "bool")
nd_args <- c("weight_size", "grad_output", "self", "padding", "stride", 
"dilation", "groups", "benchmark", "deterministic")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_depthwise_convolution_backward_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_rnn <- function(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "weight_stride0", "hx", "cx", "mode", "hidden_size", "num_layers", "batch_first", "dropout", "train", "bidirectional", "batch_sizes", "dropout_state"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "TensorList", weight_stride0 = "int64_t", 
    hx = "Tensor", cx = "Tensor", mode = "int64_t", hidden_size = "int64_t", 
    num_layers = "int64_t", batch_first = "bool", dropout = "double", 
    train = "bool", bidirectional = "bool", batch_sizes = "IntArrayRef", 
    dropout_state = "Tensor")
nd_args <- c("input", "weight", "weight_stride0", "hx", "cx", "mode", "hidden_size", 
"num_layers", "batch_first", "dropout", "train", "bidirectional", 
"batch_sizes", "dropout_state")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_rnn', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_miopen_rnn_backward <- function(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "weight_stride0", "weight_buf", "hx", "cx", "output", "grad_output", "grad_hy", "grad_cy", "mode", "hidden_size", "num_layers", "batch_first", "dropout", "train", "bidirectional", "batch_sizes", "dropout_state", "reserve", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "TensorList", weight_stride0 = "int64_t", 
    weight_buf = "Tensor", hx = "Tensor", cx = "Tensor", output = "Tensor", 
    grad_output = "Tensor", grad_hy = "Tensor", grad_cy = "Tensor", 
    mode = "int64_t", hidden_size = "int64_t", num_layers = "int64_t", 
    batch_first = "bool", dropout = "double", train = "bool", 
    bidirectional = "bool", batch_sizes = "IntArrayRef", dropout_state = "Tensor", 
    reserve = "Tensor", output_mask = "std::array<bool,4>")
nd_args <- c("input", "weight", "weight_stride0", "weight_buf", "hx", "cx", 
"output", "grad_output", "grad_hy", "grad_cy", "mode", "hidden_size", 
"num_layers", "batch_first", "dropout", "train", "bidirectional", 
"batch_sizes", "dropout_state", "reserve", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('miopen_rnn_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mkldnn_adaptive_avg_pool2d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mkldnn_adaptive_avg_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mkldnn_convolution <- function(self, weight, bias, padding, stride, dilation, groups) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "bias", "padding", "stride", "dilation", "groups"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", bias = "Tensor", padding = "IntArrayRef", 
    stride = "IntArrayRef", dilation = "IntArrayRef", groups = "int64_t")
nd_args <- c("self", "weight", "bias", "padding", "stride", "dilation", 
"groups")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mkldnn_convolution', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mkldnn_convolution_backward <- function(self, grad_output, weight, padding, stride, dilation, groups, output_mask) {
  
args <- rlang::env_get_list(nms = c("self", "grad_output", "weight", "padding", "stride", "dilation", "groups", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", output_mask = "std::array<bool,3>")
nd_args <- c("self", "grad_output", "weight", "padding", "stride", "dilation", 
"groups", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mkldnn_convolution_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mkldnn_convolution_backward_input <- function(self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined) {
  
args <- rlang::env_get_list(nms = c("self_size", "grad_output", "weight", "padding", "stride", "dilation", "groups", "bias_defined"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self_size = "IntArrayRef", grad_output = "Tensor", weight = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", bias_defined = "bool")
nd_args <- c("self_size", "grad_output", "weight", "padding", "stride", 
"dilation", "groups", "bias_defined")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mkldnn_convolution_backward_input', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mkldnn_convolution_backward_weights <- function(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined) {
  
args <- rlang::env_get_list(nms = c("weight_size", "grad_output", "self", "padding", "stride", "dilation", "groups", "bias_defined"))
args <- Filter(Negate(is.name), args)
expected_types <- list(weight_size = "IntArrayRef", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef", stride = "IntArrayRef", dilation = "IntArrayRef", 
    groups = "int64_t", bias_defined = "bool")
nd_args <- c("weight_size", "grad_output", "self", "padding", "stride", 
"dilation", "groups", "bias_defined")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mkldnn_convolution_backward_weights', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mkldnn_linear <- function(input, weight, bias = list()) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor")
nd_args <- c("input", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mkldnn_linear', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mkldnn_max_pool2d <- function(self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", ceil_mode = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mkldnn_max_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mkldnn_reorder_conv2d_weight <- function(self, padding = 0, stride = 1, dilation = 1, groups = 1) {
  
args <- rlang::env_get_list(nms = c("self", "padding", "stride", "dilation", "groups"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", padding = "IntArrayRef", stride = "IntArrayRef", 
    dilation = "IntArrayRef", groups = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mkldnn_reorder_conv2d_weight', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mm <- function(self, mat2) {
  
args <- rlang::env_get_list(nms = c("self", "mat2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mat2 = "Tensor")
nd_args <- c("self", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mm_out <- function(out, self, mat2) {
  
args <- rlang::env_get_list(nms = c("out", "self", "mat2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", mat2 = "Tensor")
nd_args <- c("out", "self", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mode <- function(self, dim = -1, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), keepdim = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mode', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mode_out <- function(values, indices, self, dim = -1, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("values", "indices", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(values = "Tensor", indices = "Tensor", self = "Tensor", 
    dim = c("int64_t", "Dimname"), keepdim = "bool")
nd_args <- c("values", "indices", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mode_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mse_loss <- function(self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mse_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mse_loss_backward <- function(grad_output, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    reduction = "int64_t")
nd_args <- c("grad_output", "self", "target", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mse_loss_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mse_loss_backward_out <- function(grad_input, grad_output, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", reduction = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "target", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mse_loss_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mse_loss_out <- function(out, self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mse_loss_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mul <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Tensor", "Scalar"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mul', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mul_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = "Tensor")
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mul_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multi_margin_loss <- function(self, target, p = 1, margin = 1, weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "p", "margin", "weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", p = "Scalar", margin = "Scalar", 
    weight = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multi_margin_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multi_margin_loss_backward <- function(grad_output, self, target, p, margin, weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "p", "margin", "weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    p = "Scalar", margin = "Scalar", weight = "Tensor", reduction = "int64_t")
nd_args <- c("grad_output", "self", "target", "p", "margin")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multi_margin_loss_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multi_margin_loss_backward_out <- function(grad_input, grad_output, self, target, p, margin, weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "p", "margin", "weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", p = "Scalar", margin = "Scalar", weight = "Tensor", 
    reduction = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "target", "p", "margin"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multi_margin_loss_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multi_margin_loss_out <- function(out, self, target, p = 1, margin = 1, weight = list(), reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "p", "margin", "weight", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", p = "Scalar", 
    margin = "Scalar", weight = "Tensor", reduction = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multi_margin_loss_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multilabel_margin_loss <- function(self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multilabel_margin_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multilabel_margin_loss_backward <- function(grad_output, self, target, reduction, is_target) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "reduction", "is_target"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    reduction = "int64_t", is_target = "Tensor")
nd_args <- c("grad_output", "self", "target", "reduction", "is_target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multilabel_margin_loss_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multilabel_margin_loss_backward_out <- function(grad_input, grad_output, self, target, reduction, is_target) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "reduction", "is_target"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", reduction = "int64_t", is_target = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "target", "reduction", 
"is_target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multilabel_margin_loss_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multilabel_margin_loss_forward <- function(self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target", "reduction")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multilabel_margin_loss_forward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multilabel_margin_loss_forward_out <- function(output, is_target, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("output", "is_target", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(output = "Tensor", is_target = "Tensor", self = "Tensor", 
    target = "Tensor", reduction = "int64_t")
nd_args <- c("output", "is_target", "self", "target", "reduction")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multilabel_margin_loss_forward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multilabel_margin_loss_out <- function(out, self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multilabel_margin_loss_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multinomial <- function(self, num_samples, replacement = FALSE, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "num_samples", "replacement", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", num_samples = "int64_t", replacement = "bool", 
    generator = "Generator *")
nd_args <- c("self", "num_samples")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multinomial', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_multinomial_out <- function(out, self, num_samples, replacement = FALSE, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "num_samples", "replacement", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", num_samples = "int64_t", 
    replacement = "bool", generator = "Generator *")
nd_args <- c("out", "self", "num_samples")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('multinomial_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mv <- function(self, vec) {
  
args <- rlang::env_get_list(nms = c("self", "vec"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", vec = "Tensor")
nd_args <- c("self", "vec")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mv', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mv_out <- function(out, self, vec) {
  
args <- rlang::env_get_list(nms = c("out", "self", "vec"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", vec = "Tensor")
nd_args <- c("out", "self", "vec")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mv_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_mvlgamma <- function(self, p) {
  
args <- rlang::env_get_list(nms = c("self", "p"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "int64_t")
nd_args <- c("self", "p")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('mvlgamma', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_narrow <- function(self, dim, start, length) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "start", "length"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", start = "int64_t", length = "int64_t")
nd_args <- c("self", "dim", "start", "length")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('narrow', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_native_batch_norm <- function(input, weight, bias, running_mean, running_var, training, momentum, eps) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "running_mean", "running_var", "training", "momentum", "eps"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", running_mean = "Tensor", 
    running_var = "Tensor", training = "bool", momentum = "double", 
    eps = "double")
nd_args <- c("input", "weight", "bias", "running_mean", "running_var", "training", 
"momentum", "eps")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('native_batch_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_native_batch_norm_backward <- function(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_out", "input", "weight", "running_mean", "running_var", "save_mean", "save_invstd", "train", "eps", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_out = "Tensor", input = "Tensor", weight = "Tensor", 
    running_mean = "Tensor", running_var = "Tensor", save_mean = "Tensor", 
    save_invstd = "Tensor", train = "bool", eps = "double", output_mask = "std::array<bool,3>")
nd_args <- c("grad_out", "input", "weight", "running_mean", "running_var", 
"save_mean", "save_invstd", "train", "eps", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('native_batch_norm_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_native_layer_norm <- function(input, weight, bias, M, False, eps) {
  
args <- rlang::env_get_list(nms = c("input", "weight", "bias", "M", "False", "eps"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", weight = "Tensor", bias = "Tensor", M = "int64_t", 
    False = "int64_t", eps = "double")
nd_args <- c("input", "weight", "bias", "M", "False", "eps")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('native_layer_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_native_layer_norm_backward <- function(grad_out, input, mean, rstd, weight, M, False, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_out", "input", "mean", "rstd", "weight", "M", "False", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_out = "Tensor", input = "Tensor", mean = "Tensor", 
    rstd = "Tensor", weight = "Tensor", M = "int64_t", False = "int64_t", 
    output_mask = "std::array<bool,3>")
nd_args <- c("grad_out", "input", "mean", "rstd", "weight", "M", "False", 
"output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('native_layer_norm_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_native_norm <- function(self, p = 2) {
  
args <- rlang::env_get_list(nms = c("self", "p"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('native_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ne <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ne', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ne_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Scalar", "Tensor"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ne_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_neg <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('neg', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_neg_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('neg_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_neg_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('neg_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss <- function(self, target, weight = list(), reduction = torch_reduction_mean(), ignore_index = -100) {
  
args <- rlang::env_get_list(nms = c("self", "target", "weight", "reduction", "ignore_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", weight = "Tensor", reduction = "int64_t", 
    ignore_index = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss_backward <- function(grad_output, self, target, weight, reduction, ignore_index, total_weight) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "weight", "reduction", "ignore_index", "total_weight"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    weight = "Tensor", reduction = "int64_t", ignore_index = "int64_t", 
    total_weight = "Tensor")
nd_args <- c("grad_output", "self", "target", "weight", "reduction", "ignore_index", 
"total_weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss_backward_out <- function(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "weight", "reduction", "ignore_index", "total_weight"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", weight = "Tensor", reduction = "int64_t", 
    ignore_index = "int64_t", total_weight = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "target", "weight", "reduction", 
"ignore_index", "total_weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss_forward <- function(self, target, weight, reduction, ignore_index) {
  
args <- rlang::env_get_list(nms = c("self", "target", "weight", "reduction", "ignore_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", weight = "Tensor", reduction = "int64_t", 
    ignore_index = "int64_t")
nd_args <- c("self", "target", "weight", "reduction", "ignore_index")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss_forward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss_forward_out <- function(output, total_weight, self, target, weight, reduction, ignore_index) {
  
args <- rlang::env_get_list(nms = c("output", "total_weight", "self", "target", "weight", "reduction", "ignore_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(output = "Tensor", total_weight = "Tensor", self = "Tensor", 
    target = "Tensor", weight = "Tensor", reduction = "int64_t", 
    ignore_index = "int64_t")
nd_args <- c("output", "total_weight", "self", "target", "weight", "reduction", 
"ignore_index")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss_forward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss_out <- function(out, self, target, weight = list(), reduction = torch_reduction_mean(), ignore_index = -100) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "weight", "reduction", "ignore_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", weight = "Tensor", 
    reduction = "int64_t", ignore_index = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss2d <- function(self, target, weight = list(), reduction = torch_reduction_mean(), ignore_index = -100) {
  
args <- rlang::env_get_list(nms = c("self", "target", "weight", "reduction", "ignore_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", weight = "Tensor", reduction = "int64_t", 
    ignore_index = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss2d_backward <- function(grad_output, self, target, weight, reduction, ignore_index, total_weight) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "weight", "reduction", "ignore_index", "total_weight"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    weight = "Tensor", reduction = "int64_t", ignore_index = "int64_t", 
    total_weight = "Tensor")
nd_args <- c("grad_output", "self", "target", "weight", "reduction", "ignore_index", 
"total_weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss2d_backward_out <- function(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "weight", "reduction", "ignore_index", "total_weight"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", weight = "Tensor", reduction = "int64_t", 
    ignore_index = "int64_t", total_weight = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "target", "weight", "reduction", 
"ignore_index", "total_weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss2d_forward <- function(self, target, weight, reduction, ignore_index) {
  
args <- rlang::env_get_list(nms = c("self", "target", "weight", "reduction", "ignore_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", weight = "Tensor", reduction = "int64_t", 
    ignore_index = "int64_t")
nd_args <- c("self", "target", "weight", "reduction", "ignore_index")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss2d_forward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss2d_forward_out <- function(output, total_weight, self, target, weight, reduction, ignore_index) {
  
args <- rlang::env_get_list(nms = c("output", "total_weight", "self", "target", "weight", "reduction", "ignore_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(output = "Tensor", total_weight = "Tensor", self = "Tensor", 
    target = "Tensor", weight = "Tensor", reduction = "int64_t", 
    ignore_index = "int64_t")
nd_args <- c("output", "total_weight", "self", "target", "weight", "reduction", 
"ignore_index")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss2d_forward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nll_loss2d_out <- function(out, self, target, weight = list(), reduction = torch_reduction_mean(), ignore_index = -100) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "weight", "reduction", "ignore_index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", weight = "Tensor", 
    reduction = "int64_t", ignore_index = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nll_loss2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nonzero <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nonzero', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nonzero_numpy <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nonzero_numpy', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nonzero_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nonzero_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_norm <- function(self, p = 2, dim, keepdim = FALSE, dtype) {
  
args <- rlang::env_get_list(nms = c("self", "p", "dim", "keepdim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "Scalar", dim = c("IntArrayRef", "DimnameList"
), keepdim = "bool", dtype = "ScalarType")
nd_args <- c("self", "dim", "dtype")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_norm_except_dim <- function(v, pow = 2, dim = 0) {
  
args <- rlang::env_get_list(nms = c("v", "pow", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(v = "Tensor", pow = "int64_t", dim = "int64_t")
nd_args <- "v"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('norm_except_dim', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_norm_out <- function(out, self, p, dim, keepdim = FALSE, dtype) {
  
args <- rlang::env_get_list(nms = c("out", "self", "p", "dim", "keepdim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", p = "Scalar", dim = c("IntArrayRef", 
"DimnameList"), keepdim = "bool", dtype = "ScalarType")
nd_args <- c("out", "self", "p", "dim", "dtype")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('norm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_normal <- function(mean, std = 1, size, generator = NULL, options = list()) {
  
args <- rlang::env_get_list(nms = c("mean", "std", "size", "generator", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(mean = c("Tensor", "double"), std = c("double", "Tensor"
), size = "IntArrayRef", generator = "Generator *", options = "TensorOptions")
nd_args <- c("mean", "size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('normal', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_normal_out <- function(out, mean, std = 1, size, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "mean", "std", "size", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", mean = c("Tensor", "double"), std = c("double", 
"Tensor"), size = "IntArrayRef", generator = "Generator *")
nd_args <- c("out", "mean", "size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('normal_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nuclear_norm <- function(self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "IntArrayRef", keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nuclear_norm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_nuclear_norm_out <- function(out, self, dim, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = "IntArrayRef", keepdim = "bool")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('nuclear_norm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_one_hot <- function(self, num_classes = -1) {
  
args <- rlang::env_get_list(nms = c("self", "num_classes"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", num_classes = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('one_hot', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ones <- function(size, names, options = list()) {
  
args <- rlang::env_get_list(nms = c("size", "names", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", names = "DimnameList", options = "TensorOptions")
nd_args <- c("size", "names")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ones', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ones_like <- function(self, options, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", options = "TensorOptions", memory_format = "MemoryFormat")
nd_args <- c("self", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ones_like', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ones_out <- function(out, size) {
  
args <- rlang::env_get_list(nms = c("out", "size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", size = "IntArrayRef")
nd_args <- c("out", "size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ones_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_orgqr <- function(self, input2) {
  
args <- rlang::env_get_list(nms = c("self", "input2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", input2 = "Tensor")
nd_args <- c("self", "input2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('orgqr', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_orgqr_out <- function(out, self, input2) {
  
args <- rlang::env_get_list(nms = c("out", "self", "input2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", input2 = "Tensor")
nd_args <- c("out", "self", "input2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('orgqr_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ormqr <- function(self, input2, input3, left = TRUE, transpose = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "input2", "input3", "left", "transpose"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", input2 = "Tensor", input3 = "Tensor", left = "bool", 
    transpose = "bool")
nd_args <- c("self", "input2", "input3")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ormqr', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_ormqr_out <- function(out, self, input2, input3, left = TRUE, transpose = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "input2", "input3", "left", "transpose"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", input2 = "Tensor", input3 = "Tensor", 
    left = "bool", transpose = "bool")
nd_args <- c("out", "self", "input2", "input3")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('ormqr_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_pairwise_distance <- function(x1, x2, p = 2, eps = 0.000001, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("x1", "x2", "p", "eps", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(x1 = "Tensor", x2 = "Tensor", p = "double", eps = "double", 
    keepdim = "bool")
nd_args <- c("x1", "x2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('pairwise_distance', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_pdist <- function(self, p = 2) {
  
args <- rlang::env_get_list(nms = c("self", "p"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "double")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('pdist', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_pinverse <- function(self, rcond = 0.000000) {
  
args <- rlang::env_get_list(nms = c("self", "rcond"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", rcond = "double")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('pinverse', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_pixel_shuffle <- function(self, upscale_factor) {
  
args <- rlang::env_get_list(nms = c("self", "upscale_factor"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", upscale_factor = "int64_t")
nd_args <- c("self", "upscale_factor")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('pixel_shuffle', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_poisson <- function(self, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", generator = "Generator *")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('poisson', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_poisson_nll_loss <- function(input, target, log_input, full, eps, reduction) {
  
args <- rlang::env_get_list(nms = c("input", "target", "log_input", "full", "eps", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", target = "Tensor", log_input = "bool", 
    full = "bool", eps = "double", reduction = "int64_t")
nd_args <- c("input", "target", "log_input", "full", "eps", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('poisson_nll_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_polygamma <- function(n, self) {
  
args <- rlang::env_get_list(nms = c("n", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(n = "int64_t", self = "Tensor")
nd_args <- c("n", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('polygamma', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_polygamma_out <- function(out, n, self) {
  
args <- rlang::env_get_list(nms = c("out", "n", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", n = "int64_t", self = "Tensor")
nd_args <- c("out", "n", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('polygamma_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_pow <- function(self, exponent) {
  
args <- rlang::env_get_list(nms = c("self", "exponent"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = c("Tensor", "Scalar"), exponent = c("Scalar", "Tensor"
))
nd_args <- c("self", "exponent")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('pow', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_pow_out <- function(out, self, exponent) {
  
args <- rlang::env_get_list(nms = c("out", "self", "exponent"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = c("Tensor", "Scalar"), exponent = c("Scalar", 
"Tensor"))
nd_args <- c("out", "self", "exponent")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('pow_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_prelu <- function(self, weight) {
  
args <- rlang::env_get_list(nms = c("self", "weight"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor")
nd_args <- c("self", "weight")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('prelu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_prelu_backward <- function(grad_output, self, weight) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "weight"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", weight = "Tensor")
nd_args <- c("grad_output", "self", "weight")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('prelu_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_prod <- function(self, dim, keepdim = FALSE, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), keepdim = "bool", 
    dtype = "ScalarType")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('prod', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_prod_out <- function(out, self, dim, keepdim = FALSE, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "keepdim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("int64_t", "Dimname"
), keepdim = "bool", dtype = "ScalarType")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('prod_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_promote_types <- function(type1, type2) {
  
args <- rlang::env_get_list(nms = c("type1", "type2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(type1 = "ScalarType", type2 = "ScalarType")
nd_args <- c("type1", "type2")
return_types <- c('ScalarType')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('promote_types', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_q_per_channel_axis <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('int64_t')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('q_per_channel_axis', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_q_per_channel_scales <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('q_per_channel_scales', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_q_per_channel_zero_points <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('q_per_channel_zero_points', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_q_scale <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('double')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('q_scale', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_q_zero_point <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('int64_t')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('q_zero_point', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_qr <- function(self, some = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "some"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", some = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('qr', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_qr_out <- function(Q, R, self, some = TRUE) {
  
args <- rlang::env_get_list(nms = c("Q", "R", "self", "some"))
args <- Filter(Negate(is.name), args)
expected_types <- list(Q = "Tensor", R = "Tensor", self = "Tensor", some = "bool")
nd_args <- c("Q", "R", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('qr_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantize_per_channel <- function(self, scales, zero_points, axis, dtype) {
  
args <- rlang::env_get_list(nms = c("self", "scales", "zero_points", "axis", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", scales = "Tensor", zero_points = "Tensor", 
    axis = "int64_t", dtype = "ScalarType")
nd_args <- c("self", "scales", "zero_points", "axis", "dtype")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantize_per_channel', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantize_per_tensor <- function(self, scale, zero_point, dtype) {
  
args <- rlang::env_get_list(nms = c("self", "scale", "zero_point", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", scale = "double", zero_point = "int64_t", 
    dtype = "ScalarType")
nd_args <- c("self", "scale", "zero_point", "dtype")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantize_per_tensor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantized_gru <- function(data, input, batch_sizes, hx, params, has_biases, num_layers, dropout, train, batch_first, bidirectional) {
  
args <- rlang::env_get_list(nms = c("data", "input", "batch_sizes", "hx", "params", "has_biases", "num_layers", "dropout", "train", "batch_first", "bidirectional"))
args <- Filter(Negate(is.name), args)
expected_types <- list(data = "Tensor", input = "Tensor", batch_sizes = "Tensor", 
    hx = "Tensor", params = "TensorList", has_biases = "bool", 
    num_layers = "int64_t", dropout = "double", train = "bool", 
    batch_first = "bool", bidirectional = "bool")
nd_args <- c("data", "input", "batch_sizes", "hx", "params", "has_biases", 
"num_layers", "dropout", "train", "batch_first", "bidirectional"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantized_gru', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantized_gru_cell <- function(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh) {
  
args <- rlang::env_get_list(nms = c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh", "packed_ih", "packed_hh", "col_offsets_ih", "col_offsets_hh", "scale_ih", "scale_hh", "zero_point_ih", "zero_point_hh"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", hx = "Tensor", w_ih = "Tensor", w_hh = "Tensor", 
    b_ih = "Tensor", b_hh = "Tensor", packed_ih = "Tensor", packed_hh = "Tensor", 
    col_offsets_ih = "Tensor", col_offsets_hh = "Tensor", scale_ih = "Scalar", 
    scale_hh = "Scalar", zero_point_ih = "Scalar", zero_point_hh = "Scalar")
nd_args <- c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh", "packed_ih", 
"packed_hh", "col_offsets_ih", "col_offsets_hh", "scale_ih", 
"scale_hh", "zero_point_ih", "zero_point_hh")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantized_gru_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantized_lstm <- function(data, input, batch_sizes, hx, params, has_biases, num_layers, dropout, train, batch_first, bidirectional, dtype = NULL, use_dynamic = FALSE) {
  
args <- rlang::env_get_list(nms = c("data", "input", "batch_sizes", "hx", "params", "has_biases", "num_layers", "dropout", "train", "batch_first", "bidirectional", "dtype", "use_dynamic"))
args <- Filter(Negate(is.name), args)
expected_types <- list(data = "Tensor", input = "Tensor", batch_sizes = "Tensor", 
    hx = "TensorList", params = "TensorList", has_biases = "bool", 
    num_layers = "int64_t", dropout = "double", train = "bool", 
    batch_first = "bool", bidirectional = "bool", dtype = "ScalarType", 
    use_dynamic = "bool")
nd_args <- c("data", "input", "batch_sizes", "hx", "params", "has_biases", 
"num_layers", "dropout", "train", "batch_first", "bidirectional"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantized_lstm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantized_lstm_cell <- function(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh) {
  
args <- rlang::env_get_list(nms = c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh", "packed_ih", "packed_hh", "col_offsets_ih", "col_offsets_hh", "scale_ih", "scale_hh", "zero_point_ih", "zero_point_hh"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", hx = "TensorList", w_ih = "Tensor", w_hh = "Tensor", 
    b_ih = "Tensor", b_hh = "Tensor", packed_ih = "Tensor", packed_hh = "Tensor", 
    col_offsets_ih = "Tensor", col_offsets_hh = "Tensor", scale_ih = "Scalar", 
    scale_hh = "Scalar", zero_point_ih = "Scalar", zero_point_hh = "Scalar")
nd_args <- c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh", "packed_ih", 
"packed_hh", "col_offsets_ih", "col_offsets_hh", "scale_ih", 
"scale_hh", "zero_point_ih", "zero_point_hh")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantized_lstm_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantized_max_pool2d <- function(self, kernel_size, stride = list(), padding = 0, dilation = 1, ceil_mode = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", kernel_size = "IntArrayRef", stride = "IntArrayRef", 
    padding = "IntArrayRef", dilation = "IntArrayRef", ceil_mode = "bool")
nd_args <- c("self", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantized_max_pool2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantized_rnn_relu_cell <- function(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh) {
  
args <- rlang::env_get_list(nms = c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh", "packed_ih", "packed_hh", "col_offsets_ih", "col_offsets_hh", "scale_ih", "scale_hh", "zero_point_ih", "zero_point_hh"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", hx = "Tensor", w_ih = "Tensor", w_hh = "Tensor", 
    b_ih = "Tensor", b_hh = "Tensor", packed_ih = "Tensor", packed_hh = "Tensor", 
    col_offsets_ih = "Tensor", col_offsets_hh = "Tensor", scale_ih = "Scalar", 
    scale_hh = "Scalar", zero_point_ih = "Scalar", zero_point_hh = "Scalar")
nd_args <- c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh", "packed_ih", 
"packed_hh", "col_offsets_ih", "col_offsets_hh", "scale_ih", 
"scale_hh", "zero_point_ih", "zero_point_hh")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantized_rnn_relu_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_quantized_rnn_tanh_cell <- function(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh) {
  
args <- rlang::env_get_list(nms = c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh", "packed_ih", "packed_hh", "col_offsets_ih", "col_offsets_hh", "scale_ih", "scale_hh", "zero_point_ih", "zero_point_hh"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", hx = "Tensor", w_ih = "Tensor", w_hh = "Tensor", 
    b_ih = "Tensor", b_hh = "Tensor", packed_ih = "Tensor", packed_hh = "Tensor", 
    col_offsets_ih = "Tensor", col_offsets_hh = "Tensor", scale_ih = "Scalar", 
    scale_hh = "Scalar", zero_point_ih = "Scalar", zero_point_hh = "Scalar")
nd_args <- c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh", "packed_ih", 
"packed_hh", "col_offsets_ih", "col_offsets_hh", "scale_ih", 
"scale_hh", "zero_point_ih", "zero_point_hh")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('quantized_rnn_tanh_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rand <- function(size, generator, names, options = list()) {
  
args <- rlang::env_get_list(nms = c("size", "generator", "names", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", generator = "Generator *", names = "DimnameList", 
    options = "TensorOptions")
nd_args <- c("size", "generator", "names")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rand', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rand_like <- function(self, options, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", options = "TensorOptions", memory_format = "MemoryFormat")
nd_args <- c("self", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rand_like', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rand_out <- function(out, size, generator) {
  
args <- rlang::env_get_list(nms = c("out", "size", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", size = "IntArrayRef", generator = "Generator *")
nd_args <- c("out", "size", "generator")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rand_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_randint <- function(low, high, size, generator, options = list()) {
  
args <- rlang::env_get_list(nms = c("low", "high", "size", "generator", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(low = "int64_t", high = "int64_t", size = "IntArrayRef", 
    generator = "Generator *", options = "TensorOptions")
nd_args <- c("low", "high", "size", "generator")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('randint', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_randint_like <- function(self, low, high, options, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "low", "high", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", low = "int64_t", high = "int64_t", options = "TensorOptions", 
    memory_format = "MemoryFormat")
nd_args <- c("self", "low", "high", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('randint_like', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_randint_out <- function(out, low, high, size, generator) {
  
args <- rlang::env_get_list(nms = c("out", "low", "high", "size", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", low = "int64_t", high = "int64_t", size = "IntArrayRef", 
    generator = "Generator *")
nd_args <- c("out", "low", "high", "size", "generator")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('randint_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_randn <- function(size, generator, names, options = list()) {
  
args <- rlang::env_get_list(nms = c("size", "generator", "names", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", generator = "Generator *", names = "DimnameList", 
    options = "TensorOptions")
nd_args <- c("size", "generator", "names")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('randn', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_randn_like <- function(self, options, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", options = "TensorOptions", memory_format = "MemoryFormat")
nd_args <- c("self", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('randn_like', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_randn_out <- function(out, size, generator) {
  
args <- rlang::env_get_list(nms = c("out", "size", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", size = "IntArrayRef", generator = "Generator *")
nd_args <- c("out", "size", "generator")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('randn_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_randperm <- function(n, generator, options = list()) {
  
args <- rlang::env_get_list(nms = c("n", "generator", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(n = "int64_t", generator = "Generator *", options = "TensorOptions")
nd_args <- c("n", "generator")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('randperm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_randperm_out <- function(out, n, generator) {
  
args <- rlang::env_get_list(nms = c("out", "n", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", n = "int64_t", generator = "Generator *")
nd_args <- c("out", "n", "generator")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('randperm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_range <- function(start, end, step = 1, options = list()) {
  
args <- rlang::env_get_list(nms = c("start", "end", "step", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(start = "Scalar", end = "Scalar", step = "Scalar", options = "TensorOptions")
nd_args <- c("start", "end")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('range', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_range_out <- function(out, start, end, step = 1) {
  
args <- rlang::env_get_list(nms = c("out", "start", "end", "step"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", start = "Scalar", end = "Scalar", step = "Scalar")
nd_args <- c("out", "start", "end")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('range_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_real <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('real', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_real_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('real_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reciprocal <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reciprocal', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reciprocal_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reciprocal_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reciprocal_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reciprocal_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reflection_pad1d <- function(self, padding) {
  
args <- rlang::env_get_list(nms = c("self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", padding = "IntArrayRef")
nd_args <- c("self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reflection_pad1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reflection_pad1d_backward <- function(grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reflection_pad1d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reflection_pad1d_backward_out <- function(grad_input, grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reflection_pad1d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reflection_pad1d_out <- function(out, self, padding) {
  
args <- rlang::env_get_list(nms = c("out", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("out", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reflection_pad1d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reflection_pad2d <- function(self, padding) {
  
args <- rlang::env_get_list(nms = c("self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", padding = "IntArrayRef")
nd_args <- c("self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reflection_pad2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reflection_pad2d_backward <- function(grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reflection_pad2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reflection_pad2d_backward_out <- function(grad_input, grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reflection_pad2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reflection_pad2d_out <- function(out, self, padding) {
  
args <- rlang::env_get_list(nms = c("out", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("out", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reflection_pad2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_relu <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('relu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_relu_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('relu_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_remainder <- function(self, other) {
  
args <- rlang::env_get_list(nms = c("self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Scalar", "Tensor"))
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('remainder', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_remainder_out <- function(out, self, other) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = c("Scalar", "Tensor"
))
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('remainder_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_renorm <- function(self, p, dim, maxnorm) {
  
args <- rlang::env_get_list(nms = c("self", "p", "dim", "maxnorm"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", p = "Scalar", dim = "int64_t", maxnorm = "Scalar")
nd_args <- c("self", "p", "dim", "maxnorm")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('renorm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_renorm_out <- function(out, self, p, dim, maxnorm) {
  
args <- rlang::env_get_list(nms = c("out", "self", "p", "dim", "maxnorm"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", p = "Scalar", dim = "int64_t", 
    maxnorm = "Scalar")
nd_args <- c("out", "self", "p", "dim", "maxnorm")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('renorm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_repeat_interleave <- function(self, repeats, dim = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "repeats", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", repeats = c("Tensor", "int64_t"), dim = "int64_t")
nd_args <- c("self", "repeats")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('repeat_interleave', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad1d <- function(self, padding) {
  
args <- rlang::env_get_list(nms = c("self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", padding = "IntArrayRef")
nd_args <- c("self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad1d_backward <- function(grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad1d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad1d_backward_out <- function(grad_input, grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad1d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad1d_out <- function(out, self, padding) {
  
args <- rlang::env_get_list(nms = c("out", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("out", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad1d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad2d <- function(self, padding) {
  
args <- rlang::env_get_list(nms = c("self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", padding = "IntArrayRef")
nd_args <- c("self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad2d_backward <- function(grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad2d_backward_out <- function(grad_input, grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad2d_out <- function(out, self, padding) {
  
args <- rlang::env_get_list(nms = c("out", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("out", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad3d <- function(self, padding) {
  
args <- rlang::env_get_list(nms = c("self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", padding = "IntArrayRef")
nd_args <- c("self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad3d_backward <- function(grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad3d_backward_out <- function(grad_input, grad_output, self, padding) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    padding = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_replication_pad3d_out <- function(out, self, padding) {
  
args <- rlang::env_get_list(nms = c("out", "self", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", padding = "IntArrayRef")
nd_args <- c("out", "self", "padding")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('replication_pad3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_reshape <- function(self, shape) {
  
args <- rlang::env_get_list(nms = c("self", "shape"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", shape = "IntArrayRef")
nd_args <- c("self", "shape")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('reshape', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_resize_as_ <- function(self, the_template, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "the_template", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", the_template = "Tensor", memory_format = "MemoryFormat")
nd_args <- c("self", "the_template")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('resize_as_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_result_type <- function(scalar, scalar1, other, scalar2, tensor) {
  
args <- rlang::env_get_list(nms = c("scalar", "scalar1", "other", "scalar2", "tensor"))
args <- Filter(Negate(is.name), args)
expected_types <- list(scalar = "Scalar", scalar1 = "Scalar", other = c("Tensor", 
"Scalar"), scalar2 = "Scalar", tensor = "Tensor")
nd_args <- c("scalar", "scalar1", "other", "scalar2", "tensor")
return_types <- c('ScalarType')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('result_type', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rfft <- function(self, signal_ndim, normalized = FALSE, onesided = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "signal_ndim", "normalized", "onesided"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", signal_ndim = "int64_t", normalized = "bool", 
    onesided = "bool")
nd_args <- c("self", "signal_ndim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rfft', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rnn_relu <- function(data, input, batch_sizes, hx, params, has_biases, num_layers, dropout, train, batch_first, bidirectional) {
  
args <- rlang::env_get_list(nms = c("data", "input", "batch_sizes", "hx", "params", "has_biases", "num_layers", "dropout", "train", "batch_first", "bidirectional"))
args <- Filter(Negate(is.name), args)
expected_types <- list(data = "Tensor", input = "Tensor", batch_sizes = "Tensor", 
    hx = "Tensor", params = "TensorList", has_biases = "bool", 
    num_layers = "int64_t", dropout = "double", train = "bool", 
    batch_first = "bool", bidirectional = "bool")
nd_args <- c("data", "input", "batch_sizes", "hx", "params", "has_biases", 
"num_layers", "dropout", "train", "batch_first", "bidirectional"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rnn_relu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rnn_relu_cell <- function(input, hx, w_ih, w_hh, b_ih = list(), b_hh = list()) {
  
args <- rlang::env_get_list(nms = c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", hx = "Tensor", w_ih = "Tensor", w_hh = "Tensor", 
    b_ih = "Tensor", b_hh = "Tensor")
nd_args <- c("input", "hx", "w_ih", "w_hh")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rnn_relu_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rnn_tanh <- function(data, input, batch_sizes, hx, params, has_biases, num_layers, dropout, train, batch_first, bidirectional) {
  
args <- rlang::env_get_list(nms = c("data", "input", "batch_sizes", "hx", "params", "has_biases", "num_layers", "dropout", "train", "batch_first", "bidirectional"))
args <- Filter(Negate(is.name), args)
expected_types <- list(data = "Tensor", input = "Tensor", batch_sizes = "Tensor", 
    hx = "Tensor", params = "TensorList", has_biases = "bool", 
    num_layers = "int64_t", dropout = "double", train = "bool", 
    batch_first = "bool", bidirectional = "bool")
nd_args <- c("data", "input", "batch_sizes", "hx", "params", "has_biases", 
"num_layers", "dropout", "train", "batch_first", "bidirectional"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rnn_tanh', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rnn_tanh_cell <- function(input, hx, w_ih, w_hh, b_ih = list(), b_hh = list()) {
  
args <- rlang::env_get_list(nms = c("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh"))
args <- Filter(Negate(is.name), args)
expected_types <- list(input = "Tensor", hx = "Tensor", w_ih = "Tensor", w_hh = "Tensor", 
    b_ih = "Tensor", b_hh = "Tensor")
nd_args <- c("input", "hx", "w_ih", "w_hh")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rnn_tanh_cell', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_roll <- function(self, shifts, dims = list()) {
  
args <- rlang::env_get_list(nms = c("self", "shifts", "dims"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", shifts = "IntArrayRef", dims = "IntArrayRef")
nd_args <- c("self", "shifts")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('roll', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rot90 <- function(self, k = 1, dims = c(0,1)) {
  
args <- rlang::env_get_list(nms = c("self", "k", "dims"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", k = "int64_t", dims = "IntArrayRef")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rot90', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_round <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('round', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_round_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('round_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_round_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('round_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rrelu <- function(self, lower = 0.125000, upper = 0.333333, training = FALSE, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "lower", "upper", "training", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", lower = "Scalar", upper = "Scalar", training = "bool", 
    generator = "Generator *")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rrelu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rrelu_ <- function(self, lower = 0.125000, upper = 0.333333, training = FALSE, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "lower", "upper", "training", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", lower = "Scalar", upper = "Scalar", training = "bool", 
    generator = "Generator *")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rrelu_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rrelu_with_noise <- function(self, noise, lower = 0.125000, upper = 0.333333, training = FALSE, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "noise", "lower", "upper", "training", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", noise = "Tensor", lower = "Scalar", upper = "Scalar", 
    training = "bool", generator = "Generator *")
nd_args <- c("self", "noise")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rrelu_with_noise', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rrelu_with_noise_ <- function(self, noise, lower = 0.125000, upper = 0.333333, training = FALSE, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "noise", "lower", "upper", "training", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", noise = "Tensor", lower = "Scalar", upper = "Scalar", 
    training = "bool", generator = "Generator *")
nd_args <- c("self", "noise")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rrelu_with_noise_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rrelu_with_noise_backward <- function(grad_output, self, noise, lower, upper, training) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "noise", "lower", "upper", "training"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", noise = "Tensor", 
    lower = "Scalar", upper = "Scalar", training = "bool")
nd_args <- c("grad_output", "self", "noise", "lower", "upper", "training"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rrelu_with_noise_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rrelu_with_noise_backward_out <- function(grad_input, grad_output, self, noise, lower, upper, training) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "noise", "lower", "upper", "training"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    noise = "Tensor", lower = "Scalar", upper = "Scalar", training = "bool")
nd_args <- c("grad_input", "grad_output", "self", "noise", "lower", "upper", 
"training")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rrelu_with_noise_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rrelu_with_noise_out <- function(out, self, noise, lower = 0.125000, upper = 0.333333, training = FALSE, generator = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "noise", "lower", "upper", "training", "generator"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", noise = "Tensor", lower = "Scalar", 
    upper = "Scalar", training = "bool", generator = "Generator *")
nd_args <- c("out", "self", "noise")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rrelu_with_noise_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rsqrt <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rsqrt', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rsqrt_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rsqrt_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rsqrt_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rsqrt_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_rsub <- function(self, other, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "other", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Tensor", "Scalar"), alpha = "Scalar")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('rsub', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_scalar_tensor <- function(s, options = list()) {
  
args <- rlang::env_get_list(nms = c("s", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(s = "Scalar", options = "TensorOptions")
nd_args <- "s"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('scalar_tensor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_scatter <- function(self, dim, index, src, value) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index", "src", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), index = "Tensor", 
    src = "Tensor", value = "Scalar")
nd_args <- c("self", "dim", "index", "src", "value")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('scatter', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_scatter_add <- function(self, dim, index, src) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index", "src"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), index = "Tensor", 
    src = "Tensor")
nd_args <- c("self", "dim", "index", "src")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('scatter_add', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_select <- function(self, dim, index) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("Dimname", "int64_t"), index = "int64_t")
nd_args <- c("self", "dim", "index")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('select', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_selu <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('selu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_selu_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('selu_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sigmoid <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sigmoid', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sigmoid_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sigmoid_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sigmoid_backward <- function(grad_output, output) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output = "Tensor")
nd_args <- c("grad_output", "output")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sigmoid_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sigmoid_backward_out <- function(grad_input, grad_output, output) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output = "Tensor")
nd_args <- c("grad_input", "grad_output", "output")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sigmoid_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sigmoid_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sigmoid_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sign <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sign', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sign_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sign_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sin <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sin', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sin_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sin_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sin_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sin_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sinh <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sinh', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sinh_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sinh_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sinh_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sinh_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_size <- function(self, dim) {
  
args <- rlang::env_get_list(nms = c("self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"))
nd_args <- c("self", "dim")
return_types <- c('int64_t')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('size', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slice <- function(self, dim = 0, start = 0, end = 9223372036854775807, step = 1) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "start", "end", "step"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", start = "int64_t", end = "int64_t", 
    step = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slice', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slogdet <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slogdet', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_dilated2d <- function(self, weight, kernel_size, bias = list(), stride = 1, padding = 0, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_dilated2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_dilated2d_backward <- function(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "weight", "kernel_size", "stride", "padding", "dilation", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef", output_mask = "std::array<bool,3>")
nd_args <- c("grad_output", "self", "weight", "kernel_size", "stride", "padding", 
"dilation", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_dilated2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_dilated3d <- function(self, weight, kernel_size, bias = list(), stride = 1, padding = 0, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_dilated3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_dilated3d_backward <- function(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "weight", "kernel_size", "stride", "padding", "dilation", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef", output_mask = "std::array<bool,3>")
nd_args <- c("grad_output", "self", "weight", "kernel_size", "stride", "padding", 
"dilation", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_dilated3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_transpose2d <- function(self, weight, kernel_size, bias = list(), stride = 1, padding = 0, output_padding = 0, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding", "output_padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", dilation = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_transpose2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_transpose2d_backward <- function(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "weight", "kernel_size", "stride", "padding", "output_padding", "dilation", "columns", "ones", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", dilation = "IntArrayRef", 
    columns = "Tensor", ones = "Tensor", output_mask = "std::array<bool,3>")
nd_args <- c("grad_output", "self", "weight", "kernel_size", "stride", "padding", 
"output_padding", "dilation", "columns", "ones", "output_mask"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_transpose2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_transpose2d_backward_out <- function(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_weight", "grad_bias", "grad_output", "self", "weight", "kernel_size", "stride", "padding", "output_padding", "dilation", "columns", "ones"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_weight = "Tensor", grad_bias = "Tensor", 
    grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", dilation = "IntArrayRef", 
    columns = "Tensor", ones = "Tensor")
nd_args <- c("grad_input", "grad_weight", "grad_bias", "grad_output", "self", 
"weight", "kernel_size", "stride", "padding", "output_padding", 
"dilation", "columns", "ones")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_transpose2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_transpose2d_out <- function(out, self, weight, kernel_size, bias = list(), stride = 1, padding = 0, output_padding = 0, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "weight", "kernel_size", "bias", "stride", "padding", "output_padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", dilation = "IntArrayRef")
nd_args <- c("out", "self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_transpose2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_transpose3d <- function(self, weight, kernel_size, bias = list(), stride = 1, padding = 0, output_padding = 0, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding", "output_padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", dilation = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_transpose3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_transpose3d_backward <- function(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "weight", "kernel_size", "stride", "padding", "output_padding", "dilation", "finput", "fgrad_input", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", dilation = "IntArrayRef", 
    finput = "Tensor", fgrad_input = "Tensor", output_mask = "std::array<bool,3>")
nd_args <- c("grad_output", "self", "weight", "kernel_size", "stride", "padding", 
"output_padding", "dilation", "finput", "fgrad_input", "output_mask"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_transpose3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_transpose3d_backward_out <- function(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_weight", "grad_bias", "grad_output", "self", "weight", "kernel_size", "stride", "padding", "output_padding", "dilation", "finput", "fgrad_input"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_weight = "Tensor", grad_bias = "Tensor", 
    grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", dilation = "IntArrayRef", 
    finput = "Tensor", fgrad_input = "Tensor")
nd_args <- c("grad_input", "grad_weight", "grad_bias", "grad_output", "self", 
"weight", "kernel_size", "stride", "padding", "output_padding", 
"dilation", "finput", "fgrad_input")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_transpose3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv_transpose3d_out <- function(out, self, weight, kernel_size, bias = list(), stride = 1, padding = 0, output_padding = 0, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "weight", "kernel_size", "bias", "stride", "padding", "output_padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    output_padding = "IntArrayRef", dilation = "IntArrayRef")
nd_args <- c("out", "self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv_transpose3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv3d <- function(self, weight, kernel_size, bias = list(), stride = 1, padding = 0) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv3d_backward <- function(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "weight", "kernel_size", "stride", "padding", "finput", "fgrad_input", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    finput = "Tensor", fgrad_input = "Tensor", output_mask = "std::array<bool,3>")
nd_args <- c("grad_output", "self", "weight", "kernel_size", "stride", "padding", 
"finput", "fgrad_input", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv3d_backward_out <- function(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_weight", "grad_bias", "grad_output", "self", "weight", "kernel_size", "stride", "padding", "finput", "fgrad_input"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_weight = "Tensor", grad_bias = "Tensor", 
    grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    finput = "Tensor", fgrad_input = "Tensor")
nd_args <- c("grad_input", "grad_weight", "grad_bias", "grad_output", "self", 
"weight", "kernel_size", "stride", "padding", "finput", "fgrad_input"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv3d_forward <- function(self, weight, kernel_size, bias, stride, padding) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size", "bias", "stride", "padding"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv3d_forward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv3d_forward_out <- function(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding) {
  
args <- rlang::env_get_list(nms = c("output", "finput", "fgrad_input", "self", "weight", "kernel_size", "bias", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(output = "Tensor", finput = "Tensor", fgrad_input = "Tensor", 
    self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("output", "finput", "fgrad_input", "self", "weight", "kernel_size", 
"bias", "stride", "padding")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv3d_forward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_slow_conv3d_out <- function(out, self, weight, kernel_size, bias = list(), stride = 1, padding = 0) {
  
args <- rlang::env_get_list(nms = c("out", "self", "weight", "kernel_size", "bias", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("out", "self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('slow_conv3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_smm <- function(self, mat2) {
  
args <- rlang::env_get_list(nms = c("self", "mat2"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mat2 = "Tensor")
nd_args <- c("self", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('smm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_smooth_l1_loss <- function(self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('smooth_l1_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_smooth_l1_loss_backward <- function(grad_output, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    reduction = "int64_t")
nd_args <- c("grad_output", "self", "target", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('smooth_l1_loss_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_smooth_l1_loss_backward_out <- function(grad_input, grad_output, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", reduction = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "target", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('smooth_l1_loss_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_smooth_l1_loss_out <- function(out, self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('smooth_l1_loss_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_soft_margin_loss <- function(self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('soft_margin_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_soft_margin_loss_backward <- function(grad_output, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", target = "Tensor", 
    reduction = "int64_t")
nd_args <- c("grad_output", "self", "target", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('soft_margin_loss_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_soft_margin_loss_backward_out <- function(grad_input, grad_output, self, target, reduction) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    target = "Tensor", reduction = "int64_t")
nd_args <- c("grad_input", "grad_output", "self", "target", "reduction")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('soft_margin_loss_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_soft_margin_loss_out <- function(out, self, target, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("out", "self", "target", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", target = "Tensor", reduction = "int64_t")
nd_args <- c("out", "self", "target")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('soft_margin_loss_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softmax <- function(self, dim, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), dtype = "ScalarType")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softmax', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softplus <- function(self, beta = 1, threshold = 20) {
  
args <- rlang::env_get_list(nms = c("self", "beta", "threshold"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", beta = "Scalar", threshold = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softplus', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softplus_backward <- function(grad_output, self, beta, threshold, output) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "beta", "threshold", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", beta = "Scalar", 
    threshold = "Scalar", output = "Tensor")
nd_args <- c("grad_output", "self", "beta", "threshold", "output")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softplus_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softplus_backward_out <- function(grad_input, grad_output, self, beta, threshold, output) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "beta", "threshold", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    beta = "Scalar", threshold = "Scalar", output = "Tensor")
nd_args <- c("grad_input", "grad_output", "self", "beta", "threshold", "output"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softplus_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softplus_out <- function(out, self, beta = 1, threshold = 20) {
  
args <- rlang::env_get_list(nms = c("out", "self", "beta", "threshold"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", beta = "Scalar", threshold = "Scalar")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softplus_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softshrink <- function(self, lambd = 0.500000) {
  
args <- rlang::env_get_list(nms = c("self", "lambd"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", lambd = "Scalar")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softshrink', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softshrink_backward <- function(grad_output, self, lambd) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "lambd"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", lambd = "Scalar")
nd_args <- c("grad_output", "self", "lambd")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softshrink_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softshrink_backward_out <- function(grad_input, grad_output, self, lambd) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "self", "lambd"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", self = "Tensor", 
    lambd = "Scalar")
nd_args <- c("grad_input", "grad_output", "self", "lambd")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softshrink_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_softshrink_out <- function(out, self, lambd = 0.500000) {
  
args <- rlang::env_get_list(nms = c("out", "self", "lambd"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", lambd = "Scalar")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('softshrink_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_solve <- function(self, A) {
  
args <- rlang::env_get_list(nms = c("self", "A"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", A = "Tensor")
nd_args <- c("self", "A")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('solve', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_solve_out <- function(solution, lu, self, A) {
  
args <- rlang::env_get_list(nms = c("solution", "lu", "self", "A"))
args <- Filter(Negate(is.name), args)
expected_types <- list(solution = "Tensor", lu = "Tensor", self = "Tensor", A = "Tensor")
nd_args <- c("solution", "lu", "self", "A")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('solve_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sort <- function(self, dim = -1, descending = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "descending"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"), descending = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sort', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sort_out <- function(values, indices, self, dim = -1, descending = FALSE) {
  
args <- rlang::env_get_list(nms = c("values", "indices", "self", "dim", "descending"))
args <- Filter(Negate(is.name), args)
expected_types <- list(values = "Tensor", indices = "Tensor", self = "Tensor", 
    dim = c("int64_t", "Dimname"), descending = "bool")
nd_args <- c("values", "indices", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sort_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sparse_coo_tensor <- function(indices, values, size, options = list()) {
  
args <- rlang::env_get_list(nms = c("indices", "values", "size", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(indices = "Tensor", values = "Tensor", size = "IntArrayRef", 
    options = "TensorOptions")
nd_args <- c("indices", "values", "size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sparse_coo_tensor', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_split <- function(self, split_size, dim = 0) {
  
args <- rlang::env_get_list(nms = c("self", "split_size", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", split_size = "int64_t", dim = "int64_t")
nd_args <- c("self", "split_size")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('split', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_split_with_sizes <- function(self, split_sizes, dim = 0) {
  
args <- rlang::env_get_list(nms = c("self", "split_sizes", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", split_sizes = "IntArrayRef", dim = "int64_t")
nd_args <- c("self", "split_sizes")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('split_with_sizes', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sqrt <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sqrt', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sqrt_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sqrt_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sqrt_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sqrt_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_squeeze <- function(self, dim) {
  
args <- rlang::env_get_list(nms = c("self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"))
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('squeeze', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sspaddmm <- function(self, mat1, mat2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "mat1", "mat2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", mat1 = "Tensor", mat2 = "Tensor", beta = "Scalar", 
    alpha = "Scalar")
nd_args <- c("self", "mat1", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sspaddmm', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sspaddmm_out <- function(out, self, mat1, mat2, beta = 1, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "mat1", "mat2", "beta", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", mat1 = "Tensor", mat2 = "Tensor", 
    beta = "Scalar", alpha = "Scalar")
nd_args <- c("out", "self", "mat1", "mat2")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sspaddmm_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_stack <- function(tensors, dim = 0) {
  
args <- rlang::env_get_list(nms = c("tensors", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(tensors = "TensorList", dim = "int64_t")
nd_args <- "tensors"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('stack', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_stack_out <- function(out, tensors, dim = 0) {
  
args <- rlang::env_get_list(nms = c("out", "tensors", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", tensors = "TensorList", dim = "int64_t")
nd_args <- c("out", "tensors")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('stack_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_std <- function(self, dim, unbiased = TRUE, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "unbiased", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    unbiased = "bool", keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('std', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_std_mean <- function(self, dim, unbiased = TRUE, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "unbiased", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    unbiased = "bool", keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('std_mean', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_std_out <- function(out, self, dim, unbiased = TRUE, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "unbiased", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("IntArrayRef", 
"DimnameList"), unbiased = "bool", keepdim = "bool")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('std_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_stft <- function(self, n_fft, hop_length = NULL, win_length = NULL, window = list(), normalized = FALSE, onesided = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "n_fft", "hop_length", "win_length", "window", "normalized", "onesided"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", n_fft = "int64_t", hop_length = "int64_t", 
    win_length = "int64_t", window = "Tensor", normalized = "bool", 
    onesided = "bool")
nd_args <- c("self", "n_fft")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('stft', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_stride <- function(self, dim) {
  
args <- rlang::env_get_list(nms = c("self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"))
nd_args <- c("self", "dim")
return_types <- c('int64_t')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('stride', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sub <- function(self, other, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("self", "other", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = c("Tensor", "Scalar"), alpha = "Scalar")
nd_args <- c("self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sub', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sub_out <- function(out, self, other, alpha = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "other", "alpha"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", other = "Tensor", alpha = "Scalar")
nd_args <- c("out", "self", "other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sub_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sum <- function(self, dim, keepdim = FALSE, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "keepdim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    keepdim = "bool", dtype = "ScalarType")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sum', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_sum_out <- function(out, self, dim, keepdim = FALSE, dtype = NULL) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "keepdim", "dtype"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("IntArrayRef", 
"DimnameList"), keepdim = "bool", dtype = "ScalarType")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('sum_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_svd <- function(self, some = TRUE, compute_uv = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "some", "compute_uv"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", some = "bool", compute_uv = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('svd', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_svd_out <- function(U, S, V, self, some = TRUE, compute_uv = TRUE) {
  
args <- rlang::env_get_list(nms = c("U", "S", "V", "self", "some", "compute_uv"))
args <- Filter(Negate(is.name), args)
expected_types <- list(U = "Tensor", S = "Tensor", V = "Tensor", self = "Tensor", 
    some = "bool", compute_uv = "bool")
nd_args <- c("U", "S", "V", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('svd_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_symeig <- function(self, eigenvectors = FALSE, upper = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "eigenvectors", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", eigenvectors = "bool", upper = "bool")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('symeig', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_symeig_out <- function(e, V, self, eigenvectors = FALSE, upper = TRUE) {
  
args <- rlang::env_get_list(nms = c("e", "V", "self", "eigenvectors", "upper"))
args <- Filter(Negate(is.name), args)
expected_types <- list(e = "Tensor", V = "Tensor", self = "Tensor", eigenvectors = "bool", 
    upper = "bool")
nd_args <- c("e", "V", "self")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('symeig_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_t <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('t', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_take <- function(self, index) {
  
args <- rlang::env_get_list(nms = c("self", "index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", index = "Tensor")
nd_args <- c("self", "index")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('take', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_take_out <- function(out, self, index) {
  
args <- rlang::env_get_list(nms = c("out", "self", "index"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", index = "Tensor")
nd_args <- c("out", "self", "index")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('take_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tan <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tan', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tan_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tan_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tan_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tan_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tanh <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tanh', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tanh_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tanh_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tanh_backward <- function(grad_output, output) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output = "Tensor")
nd_args <- c("grad_output", "output")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tanh_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tanh_backward_out <- function(grad_input, grad_output, output) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output = "Tensor")
nd_args <- c("grad_input", "grad_output", "output")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tanh_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tanh_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tanh_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tensordot <- function(self, other, dims_self, dims_other) {
  
args <- rlang::env_get_list(nms = c("self", "other", "dims_self", "dims_other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", other = "Tensor", dims_self = "IntArrayRef", 
    dims_other = "IntArrayRef")
nd_args <- c("self", "other", "dims_self", "dims_other")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tensordot', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv_depthwise2d <- function(self, weight, kernel_size, bias = list(), stride = 1, padding = 0, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv_depthwise2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv_depthwise2d_backward <- function(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "weight", "kernel_size", "stride", "padding", "dilation", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef", output_mask = "std::array<bool,2>")
nd_args <- c("grad_output", "self", "weight", "kernel_size", "stride", "padding", 
"dilation", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv_depthwise2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv_depthwise2d_backward_out <- function(grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_weight", "grad_output", "self", "weight", "kernel_size", "stride", "padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_weight = "Tensor", grad_output = "Tensor", 
    self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    stride = "IntArrayRef", padding = "IntArrayRef", dilation = "IntArrayRef")
nd_args <- c("grad_input", "grad_weight", "grad_output", "self", "weight", 
"kernel_size", "stride", "padding", "dilation")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv_depthwise2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv_depthwise2d_forward <- function(self, weight, kernel_size, bias, stride, padding, dilation) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size", "bias", "stride", "padding", 
"dilation")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv_depthwise2d_forward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv_depthwise2d_forward_out <- function(out, self, weight, kernel_size, bias, stride, padding, dilation) {
  
args <- rlang::env_get_list(nms = c("out", "self", "weight", "kernel_size", "bias", "stride", "padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef")
nd_args <- c("out", "self", "weight", "kernel_size", "bias", "stride", "padding", 
"dilation")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv_depthwise2d_forward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv_depthwise2d_out <- function(out, self, weight, kernel_size, bias = list(), stride = 1, padding = 0, dilation = 1) {
  
args <- rlang::env_get_list(nms = c("out", "self", "weight", "kernel_size", "bias", "stride", "padding", "dilation"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef", 
    dilation = "IntArrayRef")
nd_args <- c("out", "self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv_depthwise2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv2d <- function(self, weight, kernel_size, bias = list(), stride = 1, padding = 0) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv2d_backward <- function(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "weight", "kernel_size", "stride", "padding", "finput", "fgrad_input", "output_mask"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    finput = "Tensor", fgrad_input = "Tensor", output_mask = "std::array<bool,3>")
nd_args <- c("grad_output", "self", "weight", "kernel_size", "stride", "padding", 
"finput", "fgrad_input", "output_mask")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv2d_backward_out <- function(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_weight", "grad_bias", "grad_output", "self", "weight", "kernel_size", "stride", "padding", "finput", "fgrad_input"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_weight = "Tensor", grad_bias = "Tensor", 
    grad_output = "Tensor", self = "Tensor", weight = "Tensor", 
    kernel_size = "IntArrayRef", stride = "IntArrayRef", padding = "IntArrayRef", 
    finput = "Tensor", fgrad_input = "Tensor")
nd_args <- c("grad_input", "grad_weight", "grad_bias", "grad_output", "self", 
"weight", "kernel_size", "stride", "padding", "finput", "fgrad_input"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv2d_forward <- function(self, weight, kernel_size, bias, stride, padding) {
  
args <- rlang::env_get_list(nms = c("self", "weight", "kernel_size", "bias", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("self", "weight", "kernel_size", "bias", "stride", "padding"
)
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv2d_forward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv2d_forward_out <- function(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding) {
  
args <- rlang::env_get_list(nms = c("output", "finput", "fgrad_input", "self", "weight", "kernel_size", "bias", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(output = "Tensor", finput = "Tensor", fgrad_input = "Tensor", 
    self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("output", "finput", "fgrad_input", "self", "weight", "kernel_size", 
"bias", "stride", "padding")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv2d_forward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_thnn_conv2d_out <- function(out, self, weight, kernel_size, bias = list(), stride = 1, padding = 0) {
  
args <- rlang::env_get_list(nms = c("out", "self", "weight", "kernel_size", "bias", "stride", "padding"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", weight = "Tensor", kernel_size = "IntArrayRef", 
    bias = "Tensor", stride = "IntArrayRef", padding = "IntArrayRef")
nd_args <- c("out", "self", "weight", "kernel_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('thnn_conv2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_threshold <- function(self, threshold, value) {
  
args <- rlang::env_get_list(nms = c("self", "threshold", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", threshold = "Scalar", value = "Scalar")
nd_args <- c("self", "threshold", "value")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('threshold', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_threshold_ <- function(self, threshold, value) {
  
args <- rlang::env_get_list(nms = c("self", "threshold", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", threshold = "Scalar", value = "Scalar")
nd_args <- c("self", "threshold", "value")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('threshold_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_threshold_backward <- function(grad_output, self, threshold) {
  
args <- rlang::env_get_list(nms = c("grad_output", "self", "threshold"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", self = "Tensor", threshold = "Scalar")
nd_args <- c("grad_output", "self", "threshold")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('threshold_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_threshold_out <- function(out, self, threshold, value) {
  
args <- rlang::env_get_list(nms = c("out", "self", "threshold", "value"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", threshold = "Scalar", value = "Scalar")
nd_args <- c("out", "self", "threshold", "value")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('threshold_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_to_dense_backward <- function(grad, input) {
  
args <- rlang::env_get_list(nms = c("grad", "input"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", input = "Tensor")
nd_args <- c("grad", "input")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('to_dense_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_to_mkldnn_backward <- function(grad, input) {
  
args <- rlang::env_get_list(nms = c("grad", "input"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad = "Tensor", input = "Tensor")
nd_args <- c("grad", "input")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('to_mkldnn_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_topk <- function(self, k, dim = -1, largest = TRUE, sorted = TRUE) {
  
args <- rlang::env_get_list(nms = c("self", "k", "dim", "largest", "sorted"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", k = "int64_t", dim = "int64_t", largest = "bool", 
    sorted = "bool")
nd_args <- c("self", "k")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('topk', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_topk_out <- function(values, indices, self, k, dim = -1, largest = TRUE, sorted = TRUE) {
  
args <- rlang::env_get_list(nms = c("values", "indices", "self", "k", "dim", "largest", "sorted"))
args <- Filter(Negate(is.name), args)
expected_types <- list(values = "Tensor", indices = "Tensor", self = "Tensor", 
    k = "int64_t", dim = "int64_t", largest = "bool", sorted = "bool")
nd_args <- c("values", "indices", "self", "k")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('topk_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_trace <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('trace', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_transpose <- function(self, dim0, dim1) {
  
args <- rlang::env_get_list(nms = c("self", "dim0", "dim1"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim0 = c("int64_t", "Dimname"), dim1 = c("int64_t", 
"Dimname"))
nd_args <- c("self", "dim0", "dim1")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('transpose', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_trapz <- function(y, dx = 1, x, dim = -1) {
  
args <- rlang::env_get_list(nms = c("y", "dx", "x", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(y = "Tensor", dx = "double", x = "Tensor", dim = "int64_t")
nd_args <- c("y", "x")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('trapz', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_triangular_solve <- function(self, A, upper = TRUE, transpose = FALSE, unitriangular = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "A", "upper", "transpose", "unitriangular"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", A = "Tensor", upper = "bool", transpose = "bool", 
    unitriangular = "bool")
nd_args <- c("self", "A")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('triangular_solve', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_triangular_solve_out <- function(X, M, self, A, upper = TRUE, transpose = FALSE, unitriangular = FALSE) {
  
args <- rlang::env_get_list(nms = c("X", "M", "self", "A", "upper", "transpose", "unitriangular"))
args <- Filter(Negate(is.name), args)
expected_types <- list(X = "Tensor", M = "Tensor", self = "Tensor", A = "Tensor", 
    upper = "bool", transpose = "bool", unitriangular = "bool")
nd_args <- c("X", "M", "self", "A")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('triangular_solve_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tril <- function(self, diagonal = 0) {
  
args <- rlang::env_get_list(nms = c("self", "diagonal"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", diagonal = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tril', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tril_indices <- function(row, col, offset = 0, options = torch_long()) {
  
args <- rlang::env_get_list(nms = c("row", "col", "offset", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(row = "int64_t", col = "int64_t", offset = "int64_t", options = "TensorOptions")
nd_args <- c("row", "col")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tril_indices', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_tril_out <- function(out, self, diagonal = 0) {
  
args <- rlang::env_get_list(nms = c("out", "self", "diagonal"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", diagonal = "int64_t")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('tril_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_triplet_margin_loss <- function(anchor, positive, negative, margin = 1.000000, p = 2, eps = 0.000001, swap = FALSE, reduction = torch_reduction_mean()) {
  
args <- rlang::env_get_list(nms = c("anchor", "positive", "negative", "margin", "p", "eps", "swap", "reduction"))
args <- Filter(Negate(is.name), args)
expected_types <- list(anchor = "Tensor", positive = "Tensor", negative = "Tensor", 
    margin = "double", p = "double", eps = "double", swap = "bool", 
    reduction = "int64_t")
nd_args <- c("anchor", "positive", "negative")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('triplet_margin_loss', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_triu <- function(self, diagonal = 0) {
  
args <- rlang::env_get_list(nms = c("self", "diagonal"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", diagonal = "int64_t")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('triu', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_triu_indices <- function(row, col, offset = 0, options = torch_long()) {
  
args <- rlang::env_get_list(nms = c("row", "col", "offset", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(row = "int64_t", col = "int64_t", offset = "int64_t", options = "TensorOptions")
nd_args <- c("row", "col")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('triu_indices', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_triu_out <- function(out, self, diagonal = 0) {
  
args <- rlang::env_get_list(nms = c("out", "self", "diagonal"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", diagonal = "int64_t")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('triu_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_trunc <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('trunc', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_trunc_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('trunc_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_trunc_out <- function(out, self) {
  
args <- rlang::env_get_list(nms = c("out", "self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor")
nd_args <- c("out", "self")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('trunc_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_unbind <- function(self, dim = 0) {
  
args <- rlang::env_get_list(nms = c("self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("int64_t", "Dimname"))
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('unbind', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_unique_consecutive <- function(self, return_inverse = FALSE, return_counts = FALSE, dim = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "return_inverse", "return_counts", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", return_inverse = "bool", return_counts = "bool", 
    dim = "int64_t")
nd_args <- "self"
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('unique_consecutive', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_unique_dim <- function(self, dim, sorted = TRUE, return_inverse = FALSE, return_counts = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "sorted", "return_inverse", "return_counts"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", sorted = "bool", return_inverse = "bool", 
    return_counts = "bool")
nd_args <- c("self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('unique_dim', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_unique_dim_consecutive <- function(self, dim, return_inverse = FALSE, return_counts = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "return_inverse", "return_counts"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t", return_inverse = "bool", 
    return_counts = "bool")
nd_args <- c("self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('unique_dim_consecutive', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_unsqueeze <- function(self, dim) {
  
args <- rlang::env_get_list(nms = c("self", "dim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = "int64_t")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('unsqueeze', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_bicubic2d <- function(self, output_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("self", "output_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef", align_corners = "bool")
nd_args <- c("self", "output_size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_bicubic2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_bicubic2d_backward <- function(grad_output, output_size, input_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output_size", "input_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output_size = "IntArrayRef", input_size = "IntArrayRef", 
    align_corners = "bool")
nd_args <- c("grad_output", "output_size", "input_size", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_bicubic2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_bicubic2d_backward_out <- function(grad_input, grad_output, output_size, input_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output_size", "input_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output_size = "IntArrayRef", 
    input_size = "IntArrayRef", align_corners = "bool")
nd_args <- c("grad_input", "grad_output", "output_size", "input_size", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_bicubic2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_bicubic2d_out <- function(out, self, output_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef", 
    align_corners = "bool")
nd_args <- c("out", "self", "output_size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_bicubic2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_bilinear2d <- function(self, output_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("self", "output_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef", align_corners = "bool")
nd_args <- c("self", "output_size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_bilinear2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_bilinear2d_backward <- function(grad_output, output_size, input_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output_size", "input_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output_size = "IntArrayRef", input_size = "IntArrayRef", 
    align_corners = "bool")
nd_args <- c("grad_output", "output_size", "input_size", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_bilinear2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_bilinear2d_backward_out <- function(grad_input, grad_output, output_size, input_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output_size", "input_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output_size = "IntArrayRef", 
    input_size = "IntArrayRef", align_corners = "bool")
nd_args <- c("grad_input", "grad_output", "output_size", "input_size", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_bilinear2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_bilinear2d_out <- function(out, self, output_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef", 
    align_corners = "bool")
nd_args <- c("out", "self", "output_size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_bilinear2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_linear1d <- function(self, output_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("self", "output_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef", align_corners = "bool")
nd_args <- c("self", "output_size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_linear1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_linear1d_backward <- function(grad_output, output_size, input_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output_size", "input_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output_size = "IntArrayRef", input_size = "IntArrayRef", 
    align_corners = "bool")
nd_args <- c("grad_output", "output_size", "input_size", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_linear1d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_linear1d_backward_out <- function(grad_input, grad_output, output_size, input_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output_size", "input_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output_size = "IntArrayRef", 
    input_size = "IntArrayRef", align_corners = "bool")
nd_args <- c("grad_input", "grad_output", "output_size", "input_size", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_linear1d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_linear1d_out <- function(out, self, output_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef", 
    align_corners = "bool")
nd_args <- c("out", "self", "output_size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_linear1d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest1d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest1d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest1d_backward <- function(grad_output, output_size, input_size) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output_size", "input_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output_size = "IntArrayRef", input_size = "IntArrayRef")
nd_args <- c("grad_output", "output_size", "input_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest1d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest1d_backward_out <- function(grad_input, grad_output, output_size, input_size) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output_size", "input_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output_size = "IntArrayRef", 
    input_size = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "output_size", "input_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest1d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest1d_out <- function(out, self, output_size) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("out", "self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest1d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest2d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest2d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest2d_backward <- function(grad_output, output_size, input_size) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output_size", "input_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output_size = "IntArrayRef", input_size = "IntArrayRef")
nd_args <- c("grad_output", "output_size", "input_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest2d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest2d_backward_out <- function(grad_input, grad_output, output_size, input_size) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output_size", "input_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output_size = "IntArrayRef", 
    input_size = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "output_size", "input_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest2d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest2d_out <- function(out, self, output_size) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("out", "self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest2d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest3d <- function(self, output_size) {
  
args <- rlang::env_get_list(nms = c("self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest3d_backward <- function(grad_output, output_size, input_size) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output_size", "input_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output_size = "IntArrayRef", input_size = "IntArrayRef")
nd_args <- c("grad_output", "output_size", "input_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest3d_backward_out <- function(grad_input, grad_output, output_size, input_size) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output_size", "input_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output_size = "IntArrayRef", 
    input_size = "IntArrayRef")
nd_args <- c("grad_input", "grad_output", "output_size", "input_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_nearest3d_out <- function(out, self, output_size) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef")
nd_args <- c("out", "self", "output_size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_nearest3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_trilinear3d <- function(self, output_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("self", "output_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", output_size = "IntArrayRef", align_corners = "bool")
nd_args <- c("self", "output_size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_trilinear3d', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_trilinear3d_backward <- function(grad_output, output_size, input_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_output", "output_size", "input_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_output = "Tensor", output_size = "IntArrayRef", input_size = "IntArrayRef", 
    align_corners = "bool")
nd_args <- c("grad_output", "output_size", "input_size", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_trilinear3d_backward', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_trilinear3d_backward_out <- function(grad_input, grad_output, output_size, input_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("grad_input", "grad_output", "output_size", "input_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(grad_input = "Tensor", grad_output = "Tensor", output_size = "IntArrayRef", 
    input_size = "IntArrayRef", align_corners = "bool")
nd_args <- c("grad_input", "grad_output", "output_size", "input_size", "align_corners"
)
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_trilinear3d_backward_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_upsample_trilinear3d_out <- function(out, self, output_size, align_corners) {
  
args <- rlang::env_get_list(nms = c("out", "self", "output_size", "align_corners"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", output_size = "IntArrayRef", 
    align_corners = "bool")
nd_args <- c("out", "self", "output_size", "align_corners")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('upsample_trilinear3d_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_var <- function(self, dim, unbiased = TRUE, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "unbiased", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    unbiased = "bool", keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('var', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_var_mean <- function(self, dim, unbiased = TRUE, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("self", "dim", "unbiased", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", dim = c("IntArrayRef", "DimnameList"), 
    unbiased = "bool", keepdim = "bool")
nd_args <- c("self", "dim")
return_types <- c('TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('var_mean', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_var_out <- function(out, self, dim, unbiased = TRUE, keepdim = FALSE) {
  
args <- rlang::env_get_list(nms = c("out", "self", "dim", "unbiased", "keepdim"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", self = "Tensor", dim = c("IntArrayRef", 
"DimnameList"), unbiased = "bool", keepdim = "bool")
nd_args <- c("out", "self", "dim")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('var_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_where <- function(condition, self, other) {
  
args <- rlang::env_get_list(nms = c("condition", "self", "other"))
args <- Filter(Negate(is.name), args)
expected_types <- list(condition = "Tensor", self = "Tensor", other = "Tensor")
nd_args <- c("condition", "self", "other")
return_types <- c('Tensor', 'TensorList')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('where', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_zero_ <- function(self) {
  
args <- rlang::env_get_list(nms = c("self"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor")
nd_args <- "self"
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('zero_', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_zeros <- function(size, names, options = list()) {
  
args <- rlang::env_get_list(nms = c("size", "names", "options"))
args <- Filter(Negate(is.name), args)
expected_types <- list(size = "IntArrayRef", names = "DimnameList", options = "TensorOptions")
nd_args <- c("size", "names")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('zeros', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_zeros_like <- function(self, options, memory_format = NULL) {
  
args <- rlang::env_get_list(nms = c("self", "options", "memory_format"))
args <- Filter(Negate(is.name), args)
expected_types <- list(self = "Tensor", options = "TensorOptions", memory_format = "MemoryFormat")
nd_args <- c("self", "options")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('zeros_like', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}


torch_zeros_out <- function(out, size) {
  
args <- rlang::env_get_list(nms = c("out", "size"))
args <- Filter(Negate(is.name), args)
expected_types <- list(out = "Tensor", size = "IntArrayRef")
nd_args <- c("out", "size")
return_types <- c('Tensor')
args_t <- all_arguments_to_torch_type(args, expected_types)
nd_args_types <- args_t[[2]][base::names(args_t[[2]]) %in% nd_args]
fun_name <- make_cpp_function_name('zeros_out', nd_args_types, 'namespace')
out <- do_call(getNamespace('torch')[[fun_name]], args_t[[1]])
to_return_type(out, return_types)
}

