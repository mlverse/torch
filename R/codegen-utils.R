is_scalar_atomic <- function(x) {
  if (is.atomic(x) && length(x) == 1)
    TRUE
  else
    FALSE
}

as_1_based_dim <- function(x) {
  x <- as.integer(x)
  
  if (any(x == 0))
    value_error("Dimension is 1-based, but found 0.")
  
  ifelse(x > 0, x - 1, x)
}

as_1_based_tensor_list <- function(x) {
  tensors <- lapply(tensors, as_1_based_tensor)
}

as_1_based_tensor <- function(x) {
  with_no_grad({
    if (!any(x$shape == 0)) {
      e <- torch_min(torch_abs(x))$to(dtype = torch_int())
      if (e$item() == 0)
        runtime_error("Indices/Index start at 1 and got a 0.")  
    }
    
    out <- x - (x > 0)$to(dtype = x$dtype)  
  })
  out
}

argument_to_torch_type <- function(obj, expected_types, arg_name) {
  
  if (is.name(obj))
    return("Missing")
  
  if (any(arg_name == c("index", "indices", "dims")) && any("Tensor" == expected_types) && is_torch_tensor(obj))
    return("Tensor")
  
  if (any("Tensor" == expected_types) && is_torch_tensor(obj))
    return("Tensor")
  
  if (any("Scalar" == expected_types) && is_torch_scalar(obj))
    return("Scalar")
  
  if (any("DimnameList" == expected_types) && is_torch_dimname_list(obj))
    return("DimnameList")
    
  if (any("TensorOptions" == expected_types) && is_torch_tensor_options(obj))
    return("TensorOptions")
  
  if (any("MemoryFormat" == expected_types) && is_torch_memory_format(obj))
    return("MemoryFormat")
  
  if (any("ScalarType" == expected_types) && is_torch_dtype(obj))
    return("ScalarType")
  
  if (any("ScalarType" == expected_types) && is.null(obj))
    return("ScalarType")
  
  if (any("Scalar" == expected_types) && is_scalar_atomic(obj))
    return("Scalar")
  
  if (arg_name == "index" && any("Tensor" == expected_types) && is.atomic(obj) && !is.null(obj))
    return("Tensor")
  
  if (any("Tensor" == expected_types) && is.atomic(obj) && !is.null(obj))
    return("Tensor")
  
  if (any("DimnameList" == expected_types) && is.character(obj))
    return("DimnameList")
  
  if (any("IntArrayRef" == expected_types) && (is.numeric(obj) || is.list(obj)) && 
      arg_name %in% c("dims", "dims_self", "dims_other", "dim"))
    return("IntArrayRef")
  
  if (any("IntArrayRef" == expected_types) && is.numeric(obj))
    return("IntArrayRef")
  
  if (any("IntArrayRef" == expected_types) && is.list(obj))
    return("IntArrayRef")
  
  if (any("ArrayRef<double>" == expected_types) && is.numeric(obj))
    return("ArrayRef<double>")
  
  if (any("IntArrayRef" == expected_types) && is.null(obj))
    return("IntArrayRef")
  
  if (any("ArrayRef<double>" == expected_types) && is.null(obj))
    return("ArrayRef<double>")
  
  if (any("int64_t" == expected_types) && is.numeric(obj) && length(obj) == 1 && any(arg_name == c("dim", "dim0", "dim1", "dim2", "start_dim", "end_dim", "index")))
    return("int64_t")
  
  if (any("int64_t" == expected_types) && is.numeric(obj) && length(obj) == 1)
    return("int64_t")
  
  if (any("bool" == expected_types) && is.logical(obj) && length(obj) == 1)
    return("bool")
  
  if (any("double" == expected_types) && is.numeric(obj) && length(obj) == 1)
    return("double")
  
  if (any("std::string" == expected_types) && is.character(obj))
    return("std::string")
  
  if (any(c("std::array<bool,4>", "std::array<bool,3>", "std::array<bool,2>") %in% expected_types) && is.logical(obj))
    return(paste0("std::array<bool,", length(obj), ">"))
  
  if (any("TensorOptions" == expected_types) && is.list(obj))
    return("TensorOptions")
  
  if (arg_name == "indices" && any("TensorList" == expected_types) && is.list(obj))
    return("TensorList")
  
  if (any("TensorList" == expected_types) && is.list(obj))
    return("TensorList")
  
  if (any("MemoryFormat" == expected_types) && is.null(obj))
    return("MemoryFormat")
  
  if (any("Generator" == expected_types) && is_torch_generator(obj))
    return("Generator")
  
  if (any("Generator" == expected_types) && is.null(obj))
    return("Generator")
  
  if (any("Scalar" == expected_types) && is.null(obj))
    return("Scalar")
  
  if (any("int64_t" ==  expected_types) && is.null(obj))
    return("int64_t")
  
  if (any("Tensor" == expected_types) && length(obj) == 0 && is.list(obj))
    return("Tensor")
  
  if (any("Tensor" == expected_types) && is.null(obj))
    return("Tensor")
  
  if (any("double" == expected_types) && is.null(obj))
    return("double")
  
  if (any("Device" == expected_types) && is_torch_device(obj))
    return("Device")
  
  if (any("Device" == expected_types) && is.character(obj))
    return("Device")
  
  if (any("TensorList" == expected_types) && is.numeric(obj))
    return("TensorList")
  
  if (any("TensorList" == expected_types) && is_torch_tensor(obj))
    return("TensorList")
  
  if (any("Scalar" == expected_types) && is_torch_tensor(obj))
    return("Scalar")
  
  if (any("TensorList" == expected_types) && is.null(obj))
    return("TensorList")
  
  stop("Can't convert argument", call.=FALSE)
}

nd_arguments_to_torch_type <- function(arguments, expected_types) {
  
}

clean_chars <- c("'", "\"", "%", "#", ":", ">", "<", ",", " ", "*", "&")

clean_names <- function(x) {
  cpp_clean_names(x, clean_chars)
}

make_cpp_function_name <- function(method_name, arg_types, type) {
  cpp_make_function_name(method_name, names(arg_types), arg_types, type, clean_chars)
}

do_call <- function(fun, args) {
  args_needed <- names(formals(fun))
  args <- args[args_needed]
  do.call(fun, args)
}

call_c_function <- function(fun_name, args, expected_types, nd_args, return_types, fun_type) {
  
  # types <- character()
  # 
  # for (nm in nd_args) {
  #   type <- cpp_arg_to_torch_type(args[[nm]], expected_types[[nm]], nm)
  #   if (type != "Missing")
  #     types[[nm]] <- type
  # }
  
  fun_name <- create_fn_name(fun_name, fun_type, nd_args, args, expected_types)
  
  # fun_name <- make_cpp_function_name(fun_name, types, fun_type)
  f <- getNamespace('torch')[[fun_name]]
  
  if (is.null(f))
    value_error("{fun_name} does not exist")
  
  out <- do_call(f, args)
  out
}
