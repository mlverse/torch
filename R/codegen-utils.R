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
    return(NULL)
  
  if (any(arg_name == c("index", "indices", "dims")) && any("Tensor" == expected_types) && is_torch_tensor(obj))
    return(list(obj, "Tensor"))
  
  if (any("Tensor" == expected_types) && is_torch_tensor(obj))
    return(list(obj, "Tensor"))
  
  if (any("Scalar" == expected_types) && is_torch_scalar(obj))
    return(list(obj, "Scalar"))
  
  if (any("DimnameList" == expected_types) && is_torch_dimname_list(obj))
    return(list(obj, "DimnameList"))
    
  if (any("TensorOptions" == expected_types) && is_torch_tensor_options(obj))
    return(list(obj, "TensorOptions"))
  
  if (any("MemoryFormat" == expected_types) && is_torch_memory_format(obj))
    return(list(obj, "MemoryFormat"))
  
  if (any("ScalarType" == expected_types) && is_torch_dtype(obj))
    return(list(obj, "ScalarType"))
  
  if (any("ScalarType" == expected_types) && is.null(obj))
    return(list(NULL, "ScalarType"))
  
  if (any("Scalar" == expected_types) && is_scalar_atomic(obj))
    return(list(obj, "Scalar"))
  
  if (arg_name == "index" && any("Tensor" == expected_types) && is.atomic(obj) && !is.null(obj))
    return(list(obj, "Tensor"))
  
  if (any("Tensor" == expected_types) && is.atomic(obj) && !is.null(obj))
    return(list(obj, "Tensor"))
  
  if (any("DimnameList" == expected_types) && is.character(obj))
    return(list(obj, "DimnameList"))
  
  if (any("IntArrayRef" == expected_types) && (is.numeric(obj) || is.list(obj)) && 
      arg_name %in% c("dims", "dims_self", "dims_other", "dim"))
    return(list(obj, "IntArrayRef"))
  
  if (any("IntArrayRef" == expected_types) && is.numeric(obj))
    return(list(as.integer(obj), "IntArrayRef"))
  
  if (any("IntArrayRef" == expected_types) && is.list(obj))
    return(list(as.integer(obj), "IntArrayRef"))
  
  if (any("ArrayRef<double>" == expected_types) && is.numeric(obj))
    return(list(obj, "ArrayRef<double>"))
  
  if (any("IntArrayRef" == expected_types) && is.null(obj))
    return(list(NULL, "IntArrayRef"))
  
  if (any("ArrayRef<double>" == expected_types) && is.null(obj))
    return(list(NULL, "ArrayRef<double>"))
  
  if (any("int64_t" == expected_types) && is.numeric(obj) && length(obj) == 1 && any(arg_name == c("dim", "dim0", "dim1", "dim2", "start_dim", "end_dim", "index")))
    return(list(obj, "int64_t"))
  
  if (any("int64_t" == expected_types) && is.numeric(obj) && length(obj) == 1)
    return(list(as.integer(obj), "int64_t"))
  
  if (any("bool" == expected_types) && is.logical(obj) && length(obj) == 1)
    return(list(obj, "bool"))
  
  if (any("double" == expected_types) && is.numeric(obj) && length(obj) == 1)
    return(list(as.double(obj), "double"))
  
  if (any("std::string" == expected_types) && is.character(obj))
    return(list(obj, "std::string"))
  
  if (any(c("std::array<bool,4>", "std::array<bool,3>", "std::array<bool,2>") %in% expected_types) && is.logical(obj))
    return(list(obj, paste0("std::array<bool,", length(obj), ">")))
  
  if (any("TensorOptions" == expected_types) && is.list(obj))
    return(list(obj, "TensorOptions"))
  
  if (arg_name == "indices" && any("TensorList" == expected_types) && is.list(obj))
    return(list(obj, "TensorList"))
  
  if (any("TensorList" == expected_types) && is.list(obj))
    return(list(obj, "TensorList"))
  
  if (any("MemoryFormat" == expected_types) && is.null(obj))
    return(list(NULL, "MemoryFormat"))
  
  if (any("Generator" == expected_types) && is_torch_generator(obj))
    return(list(obj, "Generator"))
  
  if (any("Generator" == expected_types) && is.null(obj))
    return(list(NULL, "Generator"))
  
  if (any("Scalar" == expected_types) && is.null(obj))
    return(list(obj, "Scalar"))
  
  if (any("int64_t" ==  expected_types) && is.null(obj))
    return(list(NULL, "int64_t"))
  
  if (any("Tensor" == expected_types) && length(obj) == 0 && is.list(obj))
    return(list(NULL, "Tensor"))
  
  if (any("Tensor" == expected_types) && is.null(obj))
    return(list(NULL, "Tensor"))
  
  if (any("double" == expected_types) && is.null(obj))
    return(list(NULL, "double"))
  
  if (any("Device" == expected_types) && is_torch_device(obj))
    return(list(obj, "Device"))
  
  if (any("Device" == expected_types) && is.character(obj))
    return(list(obj, "Device"))
  
  if (any("TensorList" == expected_types) && is.numeric(obj))
    return(list(obj, "TensorList"))
  
  if (any("TensorList" == expected_types) && is_torch_tensor(obj))
    return(list(obj, "TensorList"))
  
  if (any("Scalar" == expected_types) && is_torch_tensor(obj))
    return(list(obj, "Scalar"))
  
  stop("Can't convert argument", call.=FALSE)
}

all_arguments_to_torch_type <- function(all_arguments, expected_types) {
  
  arguments <- list()
  types <- character()
  for (nm in names(all_arguments)) {
    values_and_types <- argument_to_torch_type(all_arguments[[nm]], expected_types[[nm]], nm)
    if (!is.null(values_and_types)) {
      
      if (is.null(values_and_types[[1]]))
        arguments[nm] <- list(NULL)
      else
        arguments[[nm]] <- values_and_types[[1]]
      
      types[[nm]] <- values_and_types[[2]]
    }
  }

  list(arguments, types)
}

clean_chars <- c("'", "\"", "%", "#", ":", ">", "<", ",", " ", "*")

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
  args_t <- all_arguments_to_torch_type(args, expected_types)
  nd_args_types <- args_t[[2]][names(args_t[[2]]) %in% nd_args]
  fun_name <- make_cpp_function_name(fun_name, nd_args_types, fun_type)
  f <- getNamespace('torch')[[fun_name]]
  
  if (is.null(f))
    value_error("{fun_name} does not exist")
  
  out <- do_call(f, args_t[[1]])
  out
}
