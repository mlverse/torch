is_scalar_atomic <- function(x) {
  if (is.atomic(x) && length(x) == 1)
    TRUE
  else
    FALSE
}

argument_to_torch_type <- function(obj, expected_types) {
  
  if ("Tensor" %in% expected_types && is_torch_tensor(obj))
    return(list(obj$ptr, "Tensor"))
  
  if ("Scalar" %in% expected_types && is_torch_scalar(obj))
    return(list(obj$ptr, "Scalar"))
  
  if ("DimnameList" %in% expected_types && is_torch_dimname_list(obj))
    return(list(obj$ptr, "DimnameList"))
  
  if ("TensorList" %in% expected_types && is_torch_tensor_list(obj))
    return(list(obj$ptr, "TensorList"))
  
  if ("TensorOptions" %in% expected_types && is_torch_tensor_options(obj))
    return(list(obj$ptr, "TensorOptions"))
  
  if ("MemoryFormat" %in% expected_types && is_torch_memory_format(obj))
    return(list(obj$ptr, "MemoryFormat"))
  
  if ("ScalarType" %in% expected_types && is_torch_dtype(obj))
    return(list(obj$ptr, "ScalarType"))
  
  if ("ScalarType" %in% expected_types && is.null(obj)) 
    return(list(cpp_nullopt(), "ScalarType"))
  
  if ("Scalar" %in% expected_types && is_scalar_atomic(obj)) 
    return(list(torch_scalar(obj)$ptr, "Scalar"))
  
  if ("Tensor" %in% expected_types && is.atomic(obj))
    return(list(torch_tensor(obj)$ptr, "Tensor"))
  
  if ("DimnameList" %in% expected_types && is.character(obj))
    return(list(torch_dimname_list(obj)$ptr, "DimnameList"))
  
  if ("IntArrayRef" %in% expected_types && is.numeric(obj))
    return(list(as.integer(obj), "IntArrayRef"))
  
  if ("int64_t" %in% expected_types && is.numeric(obj) && length(obj) == 1)
    return(list(as.integer(obj), "int64_t"))
  
  if ("bool" %in% expected_types && is.logical(obj) && length(obj) == 1)
    return(list(obj, "bool"))
  
  if ("double" %in% expected_types && is.numeric(obj) && length(obj) == 1)
    return(list(as.double(obj), "double"))
  
  if ("std::string" %in% expected_types && is.character(obj))
    return(list(obj, "std::string"))
  
  if (any(c("std::array<bool,4>", "std::array<bool,3>", "std::array<bool,2>") %in% expected_types) && is.logical(obj))
    return(list(obj, paste0("std::array<bool,", length(obj), ">")))
  
  if ("TensorOptions" %in% expected_types && is.list(obj))
    return(list(as_torch_tensor_options(obj)$ptr, "TensorOptions"))
  
  if ("TensorList" %in% expected_types && is.list(obj)) 
    return(list(torch_tensor_list(obj)$ptr, "TensorList"))
  
  if ("MemoryFormat" %in% expected_types && is.null(obj))
    return(list(cpp_nullopt(), "MemoryFormat"))
  
  browser()
  stop("Can't convert argument", call.=FALSE)
}

all_arguments_to_torch_type <- function(all_arguments, expected_types) {
  
  for (nm in names(all_arguments)) {
    all_arguments[[nm]] <- argument_to_torch_type(all_arguments[[nm]], expected_types[[nm]])
  }
  
  arguments <- lapply(all_arguments, function(x) x[[1]])
  types <- sapply(all_arguments, function(x) x[[2]])
  list(arguments, types)
}

clean_names <- function(x) {
  # adapted from janitor::make_clean_names
  x <- gsub("'", "", x)
  x <- gsub("\"", "", x)
  x <- gsub("%", ".percent_", x)
  x <- gsub("#", ".number_", x)
  x <- gsub(":", "", x)
  x <- gsub("<", "", x)
  x <- gsub(">", "", x)
  x <- gsub(",", "", x)
  x <- gsub(" *", "", x, fixed = TRUE)
  x <- gsub("^[[:space:][:punct:]]+", "", x)
  x
}

make_cpp_function_name <- function(method_name, arg_types, type) {
  
  suffix <- paste(names(arg_types), arg_types, sep = "_")
  suffix <- paste(suffix, collapse = "_")
  
  if (length(suffix) == 0)
    suffix <- ""
  
  clean_names(sprintf("cpp_torch_%s_%s_%s", type, method_name, suffix))
}

do_call <- function(fun, args) {
  args_needed <- names(formals(fun))
  args <- args[args_needed]
  do.call(fun, args)
}

to_return_type <- function(res, types) {
  
  if (length(types) == 1) {
    
    type <- types[[1]]
    
    if (length(type) == 1) {
      
      if (type == "Tensor")
        return(Tensor$new(ptr = res))
      
      if (type == "TensorList")
        return(lapply(res, function(x) Tensor$new(ptr=x)))
      
      if (!inherits(res, "externalptr"))
        return(res)
      
    } else {
      
      out <- seq_along(res) %>% 
        lapply(function(x) to_return_type(res[[x]], type[x]))
      return(out)
      
    }
    
  } 
  
}