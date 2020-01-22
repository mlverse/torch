is_scalar_atomic <- function(x) {
  if (is.atomic(x) && length(x) == 1)
    TRUE
  else
    FALSE
}


arg_to_torch_tensor <- function(obj, nullable = FALSE) {
  
  if (is.atomic(obj))
    return(torch_tensor(obj))
  
  if (is_torch_tensor(obj))
    return(obj)
  
  # special case where the argument would be discarded.
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a torch_tensor.")
}

arg_to_torch_scalar <- function(obj, nullable = FALSE) {
  
  if (is_scalar_atomic(obj))
    return(torch_scalar(obj))
  
  if (is.null(obj) && nullable)
    return(torch_scalar(obj))
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  if (is_torch_scalar(obj))
    return(obj)
  
  stop("Could not convert the argument to a torch_scalar.")
}

arg_to_torch_memory_format <- function(obj, nullable = FALSE) {
  
  if (is_torch_memory_format(obj))
    return(obj)
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a torch_memory_format.")
}

arg_to_torch_qscheme <- function(obj, nullable) {
  
  if (is_torch_qscheme(obj))
    return(obj)
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a torch_qscheme.")
}

arg_to_torch_dtype <- function(obj, nullable) {
  
  if (is_torch_dtype(obj))
    return(obj)
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a torch_dtype.")
}

arg_to_torch_dimname_list <- function(obj, nullable) {
  
  if (is.character(obj))
    return(torch_dimname_list(obj))
  
  if (is_torch_dimname_list(obj))
    return(obj)
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a torch_dimname_list.")
}

arg_to_torch_dimname <- function(obj, nullable) {
  
  if (is.character(obj) && length(obj) == 1)
    return(torch_dimname(obj))
  
  if (is_torch_dimname(obj))
    return(obj)
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a torch_dimname_list.")
}

arg_to_bool <- function(obj, nullable) {
  
  if (is.logical(obj))
    return(obj)
  
  if (is.null(obj) && nullable)
    return(logical())
  
  if (is.null(obj) && !nullable)
    return(NULL)
    
  stop("Could not convert the argument to a bool.") 
}

arg_to_integer <- function(obj, nullable) {
 
  if (is.numeric(obj))
    return(as.integer(obj))
  
  if (is.null(obj) && nullable)
    return(integer())
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to an integer.") 
}

arg_to_double <- function(obj, nullable) {
  
  if (is.numeric(obj))
    return(as.double(obj))
  
  if (is.null(obj) && nullable)
    return(double())
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a double.") 
}

arg_to_string <- function(obj, nullable) {
  
  if (is.character(obj))
    return(obj)
  
  if (is.null(obj) && nullable)
    return(character())
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a character.") 
}

arg_to_tensor_options <- function(obj, nullable) {
  
  if (is_torch_tensor_options(obj))
    return(obj)
  
  if (is.null(obj) && !nullable)
    return(NULL)
  
  stop("Could not convert the argument to a torch_tensor_options.") 
}

arg_to_torch_generator <- function(obj, nullable) {
  NULL
}

all_arguments_to_torch_type <- function(all_arguments, expected_types) {

  for (nm in names(all_arguments)) {
    all_arguments[[nm]] <- argument_to_torch_type(all_arguments[[nm]], expected_types[[nm]])
  }
  
  arguments <- lapply(all_arguments, function(x) x[[1]])
  types <- sapply(all_arguments, function(x) x[[2]])
  list(arguments, types)
}

argument_to_torch_type <- function(obj, expected_types) {
  
  if ("Tensor" %in% expected_types && is_torch_tensor(obj))
    return(list(obj, "Tensor"))
  
  if ("Scalar" %in% expected_types && is_torch_scalar(obj))
    return(list(obj, "Scalar"))
  
  if ("DimnameList" %in% expected_types && is_torch_dimname_list(obj))
    return(list(obj, "DimnameList"))
  
  if ("TensorList" %in% expected_types && is_torch_tensor_list(obj))
    return(list(obj, "TensorList"))
  
  if ("TensorOptions" %in% expected_types && is_torch_tensor_options(obj))
    return(list(obj, "TensorOptions"))
  
  if ("MemoryFormat" %in% expected_types && is_torch_memory_format(obj))
    return(list(obj, "MemortFormat"))
  
  if ("ScalarType" %in% expected_types && is_torch_dtype(obj))
    return(list(obj, "ScalarType"))
  
  if ("Scalar" %in% expected_types && is_scalar_atomic(obj)) 
    return(list(torch_scalar(obj), "Scalar"))
  
  if ("Tensor" %in% expected_types && is.atomic(obj))
    return(list(torch_tensor(obj), "Tensor"))
  
  if ("DimnameList" %in% expected_types && is.character(obj))
    return(list(torch_dimname_list(obj), "DimnameList"))
  
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
  
  stop("Can't convert argument", call.=FALSE)
}


torch_type_from_r_obj <- function(obj, expected_types) {
  
  if (is_torch_tensor(obj)) return("Tensor")
  if (is_torch_device(obj)) return("Device")
  if (is_torch_dimname(obj)) return("Dimname")
  if (is_torch_scalar(obj)) return("Scalar")
  if (is_torch_dimname_list(obj)) return("DimnameList")
  if (is_torch_dtype(obj)) return("ScalarType")
  if (is_torch_layout(obj)) return("Layout")
  if (is_torch_tensor_options(obj)) return("TensorOptions")
  
  if (is.logical(obj) && length(obj) == 1) return("bool")
  if (is.logical(obj)) return(paste0("std:array<bool,", length(obj), ">"))
  
  if (is.integer(obj) && "IntArrayRef" %in% expected_types)
    return("IntArrayRef")
  
  if (is.integer(obj) && length(obj) == 1 && "int64_t" %in% expected_types)
    return("int64_t")

  stop("Can't find the correct torch type for this object.")
}




arg_to <- function(obj, expected_type, nullable) {
  
  switch (expected_type,
    "Tensor"             = arg_to_torch_tensor(obj, nullable),
    "bool"               = arg_to_bool(obj, nullable),
    "DimnameList"        = arg_to_torch_dimname_list(obj, nullable),
    "TensorList"         = arg_to_tensor_list(obj, nullable),
    "IntArrayRef"        = arg_to_integer(obj, nullable),
    "int64_t"            = arg_to_integer(obj, nullable),
    "double"             = arg_to_double(obj, nullable),
    "std::array<bool,4>" = arg_to_bool(obj, nullable),
    "std::array<bool,3>" = arg_to_bool(obj, nullable),
    "std::array<bool,2>" = arg_to_bool(obj, nullable),
    "TensorOptions"      = arg_to_tensor_options(obj, nullable),
    "Generator *"        = arg_to_torch_generator(obj, nullable),
    "ScalarType"         = arg_to_torch_dtype(obj, nullable),
    "Scalar"             = arg_to_torch_scalar(obj, nullable),
    "MemoryFormat"       = arg_to_torch_memory_format(obj, nullable),
    "std::string"        = arg_to_string(obj, nullable)
  )
  
}
