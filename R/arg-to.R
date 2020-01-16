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

arg_to_torch_dimname_list <- function(obj, nullable) {
  
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
