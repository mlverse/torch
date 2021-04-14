#include "torch_types.h"
#include <set>

inline std::set<std::string> create_set(std::vector<std::string> v)
{
  std::set<std::string> s(v.begin(), v.end());
  return s;
}

inline bool is_in (std::string x, std::set<std::string> y)
{
  return y.find(x) != y.end();
}

// [[Rcpp::export]]
std::string cpp_arg_to_torch_type (SEXP obj, std::vector<std::string> expected_types, 
                                   std::string arg_name) 
{

  if (Rf_isSymbol(obj)) 
  {
    return "Missing";
  }
  
  auto etypes =  create_set(expected_types);
  
  bool e_tensor = is_in("Tensor", etypes);
  bool is_tensor = Rf_inherits(obj, "torch_tensor");
  bool is_null = Rf_isNull(obj);
  
  if (e_tensor && is_tensor)
  {
    return "Tensor";
  }
  
  if (is_in("Scalar", etypes) && (Rf_inherits(obj, "torch_scalar") || is_null))
  {
    return "Scalar"; 
  }
  
  bool e_dimname_list = is_in("DimnameList", etypes);
  if (e_dimname_list && Rf_inherits(obj, "torch_dimname_list"))
  {
    return "DimnameList";
  }
  
  bool is_list = Rf_isNewList(obj);
  if (is_in("TensorOptions", etypes) && (Rf_inherits(obj, "torch_tensor_options") || is_list || is_null))
  {
    return "TensorOptions";
  }
  
  if (is_in("MemoryFormat", etypes) && (Rf_inherits(obj, "torch_memory_format") || is_null))
  {
    return "MemoryFormat";
  }
  
  bool e_scalar_type = is_in("ScalarType", etypes);
  if (e_scalar_type && Rf_inherits(obj, "torch_dtype"))
  {
    return "ScalarType";
  }
  
  if (e_scalar_type && is_null)
  {
    return "ScalarType";
  }
  
  int len = Rf_length(obj);
  bool is_atomic = Rf_isVectorAtomic(obj);
  bool is_scalar_atomic = is_atomic && len == 1;
  
  bool e_scalar = is_in("Scalar", etypes);
  if (e_scalar && (is_scalar_atomic || is_tensor))
  {
    return "Scalar";
  }
  
  if (e_tensor && is_atomic && !is_null)
  {
    return "Tensor";
  }
  
  bool is_character = TYPEOF(obj) == STRSXP;
  if (e_dimname_list && is_character)
  {
    return "DimnameList";
  }
  
  
  bool is_numeric = Rf_isNumeric(obj);
  
  bool is_numeric_or_list_or_null = is_numeric || is_list || is_null;
  
  if (is_in("IntArrayRef", etypes) && is_numeric_or_list_or_null)
  {
    return "IntArrayRef";
  }
  
  if (is_in("ArrayRef<double>", etypes) && (is_numeric || is_null))
  {
    return "ArrayRef<double>";
  }
  
  if (is_in("int64_t", etypes) && ((is_numeric && len == 1) || is_null))
  {
    return "int64_t";
  }
  
  if (is_in("double", etypes) && ((is_numeric && len == 1) || is_null))
  {
    return "double";
  }
  
  bool is_logical = Rf_isLogical(obj);
  if (is_in("bool", etypes) && is_logical && len == 1)
  {
    return "bool";
  }
  
  if (is_in("std::string", etypes) && is_character)
  {
    return "std::string";
  }
  
  if ((is_in("std::array<bool,4>", etypes) || is_in("std::array<bool,3>", etypes)
        || is_in("std::array<bool,2>", etypes)) && is_logical)
  {
    return "std::array<bool," + std::to_string(len) + ">";
  }
  
  if (is_in("TensorList", etypes) && (is_list || is_tensor || is_null || is_numeric))
  {
    return "TensorList";
  }
  
  if (is_in("Generator", etypes) && is_null)
  {
    return "Generator";
  }
  
  if (e_tensor && (is_null || (len == 1 && is_list)))
  {
    return "Tensor";
  }
  
  if (is_in("Device", etypes) && (Rf_inherits(obj, "torch_device") || is_character))
  {
    return "Device";
  }
  
  Rcpp::stop("Can't convert argument");
}

