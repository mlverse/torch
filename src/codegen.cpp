#include <torch.h>

#include <set>

inline std::set<std::string> create_set(const std::vector<std::string>& v) {
  const std::set<std::string> s(v.begin(), v.end());
  return s;
}

inline bool is_in(const std::string& x, const std::set<std::string>& y) {
  return y.find(x) != y.end();
}

// [[Rcpp::export]]
std::string cpp_arg_to_torch_type(SEXP obj,
                                  const std::vector<std::string>& expected_types,
                                  const std::string& arg_name) {
                                  
  if (Rf_isSymbol(obj)) {
    return "Missing";
  }

  auto etypes = create_set(expected_types);

  bool e_tensor = is_in("Tensor", etypes);
  bool is_tensor = Rf_inherits(obj, "torch_tensor");
  bool is_null = Rf_isNull(obj);

  if (e_tensor && is_tensor) {
    return "Tensor";
  }

  bool is_character = TYPEOF(obj) == STRSXP;
  if (is_in("std::string", etypes) && is_character) {
    return "std::string";
  }
  if (is_in("c10::string_view", etypes) && is_character) {
    return "c10::string_view";
  }

  if (is_in("Scalar", etypes) &&
      (Rf_inherits(obj, "torch_scalar") || is_null)) {
    return "Scalar";
  }

  bool e_dimname_list = is_in("DimnameList", etypes);
  if (e_dimname_list && Rf_inherits(obj, "torch_dimname_list")) {
    return "DimnameList";
  }
  
  if (is_in("Dimname", etypes) && is_character) {
    return "Dimname";
  }

  bool is_list = Rf_isNewList(obj);
  if (is_in("TensorOptions", etypes) &&
      (Rf_inherits(obj, "torch_tensor_options") || is_list || is_null)) {
    return "TensorOptions";
  }

  if (is_in("MemoryFormat", etypes) &&
      (Rf_inherits(obj, "torch_memory_format") || is_null)) {
    return "MemoryFormat";
  }

  bool e_scalar_type = is_in("ScalarType", etypes);
  if (e_scalar_type && Rf_inherits(obj, "torch_dtype")) {
    return "ScalarType";
  }
  if (e_scalar_type && is_character) {
    return "ScalarType";
  }
  if (e_scalar_type && is_null) {
    return "ScalarType";
  }

  int len = Rf_length(obj);
  bool is_atomic = Rf_isVectorAtomic(obj);
  bool is_scalar_atomic = is_atomic && len == 1;

  bool is_numeric = Rf_isNumeric(obj);

  bool e_scalar = is_in("Scalar", etypes);
  if (e_scalar && (is_scalar_atomic || is_tensor)) {
    return "Scalar";
  }

  // int64_t and double must come before the Tensor catch-all so that a
  // plain R numeric scalar (e.g. 500 or 0.1) dispatches as int64_t/double
  // rather than Tensor when both types are valid for an argument.
  if (is_in("int64_t", etypes) && is_numeric && len == 1) {
    return "int64_t";
  }

  if (is_in("double", etypes) && is_numeric && len == 1) {
    return "double";
  }

  if (e_tensor && is_atomic && !is_null) {
    return "Tensor";
  }

  if (e_dimname_list && is_character) {
    return "DimnameList";
  }

  if (e_dimname_list && is_null) {
    return "DimnameList";
  }

  bool is_numeric_or_list_or_null = is_numeric || is_list || is_null;

  if (is_in("IntArrayRef", etypes) && is_numeric_or_list_or_null) {
    return "IntArrayRef";
  }

  if (is_in("int64_t", etypes) && is_null) {
    return "int64_t";
  }

  if (is_in("c10::SymInt", etypes) && ((is_numeric && len == 1) || is_null)) {
    return "c10::SymInt";
  }

  if (is_in("ArrayRef<double>", etypes) && (is_numeric || is_null)) {
    return "ArrayRef<double>";
  }

  if (is_in("double", etypes) && ((is_numeric && len == 1) || is_null)) {
    return "double";
  }

  if (is_in("TensorList", etypes) &&
      (is_list || is_tensor || is_null || is_numeric)) {
    return "TensorList";
  }

  bool is_logical = Rf_isLogical(obj);
  if (is_in("bool", etypes) && is_logical && len == 1) {
    return "bool";
  }

  if ((is_in("std::array<bool,4>", etypes) ||
       is_in("std::array<bool,3>", etypes) ||
       is_in("std::array<bool,2>", etypes)) &&
      is_logical) {
    return "std::array<bool," + std::to_string(len) + ">";
  }

  if (is_in("Generator", etypes) && is_null) {
    return "Generator";
  }

  if (e_tensor && (is_null || (len == 1 && is_list))) {
    return "Tensor";
  }

  if (is_in("Device", etypes) &&
      (Rf_inherits(obj, "torch_device") || is_character)) {
    return "Device";
  }

  if (is_in("const c10::List<c10::optional<Tensor>> &", etypes) && is_list) {
    return "const c10::List<c10::optional<Tensor>> &";
  }

  if (is_in("const c10::List<::std::optional<Tensor>> &", etypes) && is_list) {
    return "const c10::List<::std::optional<Tensor>> &";
  }

  Rcpp::stop("Can't convert argument:" + arg_name);
}

inline std::string cpp_suffix(const std::vector<std::string>& arg_names,
                              const std::vector<std::string>& arg_types) {
  std::string out;
  int arg_s = arg_names.size();
  for (int i = 0; i < arg_s; i++) {
    out += arg_names[i] + "_" + arg_types[i];
    if (i != (arg_s - 1)) out += "_";
  }
  return out;
}

// [[Rcpp::export]]
std::string cpp_clean_names(const std::string& x,
                            const std::vector<std::string>& r) {
  std::string out = x;
  char replace;
  int r_size = r.size();
  for (int i = 0; i < r_size; i++) {
    replace = r[i][0];
    out.erase(std::remove(out.begin(), out.end(), replace), out.end());
  }
  return out;
}

const std::vector<std::string> remove_characters = {
    "'", "\"", "%", "#", ":", ">", "<", ",", " ", "*", "&"};

// [[Rcpp::export]]
std::string cpp_make_function_name(const std::string& method_name,
                                   const std::vector<std::string>& arg_names,
                                   const std::vector<std::string>& arg_types,
                                   const std::string& type) {
  std::string out = "cpp_torch_" + type + "_" + method_name + "_";
  out += cpp_suffix(arg_names, arg_types);
  out = cpp_clean_names(out, remove_characters);
  return out;
}

// [[Rcpp::export]]
std::string create_fn_name(const std::string& fun_name,
                           const std::string& fun_type,
                           const std::vector<std::string>& nd_args,
                           const Rcpp::List& args,
                           const Rcpp::List& expected_types) {
  std::vector<std::string> arg_names;
  std::vector<std::string> arg_types;

  std::string type;
  for (auto x : nd_args) {
    type = cpp_arg_to_torch_type(args[x], expected_types[x], x);
    //std::cout << "arg_name: " << x << ": " << type << std::endl;
    for (auto y : Rcpp::as<std::vector<std::string>>(expected_types[x])) {
      //std::cout << "expected_types: " << y << std::endl;
    }
    
    if (type != "Missing") {
      arg_names.push_back(x);
      arg_types.push_back(type);
    }
  }

  const auto nm =
      cpp_make_function_name(fun_name, arg_names, arg_types, fun_type);

  return nm;
}
