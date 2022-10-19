#ifndef TORCH_IMPORTS
#define TORCH_IMPORTS

#include "torch.h"

#define IMPORT_SEXP_OPERATOR(name, type)                         \
  SEXP name(const type* self) {                                  \
    static SEXP (*fn)(const type*) = NULL;                       \
    if (fn == NULL) {                                            \
      fn = (SEXP(*)(const type*))R_GetCCallable("torch", #name); \
    }                                                            \
    return fn(self);                                             \
  }

IMPORT_SEXP_OPERATOR(operator_sexp_tensor, XPtrTorchTensor)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_tensor, XPtrTorchOptionalTensor)
IMPORT_SEXP_OPERATOR(operator_sexp_tensor_list, XPtrTorchTensorList)
IMPORT_SEXP_OPERATOR(operator_sexp_scalar, XPtrTorchScalar)
IMPORT_SEXP_OPERATOR(operator_sexp_scalar_type, XPtrTorchScalarType)
IMPORT_SEXP_OPERATOR(operator_sexp_tensor_options, XPtrTorchTensorOptions)
IMPORT_SEXP_OPERATOR(operator_sexp_compilation_unit, XPtrTorchCompilationUnit)
IMPORT_SEXP_OPERATOR(operator_sexp_device, XPtrTorchDevice)
IMPORT_SEXP_OPERATOR(operator_sexp_script_module, XPtrTorchScriptModule)
IMPORT_SEXP_OPERATOR(operator_sexp_script_method, XPtrTorchScriptMethod)
IMPORT_SEXP_OPERATOR(operator_sexp_dtype, XPtrTorchDtype)
IMPORT_SEXP_OPERATOR(operator_sexp_dimname, XPtrTorchDimname)
IMPORT_SEXP_OPERATOR(operator_sexp_dimname_list, XPtrTorchDimnameList)
IMPORT_SEXP_OPERATOR(operator_sexp_generator, XPtrTorchGenerator)
IMPORT_SEXP_OPERATOR(operator_sexp_memory_format, XPtrTorchMemoryFormat)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_string, XPtrTorchvector_string);
IMPORT_SEXP_OPERATOR(operator_sexp_vector_scalar, XPtrTorchvector_Scalar)
IMPORT_SEXP_OPERATOR(operator_sexp_string, XPtrTorchstring)
IMPORT_SEXP_OPERATOR(operator_sexp_jit_named_parameter_list,
                     XPtrTorchjit_named_parameter_list)
IMPORT_SEXP_OPERATOR(operator_sexp_jit_named_buffer_list,
                     XPtrTorchjit_named_buffer_list)
IMPORT_SEXP_OPERATOR(operator_sexp_jit_named_module_list,
                     XPtrTorchjit_named_module_list)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_bool, XPtrTorchvector_bool)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_int64_t, XPtrTorchvector_int64_t)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_double, XPtrTorchvector_double)
IMPORT_SEXP_OPERATOR(operator_sexp_stack, XPtrTorchStack)
IMPORT_SEXP_OPERATOR(operator_sexp_ivalue, XPtrTorchIValue)
IMPORT_SEXP_OPERATOR(operator_sexp_tuple, XPtrTorchTuple)
IMPORT_SEXP_OPERATOR(operator_sexp_named_tuple_helper,
                     XPtrTorchNamedTupleHelper)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_ivalue, XPtrTorchvector_IValue)
IMPORT_SEXP_OPERATOR(operator_sexp_generic_dict, XPtrTorchGenericDict)
IMPORT_SEXP_OPERATOR(operator_sexp_generic_list, XPtrTorchGenericList)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_dimname_list,
                     XPtrTorchOptionalDimnameList)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_generator,
                     XPtrTorchOptionalGenerator)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_double, XPtrTorchOptionaldouble)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_bool, XPtrTorchoptional_bool)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_int64_t, XPtrTorchoptional_int64_t)
IMPORT_SEXP_OPERATOR(operator_sexp_bool, XPtrTorchbool)
IMPORT_SEXP_OPERATOR(operator_sexp_double, XPtrTorchdouble)
IMPORT_SEXP_OPERATOR(operator_sexp_int64_t, XPtrTorchint64_t)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_scalar, XPtrTorchoptional_scalar)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_string, XPtrTorchoptional_string)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_scalar_type,
                     XPtrTorchoptional_scalar_type)
IMPORT_SEXP_OPERATOR(operator_sexp_optional_memory_format,
                     XPtrTorchoptional_memory_format)
IMPORT_SEXP_OPERATOR(operator_sexp_variable_list, XPtrTorchvariable_list)

#define IMPORT_FROM_SEXP(name, type)                      \
  type name(SEXP x) {                                     \
    static type (*fn)(SEXP) = NULL;                       \
    if (fn == NULL) {                                     \
      fn = (type(*)(SEXP))R_GetCCallable("torch", #name); \
    }                                                     \
    return fn(x);                                         \
  }

IMPORT_FROM_SEXP(from_sexp_tensor, XPtrTorchTensor)
IMPORT_FROM_SEXP(from_sexp_optional_tensor, XPtrTorchOptionalTensor)
IMPORT_FROM_SEXP(from_sexp_index_tensor, XPtrTorchIndexTensor)
IMPORT_FROM_SEXP(from_sexp_tensor_list, XPtrTorchTensorList)
IMPORT_FROM_SEXP(from_sexp_scalar, XPtrTorchScalar)
IMPORT_FROM_SEXP(from_sexp_optional_tensor_list, XPtrTorchOptionalTensorList)
IMPORT_FROM_SEXP(from_sexp_index_tensor_list, XPtrTorchIndexTensorList)
IMPORT_FROM_SEXP(from_sexp_optional_index_tensor_list,
                 XPtrTorchOptionalIndexTensorList)
IMPORT_FROM_SEXP(from_sexp_tensor_options, XPtrTorchTensorOptions)
IMPORT_FROM_SEXP(from_sexp_device, XPtrTorchDevice)
IMPORT_FROM_SEXP(from_sexp_optional_device, XPtrTorchOptionalDevice)
IMPORT_FROM_SEXP(from_sexp_script_module, XPtrTorchScriptModule)
IMPORT_FROM_SEXP(from_sexp_script_method, XPtrTorchScriptMethod)
IMPORT_FROM_SEXP(from_sexp_dtype, XPtrTorchDtype)
IMPORT_FROM_SEXP(from_sexp_dimname, XPtrTorchDimname)
IMPORT_FROM_SEXP(from_sexp_dimname_list, XPtrTorchDimnameList)
IMPORT_FROM_SEXP(from_sexp_generator, XPtrTorchGenerator)
IMPORT_FROM_SEXP(from_sexp_memory_format, XPtrTorchMemoryFormat)
// IMPORT_FROM_SEXP(from_sexp_int_array_ref, XPtrTorchIntArrayRef)
// IMPORT_FROM_SEXP(from_sexp_optional_int_array_ref,
// XPtrTorchOptionalIntArrayRef)
IMPORT_FROM_SEXP(from_sexp_string, XPtrTorchstring)
IMPORT_FROM_SEXP(from_sexp_tuple, XPtrTorchTuple)
IMPORT_FROM_SEXP(from_sexp_tensor_dict, XPtrTorchTensorDict)
IMPORT_FROM_SEXP(from_sexp_ivalue, XPtrTorchIValue)
IMPORT_FROM_SEXP(from_sexp_vector_bool, XPtrTorchvector_bool)
IMPORT_FROM_SEXP(from_sexp_vector_scalar, XPtrTorchvector_Scalar)
IMPORT_FROM_SEXP(from_sexp_vector_int64_t, XPtrTorchvector_int64_t)
IMPORT_FROM_SEXP(from_sexp_vector_double, XPtrTorchvector_double)
IMPORT_FROM_SEXP(from_sexp_named_tuple_helper, XPtrTorchNamedTupleHelper)
IMPORT_FROM_SEXP(from_sexp_stack, XPtrTorchStack)
IMPORT_FROM_SEXP(from_sexp_compilation_unit, XPtrTorchCompilationUnit)
IMPORT_FROM_SEXP(from_sexp_optional_int64_t, XPtrTorchoptional_int64_t)
IMPORT_FROM_SEXP(from_sexp_index_int64_t, XPtrTorchindex_int64_t)
IMPORT_FROM_SEXP(from_sexp_optional_index_int64_t,
                 XPtrTorchoptional_index_int64_t)
IMPORT_FROM_SEXP(from_sexp_optional_bool, XPtrTorchoptional_bool)
IMPORT_FROM_SEXP(from_sexp_bool, XPtrTorchbool)
IMPORT_FROM_SEXP(from_sexp_optional_double_array_ref,
                 XPtrTorchOptionalDoubleArrayRef)
IMPORT_FROM_SEXP(from_sexp_optional_dimname_list, XPtrTorchOptionalDimnameList)
IMPORT_FROM_SEXP(from_sexp_optional_generator, XPtrTorchOptionalGenerator)
IMPORT_FROM_SEXP(from_sexp_double, XPtrTorchdouble)
IMPORT_FROM_SEXP(from_sexp_optional_double, XPtrTorchOptionaldouble)
IMPORT_FROM_SEXP(from_sexp_int64_t, XPtrTorchint64_t)
IMPORT_FROM_SEXP(from_sexp_scalar_type, XPtrTorchScalarType)
IMPORT_FROM_SEXP(from_sexp_optional_scalar, XPtrTorchoptional_scalar)
IMPORT_FROM_SEXP(from_sexp_optional_string, XPtrTorchoptional_string)
IMPORT_FROM_SEXP(from_sexp_optional_scalar_type, XPtrTorchoptional_scalar_type)
IMPORT_FROM_SEXP(from_sexp_optional_memory_format,
                 XPtrTorchoptional_memory_format)
IMPORT_FROM_SEXP(from_sexp_vector_string, XPtrTorchvector_string)
IMPORT_FROM_SEXP(from_sexp_variable_list, XPtrTorchvariable_list)
IMPORT_FROM_SEXP(from_sexp_string_view, XPtrTorchstring_view)
IMPORT_FROM_SEXP(from_sexp_optional_string_view, XPtrTorchoptional_string_view)
IMPORT_FROM_SEXP(from_sexp_sym_int_array_ref, XPtrTorchSymIntArrayRef)
IMPORT_FROM_SEXP(from_sexp_sym_int, XPtrTorchSymInt)

#define IMPORT_DELETER(name)                                \
  void name(void* x) {                                      \
    static void (*fn)(void*) = NULL;                        \
    if (fn == NULL) {                                       \
      fn = (void (*)(void*))R_GetCCallable("torch", #name); \
    }                                                       \
    return fn(x);                                           \
  }

IMPORT_DELETER(delete_tensor)
IMPORT_DELETER(delete_script_module)
IMPORT_DELETER(delete_script_method)
IMPORT_DELETER(delete_optional_tensor)
IMPORT_DELETER(delete_tensor_list)
IMPORT_DELETER(delete_optional_tensor_list)
IMPORT_DELETER(delete_scalar_type)
IMPORT_DELETER(delete_scalar)
IMPORT_DELETER(delete_tensor_options)
IMPORT_DELETER(delete_device)
IMPORT_DELETER(delete_optional_device)
IMPORT_DELETER(delete_dtype)
IMPORT_DELETER(delete_dimname)
IMPORT_DELETER(delete_dimname_list)
IMPORT_DELETER(delete_jit_named_parameter_list)
IMPORT_DELETER(delete_jit_named_buffer_list)
IMPORT_DELETER(delete_jit_named_module_list)
IMPORT_DELETER(delete_generator)
IMPORT_DELETER(delete_memory_format)
IMPORT_DELETER(delete_vector_int64_t)
IMPORT_DELETER(delete_function_ptr)
IMPORT_DELETER(delete_optional_int_array_ref)
IMPORT_DELETER(delete_vector_string)
IMPORT_DELETER(delete_string)
IMPORT_DELETER(delete_stack)
IMPORT_DELETER(delete_ivalue)
IMPORT_DELETER(delete_tuple)
IMPORT_DELETER(delete_vector_bool)
IMPORT_DELETER(delete_vector_scalar)
IMPORT_DELETER(delete_vector_double)
IMPORT_DELETER(delete_tensor_dict)
IMPORT_DELETER(delete_generic_dict)
IMPORT_DELETER(delete_generic_list)
IMPORT_DELETER(delete_vector_ivalue)
IMPORT_DELETER(delete_named_tuple_helper)
IMPORT_DELETER(delete_compilation_unit)
IMPORT_DELETER(delete_qscheme)
IMPORT_DELETER(delete_double)
IMPORT_DELETER(delete_variable_list)
IMPORT_DELETER(delete_int64_t)
IMPORT_DELETER(delete_bool)
IMPORT_DELETER(delete_layout)
IMPORT_DELETER(delete_tensor_index)
IMPORT_DELETER(delete_optional_int64_t)
IMPORT_DELETER(delete_slice)
IMPORT_DELETER(delete_packed_sequence)
IMPORT_DELETER(delete_storage)
IMPORT_DELETER(delete_jit_module)
IMPORT_DELETER(delete_traceable_function)
IMPORT_DELETER(delete_vector_void)
IMPORT_DELETER(delete_optional_bool)
IMPORT_DELETER(delete_optional_double_array_ref)
IMPORT_DELETER(delete_optional_dimname_list)
IMPORT_DELETER(delete_optional_generator)
IMPORT_DELETER(delete_optional_double)
IMPORT_DELETER(delete_optional_string)
IMPORT_DELETER(delete_optional_scalar)
IMPORT_DELETER(delete_optional_scalar_type)
IMPORT_DELETER(delete_optional_memory_format)
IMPORT_DELETER(delete_string_view)
IMPORT_DELETER(delete_optional_string_view)

XPtrTorchIntArrayRef from_sexp_int_array_ref(SEXP x, bool allow_null,
                                             bool index) {
  static XPtrTorchIntArrayRef (*fn)(SEXP, bool, bool) = NULL;
  if (fn == NULL) {
    fn = (XPtrTorchIntArrayRef(*)(SEXP, bool, bool))R_GetCCallable(
        "torch", "from_sexp_int_array_ref");
  }
  return fn(x, allow_null, index);
}

XPtrTorchOptionalIntArrayRef from_sexp_optional_int_array_ref(SEXP x,
                                                              bool index) {
  static XPtrTorchOptionalIntArrayRef (*fn)(SEXP, bool) = NULL;
  if (fn == NULL) {
    fn = (XPtrTorchOptionalIntArrayRef(*)(SEXP, bool))R_GetCCallable(
        "torch", "from_sexp_optional_int_array_ref");
  }
  return fn(x, index);
}

void* fixme_new_string(const char* x, int size) {
  static void* (*fn)(const char*, int) = NULL;
  if (fn == NULL) {
    fn = (void* (*)(const char*, int))R_GetCCallable("torch", "fixme_new_string");
  }
  return fn(x, size);
}

void* fixme_new_dimname(const char* x) {
  static void* (*fn)(const char*) = NULL;
  if (fn == NULL) {
    fn = (void* (*)(const char*))R_GetCCallable("torch", "fixme_new_dimname");
  }
  return fn(x);
}

#endif  // TORCH_IMPORTS
