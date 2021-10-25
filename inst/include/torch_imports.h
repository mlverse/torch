#include "torch.h"

#define IMPORT_SEXP_OPERATOR(name, type)                                         \
SEXP name (const type* self)                                                     \
{                                                                                \
  static SEXP (*fn)(const type*) = NULL;                                         \
  if (fn == NULL) {                                                              \
    fn = (SEXP (*)(const type*)) R_GetCCallable("torch", #name);                 \
  }                                                                              \
  return fn(self);                                                               \
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
IMPORT_SEXP_OPERATOR(operator_sexp_jit_named_parameter_list, XPtrTorchjit_named_parameter_list)
IMPORT_SEXP_OPERATOR(operator_sexp_jit_named_buffer_list, XPtrTorchjit_named_buffer_list)
IMPORT_SEXP_OPERATOR(operator_sexp_jit_named_module_list, XPtrTorchjit_named_module_list)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_bool, XPtrTorchvector_bool)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_int64_t, XPtrTorchvector_int64_t)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_double, XPtrTorchvector_double)
IMPORT_SEXP_OPERATOR(operator_sexp_stack, XPtrTorchStack)
IMPORT_SEXP_OPERATOR(operator_sexp_ivalue, XPtrTorchIValue)
IMPORT_SEXP_OPERATOR(operator_sexp_tuple, XPtrTorchTuple)
IMPORT_SEXP_OPERATOR(operator_sexp_named_tuple_helper, XPtrTorchNamedTupleHelper)
IMPORT_SEXP_OPERATOR(operator_sexp_vector_ivalue, XPtrTorchvector_IValue)
IMPORT_SEXP_OPERATOR(operator_sexp_generic_dict, XPtrTorchGenericDict)
IMPORT_SEXP_OPERATOR(operator_sexp_generic_list, XPtrTorchGenericList)
  
#define IMPORT_FROM_SEXP(name, type)                                               \
  type name (SEXP x)                                                               \
  {                                                                                \
    static type (*fn)(SEXP) = NULL;                                                \
    if (fn == NULL) {                                                              \
      fn = (type (*) (SEXP)) R_GetCCallable("torch", #name);                       \
    }                                                                              \
    return fn(x);                                                                  \
  }                                                            

IMPORT_FROM_SEXP(from_sexp_tensor, XPtrTorchTensor)
IMPORT_FROM_SEXP(from_sexp_optional_tensor, XPtrTorchOptionalTensor)
IMPORT_FROM_SEXP(from_sexp_index_tensor, XPtrTorchIndexTensor)
IMPORT_FROM_SEXP(from_sexp_tensor_list, XPtrTorchTensorList)
IMPORT_FROM_SEXP(from_sexp_scalar, XPtrTorchScalar)
IMPORT_FROM_SEXP(from_sexp_optional_tensor_list, XPtrTorchOptionalTensorList)
IMPORT_FROM_SEXP(from_sexp_index_tensor_list, XPtrTorchIndexTensorList)
IMPORT_FROM_SEXP(from_sexp_optional_index_tensor_list, XPtrTorchOptionalIndexTensorList)
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
IMPORT_FROM_SEXP(from_sexp_int_array_ref, XPtrTorchIntArrayRef)
IMPORT_FROM_SEXP(from_sexp_optional_int_array_ref, XPtrTorchOptionalIntArrayRef)
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
IMPORT_FROM_SEXP(from_sexp_int64_t_2, XPtrTorchint64_t2)
IMPORT_FROM_SEXP(from_sexp_optional_int64_t_2, XPtrTorchoptional_int64_t2)
IMPORT_FROM_SEXP(from_sexp_index_int64_t, XPtrTorchindex_int64_t)
IMPORT_FROM_SEXP(from_sexp_optional_index_int64_t, XPtrTorchoptional_index_int64_t)