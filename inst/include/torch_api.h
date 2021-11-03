#ifndef TORCH_API
#define TORCH_API

#include "torch_types.h"
#include "torch_deleters.h"

SEXP operator_sexp_tensor (const XPtrTorchTensor* self);
SEXP operator_sexp_optional_tensor (const XPtrTorchOptionalTensor* self);
SEXP operator_sexp_tensor_list (const XPtrTorchTensorList* self);
SEXP operator_sexp_scalar (const XPtrTorchScalar* self);
SEXP operator_sexp_scalar_type (const XPtrTorchScalarType* self);
SEXP operator_sexp_tensor_options (const XPtrTorchTensorOptions* self);
SEXP operator_sexp_compilation_unit (const XPtrTorchCompilationUnit* self);
SEXP operator_sexp_device (const XPtrTorchDevice* self);
SEXP operator_sexp_script_module (const XPtrTorchScriptModule* self);
SEXP operator_sexp_script_method (const XPtrTorchScriptMethod* self);
SEXP operator_sexp_dtype (const XPtrTorchDtype* self);
SEXP operator_sexp_dimname (const XPtrTorchDimname* self);
SEXP operator_sexp_dimname_list (const XPtrTorchDimnameList* self);
SEXP operator_sexp_generator (const XPtrTorchGenerator* self);
SEXP operator_sexp_memory_format (const XPtrTorchMemoryFormat* self);
SEXP operator_sexp_vector_string (const XPtrTorchvector_string * self);
SEXP operator_sexp_vector_scalar (const XPtrTorchvector_Scalar* self);
SEXP operator_sexp_string (const XPtrTorchstring* self);
SEXP operator_sexp_jit_named_parameter_list (const XPtrTorchjit_named_parameter_list* self);
SEXP operator_sexp_jit_named_buffer_list (const XPtrTorchjit_named_buffer_list* self);
SEXP operator_sexp_jit_named_module_list (const XPtrTorchjit_named_module_list* self);
SEXP operator_sexp_vector_bool (const XPtrTorchvector_bool* self);
SEXP operator_sexp_vector_int64_t (const XPtrTorchvector_int64_t* self);
SEXP operator_sexp_vector_double (const XPtrTorchvector_double* self);
SEXP operator_sexp_stack (const XPtrTorchStack* self);
SEXP operator_sexp_ivalue (const XPtrTorchIValue* self);
SEXP operator_sexp_tuple (const XPtrTorchTuple* self);
SEXP operator_sexp_named_tuple_helper (const XPtrTorchNamedTupleHelper* self);
SEXP operator_sexp_vector_ivalue (const XPtrTorchvector_IValue* self);
SEXP operator_sexp_generic_dict (const XPtrTorchGenericDict* self);
SEXP operator_sexp_generic_list (const XPtrTorchGenericList* self);

XPtrTorchTensor from_sexp_tensor (SEXP x);
XPtrTorchOptionalTensor from_sexp_optional_tensor (SEXP x);
XPtrTorchIndexTensor from_sexp_index_tensor (SEXP x);
XPtrTorchTensorList from_sexp_tensor_list (SEXP x);
XPtrTorchScalar from_sexp_scalar (SEXP x);
XPtrTorchOptionalTensorList from_sexp_optional_tensor_list (SEXP x);
XPtrTorchIndexTensorList from_sexp_index_tensor_list (SEXP x);
XPtrTorchOptionalIndexTensorList from_sexp_optional_index_tensor_list (SEXP x);
XPtrTorchTensorOptions from_sexp_tensor_options (SEXP x);
XPtrTorchDevice from_sexp_device (SEXP x);
XPtrTorchOptionalDevice from_sexp_optional_device (SEXP x);
XPtrTorchScriptModule from_sexp_script_module (SEXP x);
XPtrTorchScriptMethod from_sexp_script_method (SEXP x);
XPtrTorchDtype from_sexp_dtype (SEXP x);
XPtrTorchDimname from_sexp_dimname (SEXP x);
XPtrTorchDimnameList from_sexp_dimname_list (SEXP x);
XPtrTorchGenerator from_sexp_generator (SEXP x);
XPtrTorchMemoryFormat from_sexp_memory_format (SEXP x);
XPtrTorchIntArrayRef from_sexp_int_array_ref (SEXP x, bool allow_null, bool index);
XPtrTorchOptionalIntArrayRef from_sexp_optional_int_array_ref (SEXP x, bool index);
XPtrTorchstring from_sexp_string (SEXP x);
XPtrTorchTuple from_sexp_tuple (SEXP x);
XPtrTorchTensorDict from_sexp_tensor_dict (SEXP x);
XPtrTorchIValue from_sexp_ivalue (SEXP x);
XPtrTorchvector_bool from_sexp_vector_bool (SEXP x);
XPtrTorchvector_Scalar from_sexp_vector_scalar (SEXP x);
XPtrTorchvector_int64_t from_sexp_vector_int64_t (SEXP x);
XPtrTorchvector_double from_sexp_vector_double (SEXP x);
XPtrTorchNamedTupleHelper from_sexp_named_tuple_helper (SEXP x);
XPtrTorchStack from_sexp_stack (SEXP x);
XPtrTorchCompilationUnit from_sexp_compilation_unit (SEXP x) ;
XPtrTorchint64_t2 from_sexp_int64_t_2 (SEXP x_);
XPtrTorchoptional_int64_t2 from_sexp_optional_int64_t_2 (SEXP x_);
XPtrTorchindex_int64_t from_sexp_index_int64_t (SEXP x_);
XPtrTorchoptional_index_int64_t from_sexp_optional_index_int64_t (SEXP x_);

#endif // TORCH_API