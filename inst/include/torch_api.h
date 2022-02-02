#ifndef TORCH_API
#define TORCH_API

#include "torch_deleters.h"
#include "torch_types.h"

SEXP operator_sexp_tensor(const XPtrTorchTensor* self);
SEXP operator_sexp_optional_tensor(const XPtrTorchOptionalTensor* self);
SEXP operator_sexp_tensor_list(const XPtrTorchTensorList* self);
SEXP operator_sexp_scalar(const XPtrTorchScalar* self);
SEXP operator_sexp_scalar_type(const XPtrTorchScalarType* self);
SEXP operator_sexp_tensor_options(const XPtrTorchTensorOptions* self);
SEXP operator_sexp_compilation_unit(const XPtrTorchCompilationUnit* self);
SEXP operator_sexp_device(const XPtrTorchDevice* self);
SEXP operator_sexp_script_module(const XPtrTorchScriptModule* self);
SEXP operator_sexp_script_method(const XPtrTorchScriptMethod* self);
SEXP operator_sexp_dtype(const XPtrTorchDtype* self);
SEXP operator_sexp_dimname(const XPtrTorchDimname* self);
SEXP operator_sexp_dimname_list(const XPtrTorchDimnameList* self);
SEXP operator_sexp_generator(const XPtrTorchGenerator* self);
SEXP operator_sexp_memory_format(const XPtrTorchMemoryFormat* self);
SEXP operator_sexp_vector_string(const XPtrTorchvector_string* self);
SEXP operator_sexp_vector_scalar(const XPtrTorchvector_Scalar* self);
SEXP operator_sexp_string(const XPtrTorchstring* self);
SEXP operator_sexp_jit_named_parameter_list(
    const XPtrTorchjit_named_parameter_list* self);
SEXP operator_sexp_jit_named_buffer_list(
    const XPtrTorchjit_named_buffer_list* self);
SEXP operator_sexp_jit_named_module_list(
    const XPtrTorchjit_named_module_list* self);
SEXP operator_sexp_vector_bool(const XPtrTorchvector_bool* self);
SEXP operator_sexp_vector_int64_t(const XPtrTorchvector_int64_t* self);
SEXP operator_sexp_vector_double(const XPtrTorchvector_double* self);
SEXP operator_sexp_stack(const XPtrTorchStack* self);
SEXP operator_sexp_ivalue(const XPtrTorchIValue* self);
SEXP operator_sexp_tuple(const XPtrTorchTuple* self);
SEXP operator_sexp_named_tuple_helper(const XPtrTorchNamedTupleHelper* self);
SEXP operator_sexp_vector_ivalue(const XPtrTorchvector_IValue* self);
SEXP operator_sexp_generic_dict(const XPtrTorchGenericDict* self);
SEXP operator_sexp_generic_list(const XPtrTorchGenericList* self);
SEXP operator_sexp_int64_t(const XPtrTorchint64_t* x);
SEXP operator_sexp_bool(const XPtrTorchbool* x);
SEXP operator_sexp_double(const XPtrTorchdouble* x);
SEXP operator_sexp_optional_dimname_list(const XPtrTorchOptionalDimnameList* x);
SEXP operator_sexp_optional_generator(const XPtrTorchOptionalGenerator* x);
SEXP operator_sexp_optional_double(const XPtrTorchOptionaldouble* x);
SEXP operator_sexp_optional_int64_t(const XPtrTorchoptional_int64_t* x);
SEXP operator_sexp_optional_bool(const XPtrTorchoptional_bool* x);
SEXP operator_sexp_optional_scalar_type(const XPtrTorchoptional_scalar_type* x);
SEXP operator_sexp_optional_string(const XPtrTorchoptional_string* x);
SEXP operator_sexp_optional_scalar(const XPtrTorchoptional_scalar* x);
SEXP operator_sexp_optional_memory_format(
    const XPtrTorchoptional_memory_format* x);
SEXP operator_sexp_variable_list(const XPtrTorchvariable_list* x);

XPtrTorchTensor from_sexp_tensor(SEXP x);
XPtrTorchOptionalTensor from_sexp_optional_tensor(SEXP x);
XPtrTorchIndexTensor from_sexp_index_tensor(SEXP x);
XPtrTorchTensorList from_sexp_tensor_list(SEXP x);
XPtrTorchScalar from_sexp_scalar(SEXP x);
XPtrTorchOptionalTensorList from_sexp_optional_tensor_list(SEXP x);
XPtrTorchIndexTensorList from_sexp_index_tensor_list(SEXP x);
XPtrTorchOptionalIndexTensorList from_sexp_optional_index_tensor_list(SEXP x);
XPtrTorchTensorOptions from_sexp_tensor_options(SEXP x);
XPtrTorchDevice from_sexp_device(SEXP x);
XPtrTorchOptionalDevice from_sexp_optional_device(SEXP x);
XPtrTorchScriptModule from_sexp_script_module(SEXP x);
XPtrTorchScriptMethod from_sexp_script_method(SEXP x);
XPtrTorchDtype from_sexp_dtype(SEXP x);
XPtrTorchDimname from_sexp_dimname(SEXP x);
XPtrTorchDimnameList from_sexp_dimname_list(SEXP x);
XPtrTorchGenerator from_sexp_generator(SEXP x);
XPtrTorchMemoryFormat from_sexp_memory_format(SEXP x);
XPtrTorchIntArrayRef from_sexp_int_array_ref(SEXP x, bool allow_null,
                                             bool index);
XPtrTorchOptionalIntArrayRef from_sexp_optional_int_array_ref(SEXP x,
                                                              bool index);
XPtrTorchstring from_sexp_string(SEXP x);
XPtrTorchTuple from_sexp_tuple(SEXP x);
XPtrTorchTensorDict from_sexp_tensor_dict(SEXP x);
XPtrTorchIValue from_sexp_ivalue(SEXP x);
XPtrTorchvector_bool from_sexp_vector_bool(SEXP x);
XPtrTorchvector_Scalar from_sexp_vector_scalar(SEXP x);
XPtrTorchvector_int64_t from_sexp_vector_int64_t(SEXP x);
XPtrTorchvector_double from_sexp_vector_double(SEXP x);
XPtrTorchNamedTupleHelper from_sexp_named_tuple_helper(SEXP x);
XPtrTorchStack from_sexp_stack(SEXP x);
XPtrTorchCompilationUnit from_sexp_compilation_unit(SEXP x);
XPtrTorchindex_int64_t from_sexp_index_int64_t(SEXP x_);
XPtrTorchoptional_index_int64_t from_sexp_optional_index_int64_t(SEXP x_);
XPtrTorchoptional_bool from_sexp_optional_bool(SEXP x);
XPtrTorchbool from_sexp_bool(SEXP x);
XPtrTorchOptionalDoubleArrayRef from_sexp_optional_double_array_ref(SEXP x);
XPtrTorchOptionalDimnameList from_sexp_optional_dimname_list(SEXP x);
XPtrTorchOptionalGenerator from_sexp_optional_generator(SEXP x);
XPtrTorchdouble from_sexp_double(SEXP x);
XPtrTorchOptionaldouble from_sexp_optional_double(SEXP x);
XPtrTorchoptional_int64_t from_sexp_optional_int64_t(SEXP x);
XPtrTorchint64_t from_sexp_int64_t(SEXP x);
XPtrTorchScalarType from_sexp_scalar_type(SEXP x);
XPtrTorchoptional_scalar_type from_sexp_optional_scalar_type(SEXP x);
XPtrTorchoptional_string from_sexp_optional_string(SEXP x);
XPtrTorchoptional_memory_format from_sexp_optional_memory_format(SEXP x);
XPtrTorchoptional_scalar from_sexp_optional_scalar(SEXP x);
XPtrTorchvector_string from_sexp_vector_string(SEXP x);
XPtrTorchvariable_list from_sexp_variable_list(SEXP x);
XPtrTorchstring_view from_sexp_string_view(SEXP x);
XPtrTorchoptional_string_view from_sexp_optional_string_view(SEXP x);

#endif  // TORCH_API