#ifdef TORCH_IMPL
#define TORCH_IMPL

#include "torch_api.h"

// torch_tensor

XPtrTorchTensor::operator SEXP() const { return operator_sexp_tensor(this); }

XPtrTorchTensor::XPtrTorchTensor(SEXP x) : XPtrTorch{from_sexp_tensor(x)} {}

// optional_torch_tensor

XPtrTorchOptionalTensor::operator SEXP() const {
  return operator_sexp_optional_tensor(this);
}

XPtrTorchOptionalTensor::XPtrTorchOptionalTensor(SEXP x)
    : XPtrTorch{from_sexp_optional_tensor(x)} {}

// index_tensor

XPtrTorchIndexTensor::operator SEXP() const { return XPtrTorchTensor(*this); }

XPtrTorchIndexTensor::XPtrTorchIndexTensor(SEXP x)
    : XPtrTorch{from_sexp_index_tensor(x)} {}

// tensor list

XPtrTorchTensorList::operator SEXP() const {
  return operator_sexp_tensor_list(this);
}

XPtrTorchTensorList::XPtrTorchTensorList(SEXP x)
    : XPtrTorch{from_sexp_tensor_list(x)} {}

// variable_list

XPtrTorchvariable_list::operator SEXP() const {
  return operator_sexp_variable_list(this);
}

XPtrTorchvariable_list::XPtrTorchvariable_list(SEXP x)
    : XPtrTorch{from_sexp_variable_list(x)} {}

// scalar type

XPtrTorchScalarType::operator SEXP() const {
  return operator_sexp_scalar_type(this);
}

XPtrTorchScalarType::XPtrTorchScalarType(SEXP x)
    : XPtrTorchScalarType{from_sexp_scalar_type(x)} {}

// optional scalar type

XPtrTorchoptional_scalar_type::operator SEXP() const {
  return operator_sexp_optional_scalar_type(this);
}

XPtrTorchoptional_scalar_type::XPtrTorchoptional_scalar_type(SEXP x)
    : XPtrTorchoptional_scalar_type{from_sexp_optional_scalar_type(x)} {}

// scalar

XPtrTorchScalar::operator SEXP() const { return operator_sexp_scalar(this); }

XPtrTorchScalar::XPtrTorchScalar(SEXP x) : XPtrTorch{from_sexp_scalar(x)} {}

// optional scalar

XPtrTorchoptional_scalar::operator SEXP() const {
  return operator_sexp_optional_scalar(this);
}

XPtrTorchoptional_scalar::XPtrTorchoptional_scalar(SEXP x)
    : XPtrTorch{from_sexp_optional_scalar(x)} {}

// tensor options

XPtrTorchTensorOptions::operator SEXP() const {
  return operator_sexp_tensor_options(this);
}

XPtrTorchTensorOptions::XPtrTorchTensorOptions(SEXP x)
    : XPtrTorch{from_sexp_tensor_options(x)} {}

// Compilation unit

XPtrTorchCompilationUnit::operator SEXP() const {
  return operator_sexp_compilation_unit(this);
}

XPtrTorchCompilationUnit::XPtrTorchCompilationUnit(SEXP x)
    : XPtrTorchCompilationUnit{from_sexp_compilation_unit(x)} {}

// device

XPtrTorchDevice::operator SEXP() const { return operator_sexp_device(this); }

XPtrTorchDevice::XPtrTorchDevice(SEXP x) : XPtrTorch{from_sexp_device(x)} {}

// script module

XPtrTorchScriptModule::operator SEXP() const {
  return operator_sexp_script_module(this);
}

XPtrTorchScriptModule::XPtrTorchScriptModule(SEXP x)
    : XPtrTorch{from_sexp_script_module(x)} {}

// script method

XPtrTorchScriptMethod::operator SEXP() const {
  return operator_sexp_script_method(this);
}

XPtrTorchScriptMethod::XPtrTorchScriptMethod(SEXP x)
    : XPtrTorch{from_sexp_script_method(x)} {}

// dtype

XPtrTorchDtype::operator SEXP() const { return operator_sexp_dtype(this); }

XPtrTorchDtype::XPtrTorchDtype(SEXP x) : XPtrTorch{from_sexp_dtype(x)} {}

// dimname

XPtrTorchDimname::operator SEXP() const { return operator_sexp_dimname(this); }

XPtrTorchDimname::XPtrTorchDimname(SEXP x) : XPtrTorch{from_sexp_dimname(x)} {}

// dimname_list

XPtrTorchDimnameList::operator SEXP() const {
  return operator_sexp_dimname_list(this);
}

XPtrTorchDimnameList::XPtrTorchDimnameList(SEXP x)
    : XPtrTorch{from_sexp_dimname_list(x)} {}

// Optional dimame list

XPtrTorchOptionalDimnameList::operator SEXP() const {
  return operator_sexp_optional_dimname_list(this);
}

XPtrTorchOptionalDimnameList::XPtrTorchOptionalDimnameList(SEXP x)
    : XPtrTorch{from_sexp_optional_dimname_list(x)} {}

// generator

XPtrTorchGenerator::operator SEXP() const {
  return operator_sexp_generator(this);
}

XPtrTorchGenerator::XPtrTorchGenerator(SEXP x)
    : XPtrTorch{from_sexp_generator(x)} {}

// optional generator

XPtrTorchOptionalGenerator::operator SEXP() const {
  return operator_sexp_optional_generator(this);
}

XPtrTorchOptionalGenerator::XPtrTorchOptionalGenerator(SEXP x)
    : XPtrTorch{from_sexp_optional_generator(x)} {}

// memory_format

XPtrTorchMemoryFormat::operator SEXP() const {
  return operator_sexp_memory_format(this);
}

XPtrTorchMemoryFormat::XPtrTorchMemoryFormat(SEXP x)
    : XPtrTorch{from_sexp_memory_format(x)} {}

// optional memory format

XPtrTorchoptional_memory_format::operator SEXP() const {
  return operator_sexp_optional_memory_format(this);
}

XPtrTorchoptional_memory_format::XPtrTorchoptional_memory_format(SEXP x)
    : XPtrTorch{from_sexp_optional_memory_format(x)} {}

// vector_string

XPtrTorchvector_string::operator SEXP() const {
  return operator_sexp_vector_string(this);
}

XPtrTorchvector_string::XPtrTorchvector_string(SEXP x)
    : XPtrTorch{from_sexp_vector_string(x)} {}

// vector scalar

XPtrTorchvector_Scalar::operator SEXP() const {
  return operator_sexp_vector_scalar(this);
}

XPtrTorchvector_Scalar::XPtrTorchvector_Scalar(SEXP x)
    : XPtrTorchvector_Scalar{from_sexp_vector_scalar(x)} {}

// string

XPtrTorchstring::operator SEXP() const { return operator_sexp_string(this); }

XPtrTorchstring::XPtrTorchstring(SEXP x)
    : XPtrTorchstring{from_sexp_string(x)} {}

// string_view

XPtrTorchstring_view::XPtrTorchstring_view(SEXP x)
    : XPtrTorchstring_view{from_sexp_string_view(x)} {}

// optional_string_view

XPtrTorchoptional_string_view::XPtrTorchoptional_string_view(SEXP x)
    : XPtrTorchoptional_string_view{from_sexp_optional_string_view(x)} {}

// optional string

XPtrTorchoptional_string::operator SEXP() const {
  return operator_sexp_optional_string(this);
}

XPtrTorchoptional_string::XPtrTorchoptional_string(SEXP x)
    : XPtrTorchoptional_string{from_sexp_optional_string(x)} {}

// jit_named_parameter_list

XPtrTorchjit_named_parameter_list::operator SEXP() const {
  return operator_sexp_jit_named_parameter_list(this);
}

// jit_named_buffer_list

XPtrTorchjit_named_buffer_list::operator SEXP() const {
  return operator_sexp_jit_named_buffer_list(this);
}

// jit_named_module_list

XPtrTorchjit_named_module_list::operator SEXP() const {
  return operator_sexp_jit_named_module_list(this);
}

// vector_bool

XPtrTorchvector_bool::operator SEXP() const {
  return operator_sexp_vector_bool(this);
}

XPtrTorchvector_bool::XPtrTorchvector_bool(SEXP x)
    : XPtrTorchvector_bool{from_sexp_vector_bool(x)} {}

// vector_int64_t

XPtrTorchvector_int64_t::operator SEXP() const {
  return operator_sexp_vector_int64_t(this);
}

XPtrTorchvector_int64_t::XPtrTorchvector_int64_t(SEXP x)
    : XPtrTorchvector_int64_t{from_sexp_vector_int64_t(x)} {}

// vector_double

XPtrTorchvector_double::operator SEXP() const {
  return operator_sexp_vector_double(this);
}

XPtrTorchvector_double::XPtrTorchvector_double(SEXP x)
    : XPtrTorchvector_double{from_sexp_vector_double(x)} {}

// stack

XPtrTorchStack::operator SEXP() const { return operator_sexp_stack(this); }

XPtrTorchStack::XPtrTorchStack(SEXP x) : XPtrTorchStack{from_sexp_stack(x)} {}

// ivalue

XPtrTorchIValue::operator SEXP() const { return operator_sexp_ivalue(this); }

XPtrTorchIValue::XPtrTorchIValue(SEXP x)
    : XPtrTorchIValue{from_sexp_ivalue(x)} {}

// tuple

XPtrTorchTuple::operator SEXP() const { return operator_sexp_tuple(this); }

XPtrTorchTuple::XPtrTorchTuple(SEXP x) : XPtrTorchTuple{from_sexp_tuple(x)} {}

// named_tuple_helper

XPtrTorchNamedTupleHelper::operator SEXP() const {
  return operator_sexp_named_tuple_helper(this);
}

XPtrTorchNamedTupleHelper::XPtrTorchNamedTupleHelper(SEXP x)
    : XPtrTorchNamedTupleHelper{from_sexp_named_tuple_helper(x)} {}

// ivalue

XPtrTorchvector_IValue::operator SEXP() const {
  return operator_sexp_vector_ivalue(this);
}

// generic dict

XPtrTorchGenericDict::operator SEXP() const {
  return operator_sexp_generic_dict(this);
}

// generic list

XPtrTorchGenericList::operator SEXP() const {
  return operator_sexp_generic_list(this);
}

// optional_tensor_list

XPtrTorchOptionalTensorList::XPtrTorchOptionalTensorList(SEXP x)
    : XPtrTorch{from_sexp_optional_tensor_list(x)} {}

// index_tensor_list

XPtrTorchIndexTensorList::XPtrTorchIndexTensorList(SEXP x)
    : XPtrTorch{from_sexp_index_tensor_list(x)} {}

// optional_index_tensor_list

XPtrTorchOptionalIndexTensorList::XPtrTorchOptionalIndexTensorList(SEXP x)
    : XPtrTorch{from_sexp_optional_index_tensor_list(x)} {}

// optional device

XPtrTorchOptionalDevice::XPtrTorchOptionalDevice(SEXP x)
    : XPtrTorch{from_sexp_optional_device(x)} {}

// optional double array ref

XPtrTorchOptionalDoubleArrayRef::XPtrTorchOptionalDoubleArrayRef(SEXP x)
    : XPtrTorch{from_sexp_optional_double_array_ref(x)} {}

// int array ref

XPtrTorchIntArrayRef::XPtrTorchIntArrayRef(SEXP x)
    : XPtrTorch{from_sexp_int_array_ref(x, false, false)} {}

// index int array ref

XPtrTorchIndexIntArrayRef::XPtrTorchIndexIntArrayRef(SEXP x)
    : XPtrTorch{from_sexp_int_array_ref(x, false, true)} {}

// optional int array ref

XPtrTorchOptionalIntArrayRef::XPtrTorchOptionalIntArrayRef(SEXP x)
    : XPtrTorchOptionalIntArrayRef(
          from_sexp_optional_int_array_ref(x, false)){};

// optional index int array ref

XPtrTorchOptionalIndexIntArrayRef::XPtrTorchOptionalIndexIntArrayRef(SEXP x)
    : XPtrTorchOptionalIntArrayRef{
          from_sexp_optional_int_array_ref(x, true)} {};

// tensor_dict

XPtrTorchTensorDict::XPtrTorchTensorDict(SEXP x)
    : XPtrTorchTensorDict{from_sexp_tensor_dict(x)} {}

// int64_t

XPtrTorchint64_t::operator SEXP() const { return operator_sexp_int64_t(this); }

XPtrTorchint64_t::XPtrTorchint64_t(SEXP x)
    : XPtrTorchint64_t{from_sexp_int64_t(x)} {}

// optional int64_t

XPtrTorchoptional_int64_t::operator SEXP() const {
  return operator_sexp_optional_int64_t(this);
}

XPtrTorchoptional_int64_t::XPtrTorchoptional_int64_t(SEXP x)
    : XPtrTorchoptional_int64_t{from_sexp_optional_int64_t(x)} {}

// bool

XPtrTorchbool::XPtrTorchbool(SEXP x) : XPtrTorchbool{from_sexp_bool(x)} {}

XPtrTorchbool::operator SEXP() const { return operator_sexp_bool(this); }

// double

XPtrTorchdouble::operator SEXP() const { return operator_sexp_double(this); }

XPtrTorchdouble::XPtrTorchdouble(SEXP x)
    : XPtrTorchdouble{from_sexp_double(x)} {}

// optional double

XPtrTorchOptionaldouble::operator SEXP() const {
  return operator_sexp_optional_double(this);
}

XPtrTorchOptionaldouble::XPtrTorchOptionaldouble(SEXP x)
    : XPtrTorchOptionaldouble{from_sexp_optional_double(x)} {}

// optional_bool

XPtrTorchoptional_bool::XPtrTorchoptional_bool(SEXP x)
    : XPtrTorchoptional_bool{from_sexp_optional_bool(x)} {}

XPtrTorchoptional_bool::operator SEXP() const {
  return operator_sexp_optional_bool(this);
}

// index_int64_t

XPtrTorchindex_int64_t::XPtrTorchindex_int64_t(SEXP x)
    : XPtrTorchindex_int64_t{from_sexp_index_int64_t(x)} {}

// optional index int64_t

XPtrTorchoptional_index_int64_t::XPtrTorchoptional_index_int64_t(SEXP x)
    : XPtrTorchoptional_index_int64_t{from_sexp_optional_index_int64_t(x)} {}

#endif  // TORCH_IMPL