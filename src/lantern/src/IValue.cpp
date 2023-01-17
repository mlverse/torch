#define LANTERN_BUILD
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

int _lantern_IValue_type(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::IValue*>(self);
  if (self_->isBlob()) return IValue_types::IValueBlobType;
  if (self_->isBool()) return IValue_types::IValueBoolType;
  if (self_->isBoolList()) return IValue_types::IValueBoolListType;
  if (self_->isCapsule()) return IValue_types::IValueCapsuleType;
  if (self_->isComplexDouble()) return IValue_types::IValueComplexDoubleType;
  if (self_->isComplexDoubleList())
    return IValue_types::IValueComplexDoubleListType;
  if (self_->isCustomClass()) return IValue_types::IValueCustomClassType;
  if (self_->isDevice()) return IValue_types::IValueDeviceType;
  if (self_->isDouble()) return IValue_types::IValueDoubleType;
  if (self_->isDoubleList()) return IValue_types::IValueDoubleListType;
  if (self_->isNone()) return IValue_types::IValueNoneType;
  if (self_->isEnum()) return IValue_types::IValueEnumType;
  if (self_->isFuture()) return IValue_types::IValueFutureType;
  if (self_->isGenerator()) return IValue_types::IValueGeneratorType;
  if (self_->isInt()) return IValue_types::IValueIntType;
  if (self_->isIntList()) return IValue_types::IValueIntListType;
  if (self_->isModule()) return IValue_types::IValueModuleType;
  if (self_->isObject()) return IValue_types::IValueObjectType;
  if (self_->isPyObject()) return IValue_types::IValuePyObjectType;
  if (self_->isQuantizer()) return IValue_types::IValueQuantizerType;
  if (self_->isRRef()) return IValue_types::IValueRRefType;
  if (self_->isScalar()) return IValue_types::IValueScalarType;
  if (self_->isStorage()) return IValue_types::IValueStorageType;
  if (self_->isStream()) return IValue_types::IValueStreamType;
  if (self_->isString()) return IValue_types::IValueStringType;
  if (self_->isTensor()) return IValue_types::IValueTensorType;
  if (self_->isGenericDict()) return IValue_types::IValueGenericDictType;
  if (self_->isTensorList()) return IValue_types::IValueTensorListType;
  if (self_->isTuple()) return IValue_types::IValueTupleType;
  if (self_->isList()) return IValue_types::IValueListType;
  if (self_->isPtrType()) return IValue_types::IValuePtrTypeType;

  return IValue_types::IValueTypeUnknownType;
  LANTERN_FUNCTION_END
}

bool _lantern_IValue_Bool(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::IValue*>(self);
  return self_->toBool();
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Bool(bool self) {
  LANTERN_FUNCTION_START
  return (void*)new torch::jit::IValue(self);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_BoolList(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::IValue*>(self);
  return (void*)new std::vector<bool>(self_->toBoolList().vec());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_BoolList(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = *reinterpret_cast<std::vector<bool>*>(self);
  return (void*)new torch::IValue(self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_Device(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::IValue*>(self);
  torch::Device out = self_->toDevice();
  return make_raw::Device(out);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Device(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::Device(self);
  return (void*)new torch::IValue(self_);
  LANTERN_FUNCTION_END
}

double _lantern_IValue_Double(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::IValue*>(self);
  return self_->toDouble();
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Double(double self) {
  LANTERN_FUNCTION_START
  return (void*)new torch::IValue(self);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_DoubleList(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::IValue*>(self);
  return make_raw::vector::double_t(self_->toDoubleList().vec());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_DoubleList(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::vector::double_t(self);
  return make_raw::IValue(self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_Generator(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_raw::Generator(self_.toGenerator());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Generator(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::Generator(self);
  return make_raw::IValue(self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_GenericDict(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_ptr<c10::impl::GenericDict>(self_.toGenericDict());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_GenericDict(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<c10::impl::GenericDict*>(self);
  return make_raw::IValue(*self_);
  LANTERN_FUNCTION_END
}

int64_t _lantern_IValue_Int(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return self_.toInt();
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Int(int64_t self) {
  LANTERN_FUNCTION_START
  return make_raw::IValue(self);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_IntList(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_raw::vector::int64_t(self_.toIntList().vec());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_IntList(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::vector::int64_t(self);
  return make_raw::IValue(self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_List(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_ptr<c10::impl::GenericList>(self_.toList());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_List(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<c10::impl::GenericList*>(self);
  return make_raw::IValue(*self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_Module(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_ptr<torch::jit::script::Module>(self_.toModule());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Module(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::script::Module*>(self);
  return make_raw::IValue(self_->_ivalue());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_Scalar(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_raw::Scalar(self_.toScalar());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Scalar(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::Scalar(self);
  return make_raw::IValue(self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_String(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_raw::string(std::string(self_.toString().get()->string()));
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_String(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::string(self);
  return make_raw::IValue(self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_Tensor(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_raw::Tensor(self_.toTensor());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Tensor(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::Tensor(self);
  return make_raw::IValue(self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_TensorList(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  return make_raw::TensorList(self_.toTensorList().vec());
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_TensorList(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::TensorList(self);
  return make_raw::IValue(self_);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_None() {
  LANTERN_FUNCTION_START
  return make_raw::IValue(c10::nullopt);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_Tuple(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::IValue(self);
  auto tuple = self_.toTuple();

  std::vector<std::string> names;
  if (tuple->type()->schema()) {
    auto arguments = tuple->type()->schema()->arguments();

    for (const auto& el : arguments) {
      names.push_back(el.name());
    }
  }

  return make_ptr<NamedTupleHelper>({tuple->elements(), names});
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_Tuple(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<std::vector<torch::IValue>*>(self);
  return make_raw::IValue(c10::ivalue::Tuple::create(*self_));
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_NamedTuple(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<NamedTupleHelper*>(self);

  std::vector<c10::TypePtr> types;
  for (auto el : self_->elements) {
    types.push_back(el.type());
  }
  auto tuple = c10::ivalue::Tuple::createNamed(
      self_->elements,
      c10::TupleType::createNamed(c10::nullopt, self_->names, types));

  return make_raw::IValue(tuple);
  LANTERN_FUNCTION_END
}

void* _lantern_IValue_from_TensorDict(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::TensorDict(self);
  return make_raw::IValue(self_);
  LANTERN_FUNCTION_END
}