#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include "utils.hpp"


int _lantern_IValue_type (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    if (self_->isBlob()) return IValue_types::IValueBlobType;
    if (self_->isBool()) return IValue_types::IValueBoolType;
    if (self_->isBoolList()) return IValue_types::IValueBoolListType;
    if (self_->isCapsule()) return IValue_types::IValueCapsuleType;
    if (self_->isComplexDouble()) return IValue_types::IValueComplexDoubleType;
    if (self_->isComplexDoubleList()) return IValue_types::IValueComplexDoubleListType;
    if (self_->isCustomClass()) return IValue_types::IValueCustomClassType;
    if (self_->isDevice()) return IValue_types::IValueDeviceType;
    if (self_->isDouble()) return IValue_types::IValueDoubleType;
    if (self_->isDoubleList()) return IValue_types::IValueDoubleListType;
    if (self_->isNone()) return IValue_types::IValueNoneType;
    if (self_->isEnum()) return IValue_types::IValueEnumType;
    if (self_->isFuture()) return IValue_types::IValueFutureType;
    if (self_->isGenerator()) return IValue_types::IValueGeneratorType;
    if (self_->isGenericDict()) return IValue_types::IValueGenericDictType;
    if (self_->isInt()) return IValue_types::IValueIntType;
    if (self_->isIntList()) return IValue_types::IValueIntListType;
    if (self_->isList()) return IValue_types::IValueListType;
    if (self_->isModule()) return IValue_types::IValueModuleType;
    if (self_->isObject()) return IValue_types::IValueObjectType;
    if (self_->isPtrType()) return IValue_types::IValuePtrTypeType;
    if (self_->isPyObject()) return IValue_types::IValuePyObjectType;
    if (self_->isQuantizer()) return IValue_types::IValueQuantizerType;
    if (self_->isRRef()) return IValue_types::IValueRRefType;
    if (self_->isScalar()) return IValue_types::IValueScalarType;
    if (self_->isStorage()) return IValue_types::IValueStorageType;
    if (self_->isStream()) return IValue_types::IValueStreamType;
    if (self_->isString()) return IValue_types::IValueStringType;
    if (self_->isTensor()) return IValue_types::IValueTensorType;
    if (self_->isTensorList()) return IValue_types::IValueTensorListType;
    if (self_->isTuple()) return IValue_types::IValueTupleType;
    
    return IValue_types::IValueTypeUnknownType;
}

bool _lantern_IValue_Bool (void * self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return self_->toBool();
}

void* _lantern_IValue_from_Bool (bool self)
{
    return (void*) new torch::IValue(self);
}

void* _lantern_IValue_BoolList (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void*) new std::vector<bool>(self_->toBoolList().vec());
}

void* _lantern_IValue_from_BoolList (void* self)
{
    auto self_ = *reinterpret_cast<std::vector<bool>*>(self);
    return (void*) new torch::IValue(self_);
}

void* _lantern_IValue_Device (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    torch::Device out = self_->toDevice();
    return (void*) new LanternPtr<torch::Device>(out);
}

void* _lantern_IValue_from_Device (void* self)
{
    auto self_ = reinterpret_cast<LanternPtr<torch::Device>*>(self);
    return (void*) new torch::IValue(self_->get());
}

double _lantern_IValue_Double (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return self_->toDouble();
}

void* _lantern_IValue_from_Double (double self)
{
    return (void*) new torch::IValue(self);
}

void* _lantern_IValue_DoubleList (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void *)new LanternObject<std::vector<double>>(self_->toDoubleList().vec());
}

void* _lantern_IValue_from_DoubleList (void* self)
{
    auto self_ = reinterpret_cast<LanternObject<std::vector<double>>*>(self);
    return (void*) new torch::IValue(self_->get());
}

void* _lantern_IValue_Generator (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void *)new LanternObject<torch::Generator>(self_->toGenerator());
}

void* _lantern_IValue_from_Generator (void* self)
{
    auto self_ = reinterpret_cast<LanternObject<torch::Generator>*>(self);
    return (void*) new torch::IValue(self_);
}

void* _lantern_IValue_GenericDict (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void*) new c10::impl::GenericDict(self_->toGenericDict());
}

void* _lantern_IValue_from_GenericDict (void* self)
{
    auto self_ = reinterpret_cast<c10::impl::GenericDict*>(self);
    return (void*) new torch::IValue(*self_);
}

int64_t _lantern_IValue_Int (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return self_->toInt();
}

void* _lantern_IValue_from_Int (int64_t self)
{
    return (void*) new torch::IValue(self);
}

void* _lantern_IValue_IntList (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void *)new LanternObject<std::vector<int64_t>>(self_->toIntList().vec());
}

void* _lantern_IValue_from_IntList (void* self)
{
    auto self_ = reinterpret_cast<LanternObject<std::vector<int64_t>>*>(self);
    return (void*) new torch::IValue(self_->get());
}

void* _lantern_IValue_List (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void*) new c10::impl::GenericList(self_->toList());
}

void* _lantern_IValue_from_List (void* self)
{
    auto self_ = reinterpret_cast<c10::impl::GenericList*>(self);
    return (void*) new torch::IValue(self_);
}

void* _lantern_IValue_Module (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void*) new torch::jit::script::Module(self_->toModule());
}

void* _lantern_IValue_from_Module (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::script::Module *>(self);
    return (void*) new torch::IValue(self_->_ivalue());
}

void* _lantern_IValue_Scalar (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void*) new LanternObject<torch::Scalar>(self_->toScalar());
}

void* _lantern_IValue_from_Scalar (void* self)
{
    auto self_ = reinterpret_cast<LanternObject<torch::Scalar>*>(self);
    return (void*) new torch::IValue(self_->get());
}

void* _lantern_IValue_String (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void *)new LanternObject<std::string>(std::string(self_->toString().get()->string()));
}

void* _lantern_IValue_from_String (void* self)
{
    auto self_ = reinterpret_cast<LanternObject<std::string>*>(self);
    return (void*) new torch::IValue(self_->get());
}

void* _lantern_IValue_Tensor (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void *)new LanternObject<torch::Tensor>(self_->toTensor());
}

void* _lantern_IValue_from_Tensor (void* self)
{
    auto self_ = reinterpret_cast<LanternObject<torch::Tensor>*>(self);
    return (void*) new torch::IValue(self_->get());
}

void* _lantern_IValue_TensorList (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void *)new LanternObject<std::vector<torch::Tensor>>(self_->toTensorList().vec());
}

void* _lantern_IValue_from_TensorList (void* self)
{
    auto self_ = reinterpret_cast<LanternObject<std::vector<torch::Tensor>>*>(self);
    return (void*) new torch::IValue(self_->get());
}

void* _lantern_IValue_Tuple (void* self)
{
    auto self_ = reinterpret_cast<torch::jit::IValue *>(self);
    return (void *)new c10::ivalue::Tuple(*self_->toTuple().get());
}