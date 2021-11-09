#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_from_blob(void *data, int64_t *sizes, size_t sizes_size, void *options)
{
  LANTERN_FUNCTION_START
  return make_unique::Tensor(torch::from_blob(
      data,
      std::vector<int64_t>(sizes, sizes + sizes_size),
      from_raw::TensorOptions(options)));
  LANTERN_FUNCTION_END
}

const char *_lantern_Tensor_StreamInsertion(void *x)
{
  LANTERN_FUNCTION_START
  std::stringstream ss;

  auto tensor = from_raw::Tensor(x);

  // the stream insertion method for quantized tensors does not
  // exist so we dequantize before printing.
  if (tensor.is_quantized())
  {
    tensor = tensor.dequantize();
  }

  ss << tensor;
  std::string str = ss.str();
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_clone(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  return make_unique::Tensor(x.clone());
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_permute(void *self, void *dims)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  std::vector<int64_t> y = reinterpret_cast<LanternObject<std::vector<int64_t>> *>(dims)->get();
  return make_unique::Tensor(x.permute(y));
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_contiguous(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return make_unique::Tensor(x.contiguous());
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_to(void *self, void *options)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  auto y = from_raw::TensorOptions(options);
  return make_unique::Tensor(x.to(y));
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_set_requires_grad(void *self, bool requires_grad)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return make_unique::Tensor(x.set_requires_grad(requires_grad));
  LANTERN_FUNCTION_END
}

double *_lantern_Tensor_data_ptr_double(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.data_ptr<double>();
  LANTERN_FUNCTION_END
}

uint8_t *_lantern_Tensor_data_ptr_uint8_t(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.data_ptr<uint8_t>();
  LANTERN_FUNCTION_END
}

int64_t *_lantern_Tensor_data_ptr_int64_t(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.data_ptr<int64_t>();
  LANTERN_FUNCTION_END
}

int32_t *_lantern_Tensor_data_ptr_int32_t(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.data_ptr<int32_t>();
  LANTERN_FUNCTION_END
}

int16_t *_lantern_Tensor_data_ptr_int16_t(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.data_ptr<int16_t>();
  LANTERN_FUNCTION_END
}

bool *_lantern_Tensor_data_ptr_bool(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.data_ptr<bool>();
  LANTERN_FUNCTION_END
}

int64_t _lantern_Tensor_numel(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.numel();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_element_size(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.element_size();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_ndimension(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.ndimension();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_size(void *self, int64_t i)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.size(i);
  LANTERN_FUNCTION_END_RET(0)
}

void *_lantern_Tensor_dtype(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  torch::Dtype dtype = c10::typeMetaToScalarType(x.dtype());
  return (void *)new LanternObject<torch::Dtype>(dtype);
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_device(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  torch::Device device = x.device();
  return make_unique::Device(device);
  LANTERN_FUNCTION_END
}

bool _lantern_Tensor_is_undefined(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.dtype() == torch::ScalarType::Undefined;
  LANTERN_FUNCTION_END_RET(false)
}

bool _lantern_Tensor_is_contiguous(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.is_contiguous();
  LANTERN_FUNCTION_END_RET(false)
}

bool _lantern_Tensor_has_names (void * self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return x.has_names();
  LANTERN_FUNCTION_END_RET(false)
}

void* _lantern_Tensor_names (void* self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  std::vector<torch::Dimname> nms = x.names().vec();
  return (void *)new LanternPtr<std::vector<torch::Dimname>>(nms);
  LANTERN_FUNCTION_END
}

// an utility function to quickly check if a tensor has any zeros
bool _lantern_Tensor_has_any_zeros (void * self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);;
  return (x == 0).any().item().toBool();
  LANTERN_FUNCTION_END
}

void* _lantern_normal_double_double_intarrayref_generator_tensoroptions (double mean, double std, void* size, void* generator, void* options)
{
  LANTERN_FUNCTION_START
  auto size_ = reinterpret_cast<LanternObject<std::vector<int64_t>>*>(size)->get();
  auto generator_ = reinterpret_cast<LanternObject<torch::Generator>*>(generator)->get();
  auto options_ = from_raw::TensorOptions(options);
  auto ten = at::normal(mean, std, size_, generator_, options_);
  return make_unique::Tensor(ten);
  LANTERN_FUNCTION_END
}

void* _lantern_normal_tensor_tensor_generator (void* mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
  auto mean_ = from_raw::Tensor(mean);
  auto std_ = from_raw::Tensor(std);
  auto generator_ = reinterpret_cast<LanternObject<torch::Generator>*>(generator)->get();
  auto ten = at::normal(mean_, std_, generator_);
  return make_unique::Tensor(ten);
  LANTERN_FUNCTION_END
}

void* _lantern_normal_double_tensor_generator (double mean, void* std, void* generator)
{
  LANTERN_FUNCTION_START
  auto std_ = from_raw::Tensor(std);
  auto generator_ = reinterpret_cast<LanternObject<torch::Generator>*>(generator)->get();
  auto ten = at::normal(mean, std_, generator_);
  return make_unique::Tensor(ten);
  LANTERN_FUNCTION_END
}

void* _lantern_normal_tensor_double_generator (void* mean, double std, void* generator)
{
  LANTERN_FUNCTION_START
  auto mean_ = from_raw::Tensor(mean);
  auto generator_ = reinterpret_cast<LanternObject<torch::Generator>*>(generator)->get();
  auto ten = at::normal(mean_, std, generator_);
  return make_unique::Tensor(ten);
  LANTERN_FUNCTION_END
}

void *_lantern_optional_tensor(void* x, bool is_null)
{
  LANTERN_FUNCTION_START
  c10::optional<torch::Tensor> out;
  if (is_null)
    out = c10::nullopt;
  else
  {
    torch::Tensor value = from_raw::Tensor(x);
    out = value;
  }
    

  return (void *)new LanternObject<c10::optional<torch::Tensor>>(out);
  LANTERN_FUNCTION_END
}

bool _lantern_optional_tensor_has_value (void*x)
{
  auto value = reinterpret_cast<LanternObject<c10::optional<torch::Tensor>>*>(x)->get();
  return value.has_value();
}

void* _lantern_optional_tensor_value (void* x)
{
  auto value = reinterpret_cast<LanternObject<c10::optional<torch::Tensor>>*>(x)->get();
  return make_unique::Tensor(value.value());
}

void _lantern_tensor_set_pyobj (void*x, void* ptr)
{
  LANTERN_FUNCTION_START
  PyObject * ptr_ = reinterpret_cast<PyObject*>(ptr);
  auto t = from_raw::Tensor(x);
  t.unsafeGetTensorImpl()->set_pyobj(ptr_);
  LANTERN_FUNCTION_END_VOID
}

void* _lantern_tensor_get_pyobj (void* x)
{
  LANTERN_FUNCTION_START
  auto t = from_raw::Tensor(x);
  return t.unsafeGetTensorImpl()->pyobj();
  LANTERN_FUNCTION_END
}