#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_from_blob(void *data, int64_t *sizes, size_t sizes_size, void *options)
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::Tensor>(torch::from_blob(
      data,
      std::vector<int64_t>(sizes, sizes + sizes_size),
      reinterpret_cast<LanternObject<torch::TensorOptions> *>(options)->get()));
  LANTERN_FUNCTION_END
}

const char *_lantern_Tensor_StreamInsertion(void *x)
{
  LANTERN_FUNCTION_START
  std::stringstream ss;

  auto tensor = reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get();

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
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return (void *)new LanternObject<torch::Tensor>(x.clone());
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_permute(void *self, void *dims)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  std::vector<int64_t> y = reinterpret_cast<LanternObject<std::vector<int64_t>> *>(dims)->get();
  return (void *)new LanternObject<torch::Tensor>(x.permute(y));
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_contiguous(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return (void *)new LanternObject<torch::Tensor>(x.contiguous());
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_to(void *self, void *options)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  torch::TensorOptions y = reinterpret_cast<LanternObject<torch::TensorOptions> *>(options)->get();
  return (void *)new LanternObject<torch::Tensor>(x.to(y));
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_set_requires_grad(void *self, bool requires_grad)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return (void *)new LanternObject<torch::Tensor>(x.set_requires_grad(requires_grad));
  LANTERN_FUNCTION_END
}

double *_lantern_Tensor_data_ptr_double(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<double>();
  LANTERN_FUNCTION_END
}

uint8_t *_lantern_Tensor_data_ptr_uint8_t(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<uint8_t>();
  LANTERN_FUNCTION_END
}

int64_t *_lantern_Tensor_data_ptr_int64_t(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<int64_t>();
  LANTERN_FUNCTION_END
}

int32_t *_lantern_Tensor_data_ptr_int32_t(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<int32_t>();
  LANTERN_FUNCTION_END
}

int16_t *_lantern_Tensor_data_ptr_int16_t(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<int16_t>();
  LANTERN_FUNCTION_END
}

bool *_lantern_Tensor_data_ptr_bool(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<bool>();
  LANTERN_FUNCTION_END
}

int64_t _lantern_Tensor_numel(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.numel();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_element_size(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.element_size();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_ndimension(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.ndimension();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_size(void *self, int64_t i)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.size(i);
  LANTERN_FUNCTION_END_RET(0)
}

void *_lantern_Tensor_dtype(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  torch::Dtype dtype = c10::typeMetaToScalarType(x.dtype());
  return (void *)new LanternObject<torch::Dtype>(dtype);
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_device(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  torch::Device device = x.device();
  return (void *)new LanternPtr<torch::Device>(device);
  LANTERN_FUNCTION_END
}

bool _lantern_Tensor_is_undefined(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.dtype() == torch::ScalarType::Undefined;
  LANTERN_FUNCTION_END_RET(false)
}

bool _lantern_Tensor_is_contiguous(void *self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.is_contiguous();
  LANTERN_FUNCTION_END_RET(false)
}

bool _lantern_Tensor_has_names (void * self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.has_names();
  LANTERN_FUNCTION_END_RET(false)
}

void* _lantern_Tensor_names (void* self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  std::vector<torch::Dimname> nms = x.names().vec();
  return (void *)new LanternPtr<std::vector<torch::Dimname>>(nms);
  LANTERN_FUNCTION_END
}

// an utility function to quickly check if a tensor has any zeros
bool _lantern_Tensor_has_any_zeros (void * self)
{
  LANTERN_FUNCTION_START
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return (x == 0).any().item().toBool();
  LANTERN_FUNCTION_END
}