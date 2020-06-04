#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_from_blob(void *data, int64_t *sizes, size_t sizes_size, void *options)
{
  return (void *)new LanternObject<torch::Tensor>(torch::from_blob(
      data,
      std::vector<int64_t>(sizes, sizes + sizes_size),
      reinterpret_cast<LanternObject<torch::TensorOptions> *>(options)->get()));
}

const char *lantern_Tensor_StreamInsertion(void *x)
{
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
}

void *lantern_Tensor_clone(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return (void *)new LanternObject<torch::Tensor>(x.clone());
}

void *lantern_Tensor_permute(void *self, void *dims)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  std::vector<int64_t> y = reinterpret_cast<LanternObject<std::vector<int64_t>> *>(dims)->get();
  return (void *)new LanternObject<torch::Tensor>(x.permute(y));
}

void *lantern_Tensor_contiguous(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return (void *)new LanternObject<torch::Tensor>(x.contiguous());
}

void *lantern_Tensor_to(void *self, void *options)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  torch::TensorOptions y = reinterpret_cast<LanternObject<torch::TensorOptions> *>(options)->get();
  return (void *)new LanternObject<torch::Tensor>(x.to(y));
}

void *lantern_Tensor_set_requires_grad(void *self, bool requires_grad)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return (void *)new LanternObject<torch::Tensor>(x.set_requires_grad(requires_grad));
}

double *lantern_Tensor_data_ptr_double(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<double>();
}

uint8_t *lantern_Tensor_data_ptr_uint8_t(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<uint8_t>();
}

int32_t *lantern_Tensor_data_ptr_int32_t(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<int32_t>();
}

int16_t *lantern_Tensor_data_ptr_int16_t(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<int16_t>();
}

bool *lantern_Tensor_data_ptr_bool(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.data_ptr<bool>();
}

int64_t lantern_Tensor_numel(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.numel();
}

int64_t lantern_Tensor_ndimension(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.ndimension();
}

int64_t lantern_Tensor_size(void *self, int64_t i)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  return x.size(i);
}

void *lantern_Tensor_dtype(void *self)
{
  torch::Tensor x = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();
  torch::Dtype dtype = c10::typeMetaToScalarType(x.dtype());
  return (void *)new LanternObject<torch::Dtype>(dtype);
}
