#include <iostream>
#include <regex>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void *_lantern_from_blob(void *data, int64_t *sizes, size_t sizes_size,
                         int64_t *strides, size_t strides_size, void *options) {
  LANTERN_FUNCTION_START
  return make_raw::Tensor(
      torch::from_blob(data, std::vector<int64_t>(sizes, sizes + sizes_size),
                       std::vector<int64_t>(strides, strides + strides_size),
                       from_raw::TensorOptions(options)));
  LANTERN_FUNCTION_END
}

const char *_lantern_Tensor_StreamInsertion(void *x) {
  LANTERN_FUNCTION_START
  std::stringstream ss;

  auto tensor = from_raw::Tensor(x);

  // the stream insertion method for quantized tensors does not
  // exist so we dequantize before printing.
  if (tensor.is_quantized()) {
    tensor = tensor.dequantize();
  }

  // the stream insertion method seems to cast tensors to float64
  // before printing and that's not supported by the MPS device
  // thus we first cast to CPU, print and later do a regex replace
  // to change the device to MPS.
  auto is_mps = (tensor.dtype() != torch::ScalarType::Undefined);
  if (is_mps) {
    is_mps = (tensor.device().is_mps());
  }
  if (is_mps) {
    tensor = tensor.cpu();
  }

  ss << tensor;
  std::string str = ss.str();
  
  if (is_mps) {
    str = std::regex_replace(str, std::regex("CPU"), "MPS");
  }
  
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_clone(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  return make_raw::Tensor(x.clone());
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_permute(void *self, void *dims) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  auto y = from_raw::vector::int64_t(dims);
  return make_raw::Tensor(x.permute(y));
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_contiguous(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return make_raw::Tensor(x.contiguous());
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_to(void *self, void *options) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  auto y = from_raw::TensorOptions(options);
  return make_raw::Tensor(x.to(y));
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_set_requires_grad(void *self, bool requires_grad) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return make_raw::Tensor(x.set_requires_grad(requires_grad));
  LANTERN_FUNCTION_END
}

double *_lantern_Tensor_data_ptr_double(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.data_ptr<double>();
  LANTERN_FUNCTION_END
}

uint8_t *_lantern_Tensor_data_ptr_uint8_t(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.data_ptr<uint8_t>();
  LANTERN_FUNCTION_END
}

int64_t *_lantern_Tensor_data_ptr_int64_t(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.data_ptr<int64_t>();
  LANTERN_FUNCTION_END
}

int32_t *_lantern_Tensor_data_ptr_int32_t(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.data_ptr<int32_t>();
  LANTERN_FUNCTION_END
}

int16_t *_lantern_Tensor_data_ptr_int16_t(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.data_ptr<int16_t>();
  LANTERN_FUNCTION_END
}

bool *_lantern_Tensor_data_ptr_bool(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.data_ptr<bool>();
  LANTERN_FUNCTION_END
}

int64_t _lantern_Tensor_numel(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.numel();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_element_size(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.element_size();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_ndimension(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.ndimension();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_Tensor_size(void *self, int64_t i) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.size(i);
  LANTERN_FUNCTION_END_RET(0)
}

void *_lantern_Tensor_dtype(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  torch::Dtype dtype = c10::typeMetaToScalarType(x.dtype());
  return make_raw::Dtype(dtype);
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_device(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  torch::Device device = x.device();
  return make_raw::Device(device);
  LANTERN_FUNCTION_END
}

bool _lantern_Tensor_is_undefined(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.dtype() == torch::ScalarType::Undefined;
  LANTERN_FUNCTION_END_RET(false)
}

bool _lantern_Tensor_is_contiguous(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.is_contiguous();
  LANTERN_FUNCTION_END_RET(false)
}

bool _lantern_Tensor_has_names(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return x.has_names();
  LANTERN_FUNCTION_END_RET(false)
}

void *_lantern_Tensor_names(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  return make_raw::DimnameList(x.names());
  LANTERN_FUNCTION_END
}

// an utility function to quickly check if a tensor has any zeros
bool _lantern_Tensor_has_any_zeros(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  ;
  return (x == 0).any().item().toBool();
  LANTERN_FUNCTION_END
}

void *_lantern_normal_double_double_intarrayref_generator_tensoroptions(
    double mean, double std, void *size, void *generator, void *options) {
  LANTERN_FUNCTION_START
  auto size_ = from_raw::IntArrayRef(size);
  auto generator_ = from_raw::Generator(generator);
  auto options_ = from_raw::TensorOptions(options);
  auto ten = at::normal(mean, std, size_, generator_, options_);
  return make_raw::Tensor(ten);
  LANTERN_FUNCTION_END
}

void *_lantern_normal_tensor_tensor_generator(void *mean, void *std,
                                              void *generator) {
  LANTERN_FUNCTION_START
  auto mean_ = from_raw::Tensor(mean);
  auto std_ = from_raw::Tensor(std);
  auto generator_ = from_raw::Generator(generator);
  auto ten = at::normal(mean_, std_, generator_);
  return make_raw::Tensor(ten);
  LANTERN_FUNCTION_END
}

void *_lantern_normal_double_tensor_generator(double mean, void *std,
                                              void *generator) {
  LANTERN_FUNCTION_START
  auto std_ = from_raw::Tensor(std);
  auto generator_ = from_raw::Generator(generator);
  auto ten = at::normal(mean, std_, generator_);
  return make_raw::Tensor(ten);
  LANTERN_FUNCTION_END
}

void *_lantern_normal_tensor_double_generator(void *mean, double std,
                                              void *generator) {
  LANTERN_FUNCTION_START
  auto mean_ = from_raw::Tensor(mean);
  auto generator_ = from_raw::Generator(generator);
  auto ten = at::normal(mean_, std, generator_);
  return make_raw::Tensor(ten);
  LANTERN_FUNCTION_END
}

std::string lantern_name_sig(const c10::impl::PyInterpreter *x) {
  return "LanternInterpreter";
}

bool lantern_is_contiguous_sig (const c10::impl::PyInterpreter*, const at::TensorImpl*) {
  return true;
}

// global interpreter definition.
// unlike in python, we currently don't have behaviors where a tensor
// can be owned by different interpreters.

c10::impl::PyInterpreter lantern_interpreter(*lantern_name_sig, nullptr,
                                             nullptr, nullptr, *lantern_is_contiguous_sig);

void _lantern_tensor_set_pyobj(void *x, void *ptr) {
  LANTERN_FUNCTION_START
  PyObject *ptr_ = reinterpret_cast<PyObject *>(ptr);
  auto t = from_raw::Tensor(x);
  t.unsafeGetTensorImpl()->init_pyobj(
      &lantern_interpreter, ptr_,
      c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_tensor_get_pyobj(void *x) {
  LANTERN_FUNCTION_START
  auto t = from_raw::Tensor(x);
  auto pyobj = t.unsafeGetTensorImpl()->check_pyobj(&lantern_interpreter);
  if (pyobj.has_value()) {
    return (void *)pyobj.value();
  } else {
    return nullptr;
  }
  LANTERN_FUNCTION_END
}

bool _lantern_Tensor_is_sparse (void* x) {
  LANTERN_FUNCTION_START
  auto t = from_raw::Tensor(x);
  return t.is_sparse();
  LANTERN_FUNCTION_END_RET(false)
}