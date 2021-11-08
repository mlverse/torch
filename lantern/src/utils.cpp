#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_vector_int64_t(int64_t *x, size_t x_size)
{
  LANTERN_FUNCTION_START
  auto out = std::vector<int64_t>(x, x + x_size);
  return (void *)new LanternObject<std::vector<int64_t>>(out);
  LANTERN_FUNCTION_END
}

int64_t _lantern_vector_int64_t_size(void* self)
{
  LANTERN_FUNCTION_START
  return reinterpret_cast<std::vector<int64_t> *>(self)->size();
  LANTERN_FUNCTION_END
}

int64_t _lantern_vector_int64_t_at (void* self, int64_t index)
{
  return reinterpret_cast<std::vector<int64_t> *>(self)->at(index);
}

void *_lantern_vector_double(double *x, size_t x_size)
{
  LANTERN_FUNCTION_START
  auto out = std::vector<double>(x, x + x_size);
  return (void *)new LanternObject<std::vector<double>>(out);
  LANTERN_FUNCTION_END
}

void* _lantern_vector_double_new ()
{
  return (void *)new std::vector<double>();
}

void* _lantern_vector_int64_t_new ()
{
  return (void *)new std::vector<int64_t>();
}

void _lantern_vector_double_push_back(void* self, double x)
{
  reinterpret_cast<std::vector<double>*>(self)->push_back(x);
}

void _lantern_vector_int64_t_push_back(void* self, int64_t x)
{
  reinterpret_cast<std::vector<int64_t>*>(self)->push_back(x);
}

double _lantern_vector_double_size(void* self)
{
  LANTERN_FUNCTION_START
  return reinterpret_cast<std::vector<double> *>(self)->size();
  LANTERN_FUNCTION_END
}

double _lantern_vector_double_at (void* self, int64_t index)
{
  return reinterpret_cast<std::vector<double> *>(self)->at(index);
}

void * _lantern_optional_vector_double(double * x, size_t x_size, bool is_null)
{
  LANTERN_FUNCTION_START
  c10::optional<torch::ArrayRef<double>> out;
  
  if (is_null)
    out = c10::nullopt;
  else 
    out = torch::ArrayRef<double>(x, x + x_size);;
  
  return (void *)new LanternObject<c10::optional<torch::ArrayRef<double>>>(out);
  LANTERN_FUNCTION_END
}

void * _lantern_optional_vector_int64_t(int64_t * x, size_t x_size, bool is_null)
{
  LANTERN_FUNCTION_START
  c10::optional<torch::ArrayRef<int64_t>> out;

  if (is_null)
    out = c10::nullopt;
  else 
    out = torch::ArrayRef<int64_t>(x, x + x_size);

  return (void *)new LanternObject<c10::optional<torch::ArrayRef<int64_t>>>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_IntArrayRef(int64_t *x, size_t x_size)
{
  LANTERN_FUNCTION_START
  torch::IntArrayRef out = std::vector<int64_t>(x, x + x_size);
  return (void *)new LanternObject<torch::IntArrayRef>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_int(int x)
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<int>(x);
  LANTERN_FUNCTION_END
}

void *_lantern_int64_t(int64_t x)
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<int64_t>(x);
  LANTERN_FUNCTION_END
}

void *_lantern_double(double x)
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<double>(x);
  LANTERN_FUNCTION_END
}

void *_lantern_optional_double(double x, bool is_null)
{
  LANTERN_FUNCTION_START
  c10::optional<double> out;
  if (is_null)
    out = c10::nullopt;
  else
    out = x;

  return (void *)new LanternObject<c10::optional<double>>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_optional_int64_t(int64_t x, bool is_null)
{
  LANTERN_FUNCTION_START
  c10::optional<int64_t> out;
  if (is_null)
    out = c10::nullopt;
  else
    out = x;

  return (void *)new LanternObject<c10::optional<int64_t>>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_bool(bool x)
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<bool>(x);
  LANTERN_FUNCTION_END
}

void *_lantern_vector_get(void *x, int i)
{
  LANTERN_FUNCTION_START
  auto v = reinterpret_cast<LanternObject<std::vector<void *>> *>(x)->get();
  return v.at(i);
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_undefined()
{
  LANTERN_FUNCTION_START
  torch::Tensor x = {};
  return make_unique::Tensor(x);
  LANTERN_FUNCTION_END
}

void *_lantern_vector_string_new()
{
  LANTERN_FUNCTION_START
  return (void *)new std::vector<std::string>();
  LANTERN_FUNCTION_END
}

void _lantern_vector_string_push_back(void *self, const char *x)
{
  LANTERN_FUNCTION_START
  reinterpret_cast<std::vector<std::string> *>(self)->push_back(std::string(x));
  LANTERN_FUNCTION_END_VOID
}

int64_t _lantern_vector_string_size(void *self)
{
  LANTERN_FUNCTION_START
  return reinterpret_cast<std::vector<std::string> *>(self)->size();
  LANTERN_FUNCTION_END_RET(0)
}

const char *_lantern_vector_string_at(void *self, int64_t i)
{
  LANTERN_FUNCTION_START
  auto str = reinterpret_cast<std::vector<std::string> *>(self)->at(i);
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

void *_lantern_vector_bool_new()
{
  LANTERN_FUNCTION_START
  return (void *)new std::vector<bool>();
  LANTERN_FUNCTION_END
}

void _lantern_vector_bool_push_back(void *self, bool x)
{
  LANTERN_FUNCTION_START
  reinterpret_cast<std::vector<bool> *>(self)->push_back(x);
  LANTERN_FUNCTION_END_VOID
}

int64_t _lantern_vector_bool_size(void *self)
{
  LANTERN_FUNCTION_START
  return reinterpret_cast<std::vector<bool> *>(self)->size();
  LANTERN_FUNCTION_END_RET(0)
}

bool _lantern_vector_bool_at(void *self, int64_t i)
{
  LANTERN_FUNCTION_START
  return reinterpret_cast<std::vector<bool> *>(self)->at(i);
  LANTERN_FUNCTION_END_RET(false)
}

void * _lantern_string_new (const char * value)
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<std::string>(std::string(value));
  LANTERN_FUNCTION_END
}

const char * _lantern_string_get (void* self)
{
  auto str = *reinterpret_cast<std::string*>(self);
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
}

void _lantern_print_stuff (void* x)
{
  LANTERN_FUNCTION_START
  auto v = reinterpret_cast<LanternPtr<c10::optional<torch::Device>>*>(x);
  std::cout << v->get().value().type() << std::endl;
  LANTERN_FUNCTION_END_VOID
}

void lantern_host_handler() {}
bool lantern_loaded = false;
void check_lantern_loaded() {}

void* _lantern_nn_functional_pad_circular (void* input, void* padding)
{
  LANTERN_FUNCTION_START
  auto input_ = from_raw::Tensor(input);
  auto padding_ = reinterpret_cast<LanternObject<std::vector<int64_t>>*>(padding)->get();
  auto out = torch::nn::functional::_pad_circular(input_, padding_);
  return make_unique::Tensor(out);
  LANTERN_FUNCTION_END
}

template<class T>
void* make_ptr (const T& x) {
  return (void*) std::make_unique<T>(x).release();
}

namespace make_unique {
  void* Tensor (const torch::Tensor& x)
  {
    return make_ptr<torch::Tensor>(x);
  }
}

#define LANTERN_FROM_RAW(name, type) \
  type& name(void* x) {return reinterpret_cast<LanternObject<type>*>(x)->get();}

namespace from_raw {
  LANTERN_FROM_RAW(Tensor, torch::Tensor)
}

