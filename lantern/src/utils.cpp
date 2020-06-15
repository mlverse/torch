#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_vector_int64_t(int64_t *x, size_t x_size)
{
  auto out = std::vector<int64_t>(x, x + x_size);
  return (void *)new LanternObject<std::vector<int64_t>>(out);
}

void *lantern_IntArrayRef(int64_t *x, size_t x_size)
{
  torch::IntArrayRef out = std::vector<int64_t>(x, x + x_size);
  return (void *)new LanternObject<torch::IntArrayRef>(out);
}

void *lantern_int(int x)
{
  return (void *)new LanternObject<int>(x);
}

void *lantern_int64_t(int64_t x)
{
  return (void *)new LanternObject<int64_t>(x);
}

void *lantern_double(double x)
{
  return (void *)new LanternObject<double>(x);
}

void *lantern_optional_double(double x, bool is_null)
{
  c10::optional<double> out;
  if (is_null)
    out = NULL;
  else
    out = x;

  return (void *)new LanternObject<c10::optional<double>>(out);
}

void *lantern_optional_int64_t(int64_t x, bool is_null)
{
  c10::optional<int64_t> out;
  if (is_null)
    out = c10::nullopt;
  else
    out = x;

  return (void *)new LanternObject<c10::optional<int64_t>>(out);
}

void *lantern_bool(bool x)
{
  return (void *)new LanternObject<bool>(x);
}

void *lantern_vector_get(void *x, int i)
{
  auto v = reinterpret_cast<LanternObject<std::vector<void *>> *>(x)->get();
  return v.at(i);
}

void *lantern_Tensor_undefined()
{
  torch::Tensor x = {};
  return (void *)new LanternObject<torch::Tensor>(x);
}

void *lantern_vector_string_new()
{
  return (void *)new std::vector<std::string>();
}

void lantern_vector_string_push_back(void *self, const char *x)
{
  reinterpret_cast<std::vector<std::string> *>(self)->push_back(std::string(x));
}

int64_t lantern_vector_string_size(void *self)
{
  return reinterpret_cast<std::vector<std::string> *>(self)->size();
}

const char *lantern_vector_string_at(void *self, int64_t i)
{
  return reinterpret_cast<std::vector<std::string> *>(self)->at(i).c_str();
}

void *lantern_vector_bool_new()
{
  return (void *)new std::vector<bool>();
}

void lantern_vector_bool_push_back(void *self, bool x)
{
  reinterpret_cast<std::vector<bool> *>(self)->push_back(x);
}

int64_t lantern_vector_bool_size(void *self)
{
  return reinterpret_cast<std::vector<bool> *>(self)->size();
}

bool lantern_vector_bool_at(void *self, int64_t i)
{
  return reinterpret_cast<std::vector<bool> *>(self)->at(i);
}
