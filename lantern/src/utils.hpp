#pragma once

template <class T>
class LanternObject
{
private:
  T _object;

public:
  LanternObject(T object) : _object(std::forward<T>(object))
  {
  }

  LanternObject()
  {
  }

  T &get()
  {
    return _object;
  }
};

template <class T>
class LanternPtr
{
private:
  T *_object;

public:
  LanternPtr(const T &object)
  {
    _object = new T(object);
  }

  LanternPtr()
  {
    _object = new T;
  }

  ~LanternPtr()
  {
    delete _object;
    _object = NULL;
  }

  T &get()
  {
    return *_object;
  }

};

// https://pt.stackoverflow.com/a/438284/6036
//
template <class... T>
std::vector<void *> to_vector(std::tuple<T...> x)
{
  std::vector<void *> out;
  out.reserve(sizeof...(T));
  std::apply([&out](auto &&... args) {
    ((out.push_back(new std::remove_reference_t<decltype(args)>(std::forward<decltype(args)>(args)))), ...);
  },
             x);
  return out;
}

template <class T>
auto optional(void *x)
{

  if (x == nullptr)
  {
    return std::make_shared<LanternObject<c10::optional<T>>>(c10::nullopt);
  } 

  auto z = ((LanternObject<T> *)x)->get();
  return std::make_shared<LanternObject<c10::optional<T>>>(z);
}

struct NamedTupleHelper {
    std::vector<torch::IValue> elements;
    std::vector<std::string> names;
};

namespace make_unique {
  void* Tensor (const torch::Tensor& x);
  void* TensorList (const torch::TensorList& x);
  void* ScalarType (const torch::ScalarType& x);
  void* Scalar (const torch::Scalar& x);
  void* TensorOptions (const torch::TensorOptions& x);
  void* Device (torch::Device& x);
  void* Dtype (const torch::Dtype& x);
  void* Dimname (torch::Dimname& x);
  void* DimnameList (const torch::DimnameList& x);
  void* Generator (const torch::Generator& x);
  void* MemoryFormat (const torch::MemoryFormat& x);
  void* IntArrayRef (const torch::IntArrayRef& x);
}

#define LANTERN_FROM_RAW_DECL(name, type)                                                 \
  type& name (void* x);                 

namespace from_raw {
  LANTERN_FROM_RAW_DECL(Tensor, torch::Tensor)
  // TensorLists are passed as std::vector<torch::Tensor> because they don't own the 
  // underlying memory. Passing them as vectors is also fine as they are trivially
  // constructed from them. 
  LANTERN_FROM_RAW_DECL(TensorList, std::vector<torch::Tensor>)
  LANTERN_FROM_RAW_DECL(ScalarType, torch::ScalarType)
  LANTERN_FROM_RAW_DECL(Scalar, torch::Scalar)
  LANTERN_FROM_RAW_DECL(TensorOptions, torch::TensorOptions)
  LANTERN_FROM_RAW_DECL(Device, torch::Device)
  LANTERN_FROM_RAW_DECL(Dtype, torch::Dtype)
  LANTERN_FROM_RAW_DECL(Dimname, torch::Dimname)
  LANTERN_FROM_RAW_DECL(DimnameList, std::vector<torch::Dimname>)
  LANTERN_FROM_RAW_DECL(Generator, torch::Generator)
  LANTERN_FROM_RAW_DECL(MemoryFormat, torch::MemoryFormat)
  LANTERN_FROM_RAW_DECL(IntArrayRef, std::vector<int64_t>)

  namespace optional {
    c10::optional<torch::DimnameList> DimnameList (void* x);
    c10::optional<torch::Generator> Generator (void* x);
  }
}
