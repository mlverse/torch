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

template<class T>
void* make_ptr (const T& x) {
  return (void*) std::make_unique<T>(x).release();
}

struct NamedTupleHelper {
    std::vector<torch::IValue> elements;
    std::vector<std::string> names;
};


// a template vector that can be easily converted to array<>
// with a few default casting for a few different sizes.
template<typename T>
class Vector {
  public:
    Vector(std::vector<T> x) : x_(x) {}
    std::vector<T> x_;
    operator std::array<T,2>() const {
      return std::array<T,2>{x_[0], x_[1]};
    }
    operator std::array<T,3>() const {
      return std::array<T,3>{x_[0], x_[1], x_[2]};
    }
    operator std::array<T,4>() const {
      return std::array<T,4>{x_[0], x_[1], x_[2], x_[3]};
    }
    operator std::vector<T>() const {
      return x_;
    }
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
  void* IntArrayRef (const torch::IntArrayRef& x);
  void* TensorDict (const c10::Dict<std::string,torch::Tensor>& x);
  void* CompilationUnit (torch::jit::CompilationUnit& x);
  void* QScheme (const torch::QScheme& x);
  void* variable_list (const torch::autograd::variable_list& x);
  void* Layout (const torch::Layout& x);
  void* Storage (const torch::Storage& x);
  void* string (const std::string& x);
  void* int64_t (const std::int64_t& x);
  void* bool_t (const bool& x);
  void* double_t (const double& x);

  namespace vector {
    void* string (const std::vector<std::string>& x);
    void* int64_t (const std::vector<std::int64_t>& x);
    void* double_t (const std::vector<double>& x);
    void* bool_t (const std::vector<bool>& x);
    void* Scalar (const std::vector<torch::Scalar>& x);
  }

  template <class... T>
  void* tuple (std::tuple<T...> x)
  {
    return make_ptr<std::vector<void*>>(to_vector(x));
  }

  namespace optional { 
    void* bool_t (const c10::optional<bool>& x);
    void* string (const c10::optional<std::string>& x);
    void* TensorList (const c10::List<c10::optional<torch::Tensor>>& x);
  }

}

#define LANTERN_FROM_RAW_DECL(name, type)                                                 \
  type& name (void* x);                 

namespace alias {
  using TensorDict = c10::Dict<std::string,torch::Tensor>;
}


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
  LANTERN_FROM_RAW_DECL(IntArrayRef, std::vector<std::int64_t>)
  LANTERN_FROM_RAW_DECL(TensorDict, alias::TensorDict)
  LANTERN_FROM_RAW_DECL(CompilationUnit, torch::CompilationUnit)
  LANTERN_FROM_RAW_DECL(QScheme, torch::QScheme)
  LANTERN_FROM_RAW_DECL(variable_list, torch::autograd::variable_list)
  LANTERN_FROM_RAW_DECL(Layout, torch::Layout)
  LANTERN_FROM_RAW_DECL(Storage, torch::Storage)
  LANTERN_FROM_RAW_DECL(string, std::string)
  LANTERN_FROM_RAW_DECL(int64_t, std::int64_t)
  LANTERN_FROM_RAW_DECL(bool_t, bool)
  LANTERN_FROM_RAW_DECL(double_t, double)

  namespace optional {
    c10::optional<torch::DimnameList> DimnameList (void* x);
    c10::optional<torch::Generator> Generator (void* x);
    c10::optional<torch::Tensor> Tensor (void* x);
    c10::optional<double> double_t (void* x);
    c10::optional<std::int64_t> int64_t (void* x);
    c10::optional<bool> bool_t (void* x);
    c10::optional<torch::ScalarType> ScalarType (void* x);
    c10::optional<std::string> string (void* x);
    c10::optional<torch::MemoryFormat> MemoryFormat (void* x);
    c10::optional<torch::Scalar> Scalar (void* x);
    c10::List<c10::optional<torch::Tensor>>& TensorList (void* x);
  }

  namespace vector {
    LANTERN_FROM_RAW_DECL(string, std::vector<std::string>)
    LANTERN_FROM_RAW_DECL(int64_t, std::vector<std::int64_t>)
    LANTERN_FROM_RAW_DECL(bool_t, Vector<bool>)
    LANTERN_FROM_RAW_DECL(double_t, std::vector<double>)
    LANTERN_FROM_RAW_DECL(Scalar, std::vector<torch::Scalar>)
  }

  LANTERN_FROM_RAW_DECL(tuple, std::vector<void*>)
}
