#pragma once

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
  
  operator T &() const
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

template<class T>
void* make_ptr (const T& x) {
  return (void*) std::make_unique<T>(x).release();
}

template<class T>
void* make_ptr () {
  return (void*) std::make_unique<T>().release();
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
    Vector() : x_() {}
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
    void push_back(T x) {
      x_.push_back(x);
    }
    size_t size() const { return x_.size(); }
    T at(size_t i) const { return x_.at(i);}
};

// a wrapper class for optional<torch::ArrayRef<T>> that owns all of it's memory and
// can easily be cast to the array ref type.
template <typename T>
class OptionalArrayRef {
  public:
    std::shared_ptr<std::vector<T>> x_;
    std::shared_ptr<c10::optional<torch::ArrayRef<T>>> x_ref_;
    OptionalArrayRef(const c10::optional<torch::ArrayRef<T>>& x) {
      if (x.has_value()) {
        x_ = std::make_shared<std::vector<T>>(x.value().vec());
        x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(*x_);
      } else {
        x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(c10::nullopt);
      }
    }
    OptionalArrayRef(const std::vector<T>& x){
      if (x.size() == 0) {
        x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(c10::nullopt);
      } else {
        x_ = std::make_shared<std::vector<T>>(x);
        x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(*x_);
      }
    }
    operator c10::optional<torch::ArrayRef<T>> &() {
      return *x_ref_;
    }
};


template <typename T>
class Box {
  public:
    std::shared_ptr<T> x_;
    Box(const T& x) {
      x_ = std::make_shared<T>(x);
    }
    operator T&() {
      return *x_;
    }
};


// Objects return from lantern must own all memory necessary to re-use them.
// This is kind of easy for tensors as heap allocated tensors own all their memory
// memory. However this is not true for `torch::TensorList` which is just a a reference
// to a stack allocated vector of tensors. Thus if we simply return `torch::TensorList`
// we won't be able to reuse it when the stack allocated memory is no longer available.
// 
// Types defined in this namespace are wrappers around torch types that don't own their memory
// thus need a wrapper that can hold the necessary information.
// Another important thing is that types here must be able to return their objects as references
// so they can be used in any possible lantern usage - that might include in-place modification
// of that object.
namespace self_contained {
  namespace optional {
    
    class DimnameList {
      public:
        std::shared_ptr<c10::optional<torch::DimnameList>> x_;
        std::shared_ptr<std::vector<torch::Dimname>> vec_;
        DimnameList (const c10::optional<torch::DimnameList>& x);
        operator c10::optional<torch::DimnameList>&();
    };

    using Generator = Box<c10::optional<torch::Generator>>;
    using Tensor = Box<c10::optional<torch::Tensor>>;
    using double_t = Box<c10::optional<double>>;
    using int64_t = Box<c10::optional<std::int64_t>>;
    using bool_t = Box<c10::optional<bool>>;
    using ScalarType = Box<c10::optional<torch::ScalarType>>;
    using string = Box<c10::optional<std::string>>;
    using MemoryFormat = Box<c10::optional<torch::MemoryFormat>>;
    using Scalar = Box<c10::optional<torch::Scalar>>;
    using IntArrayRef = OptionalArrayRef<std::int64_t>;
    using DoubleArrayRef = OptionalArrayRef<double>;
    
  }
}


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
  void* Stream (const at::Stream& x);
  void* IValue (const torch::IValue& x);

  namespace vector {
    void* string (const std::vector<std::string>& x);
    void* string ();
    void* int64_t (const std::vector<std::int64_t>& x);
    void* int64_t ();
    void* double_t (const std::vector<double>& x);
    void* double_t ();
    void* bool_t (const std::vector<bool>& x);
    void* bool_t ();
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
    void* IntArrayRef (const c10::optional<torch::IntArrayRef>& x);
    void* DoubleArrayRef (const c10::optional<torch::ArrayRef<double>>& x);
    void* Scalar (const c10::optional<torch::Scalar>& x);
    void* double_t (const c10::optional<double>& x);
    void* int64_t (const c10::optional<std::int64_t>& x);
    void* DimnameList (const c10::optional<torch::DimnameList>& x);
    void* Generator (const c10::optional<torch::Generator>& x);
    void* Tensor (const c10::optional<torch::Tensor>& x);
    void* ScalarType (const c10::optional<torch::ScalarType>& x);
    void* MemoryFormat (const c10::optional<torch::MemoryFormat>& x);
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
  LANTERN_FROM_RAW_DECL(Stream, at::Stream)
  LANTERN_FROM_RAW_DECL(IValue, torch::IValue)

  namespace optional {
    LANTERN_FROM_RAW_DECL(DimnameList, c10::optional<torch::DimnameList>)
    LANTERN_FROM_RAW_DECL(Generator, c10::optional<torch::Generator>)
    LANTERN_FROM_RAW_DECL(Tensor, c10::optional<torch::Tensor>)
    LANTERN_FROM_RAW_DECL(double_t, c10::optional<double>)
    LANTERN_FROM_RAW_DECL(int64_t, c10::optional<std::int64_t>)
    LANTERN_FROM_RAW_DECL(bool_t, c10::optional<bool>)
    LANTERN_FROM_RAW_DECL(ScalarType, c10::optional<torch::ScalarType>)
    LANTERN_FROM_RAW_DECL(string, c10::optional<std::string>)
    LANTERN_FROM_RAW_DECL(MemoryFormat, c10::optional<torch::MemoryFormat>)
    LANTERN_FROM_RAW_DECL(Scalar, c10::optional<torch::Scalar>)
    LANTERN_FROM_RAW_DECL(TensorList, c10::List<c10::optional<torch::Tensor>>)
    LANTERN_FROM_RAW_DECL(IntArrayRef, c10::optional<torch::IntArrayRef>)
    LANTERN_FROM_RAW_DECL(DoubleArrayRef, c10::optional<torch::ArrayRef<double>>)
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
