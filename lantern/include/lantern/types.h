#include <torch/torch.h>

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

template<class T>
void* make_ptr (const T& x) {
  return (void*) std::make_unique<T>(x).release();
}

template<class T>
void* make_ptr () {
  return (void*) std::make_unique<T>().release();
}

namespace make_raw {
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
    void* Device (const c10::optional<torch::Device>& x);
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
  LANTERN_FROM_RAW_DECL(TensorList, torch::TensorList)
  LANTERN_FROM_RAW_DECL(ScalarType, torch::ScalarType)
  LANTERN_FROM_RAW_DECL(Scalar, torch::Scalar)
  LANTERN_FROM_RAW_DECL(TensorOptions, torch::TensorOptions)
  LANTERN_FROM_RAW_DECL(Device, torch::Device)
  LANTERN_FROM_RAW_DECL(Dtype, torch::Dtype)
  LANTERN_FROM_RAW_DECL(Dimname, torch::Dimname)
  LANTERN_FROM_RAW_DECL(DimnameList, torch::DimnameList)
  LANTERN_FROM_RAW_DECL(Generator, torch::Generator)
  LANTERN_FROM_RAW_DECL(MemoryFormat, torch::MemoryFormat)
  LANTERN_FROM_RAW_DECL(IntArrayRef, torch::IntArrayRef)
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
    LANTERN_FROM_RAW_DECL(Device, c10::optional<torch::Device>)
  }

  namespace vector {
    LANTERN_FROM_RAW_DECL(string, std::vector<std::string>)
    LANTERN_FROM_RAW_DECL(int64_t, std::vector<std::int64_t>)
    LANTERN_FROM_RAW_DECL(double_t, std::vector<double>)
    LANTERN_FROM_RAW_DECL(Scalar, std::vector<torch::Scalar>)
    // This special type is used to allow converting to std::array<>'s 
    // of multiple different sizes.
    LANTERN_FROM_RAW_DECL(bool_t, Vector<bool>)
  }

  LANTERN_FROM_RAW_DECL(tuple, std::vector<void*>)
}