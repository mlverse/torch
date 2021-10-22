#include <torch.h>

// This file defines torch functions that are exported in order to create torch's
// C API. 
// Most of functions here should not be directly used, but they are called by
// torch type wrappers.

static inline void tensor_finalizer (SEXP ptr)
{
  auto xptr = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(ptr);
  lantern_tensor_set_pyobj(xptr->get(), nullptr);
}


// torch_tensor 

SEXP operator_sexp_tensor (const XPtrTorchTensor* self)
{
  // If there's an R object stored in the Tensor Implementation
  // we want to return it directly so we have a unique R object
  // that points to each tensor.
  if (lantern_tensor_get_pyobj(self->get()))
  {
    // It could be that the R objet is still stored in the TensorImpl but
    // it has already been scheduled for finalization by the GC.
    // Thus we need to run the pending finalizers and retry.
    R_RunPendingFinalizers();  
    void* ptr = lantern_tensor_get_pyobj(self->get());
    if (ptr)
    {
      SEXP out = PROTECT(Rf_duplicate((SEXP) ptr));
      UNPROTECT(1);
      return out;
    }
  }
  
  // If there's no R object stored in the Tensor, we will create a new one 
  // and store the weak reference.
  // Since this will be the only R object that points to that tensor, we also
  // register a finalizer that will erase the reference to the R object in the 
  // C++ object whenever this object gets out of scope.
  auto xptr = make_xptr<XPtrTorchTensor>(*self);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
  SEXP xptr_ = PROTECT(Rcpp::wrap(xptr));
  R_RegisterCFinalizer(xptr_, tensor_finalizer);
  lantern_tensor_set_pyobj(self->get(), (void*) xptr_);
  UNPROTECT(1);
  return xptr_; 
}

XPtrTorchTensor from_sexp_tensor (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(x);
    return XPtrTorchTensor( out->get_shared());
  }
  
  if (TYPEOF(x) == NILSXP || (TYPEOF(x) == VECSXP && LENGTH(x) == 0)) {
    return cpp_tensor_undefined();
  }
  
  // TODO: it would be nice to make it all C++ 
  if (Rf_isVectorAtomic(x)) {
    Rcpp::Environment torch_pkg = Rcpp::Environment("package:torch");
    Rcpp::Function f = torch_pkg["torch_tensor"];
    return XPtrTorchTensor(Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(f(x))->get_shared());
  }
  
  Rcpp::stop("Expected a torch_tensor.");
}

// optional_torch_tensor

SEXP operator_sexp_optional_tensor (const XPtrTorchOptionalTensor* self) 
{
  bool has_value = lantern_optional_tensor_has_value(self->get());
  if (!has_value)
  {
    return R_NilValue;
  }
  else 
  {
    auto ten = PROTECT(XPtrTorchTensor(lantern_optional_tensor_value(self->get())));
    auto sxp = Rcpp::wrap(ten);
    UNPROTECT(1);
    return sxp;
  }
}

XPtrTorchOptionalTensor from_sexp_optional_tensor (SEXP x)
{
  const bool is_null = TYPEOF(x) == NILSXP || (TYPEOF(x) == VECSXP && LENGTH(x) == 0);
  if (is_null)
  {
    return XPtrTorchOptionalTensor(lantern_optional_tensor(nullptr, true));
  }
  else
  {
    return XPtrTorchOptionalTensor(lantern_optional_tensor(XPtrTorchTensor(x).get(), false));
  }
}

// index tensor

XPtrTorchIndexTensor from_sexp_index_tensor (SEXP x)
{
  XPtrTorchTensor t = from_sexp_tensor(x);
  XPtrTorchTensor zero_index = to_index_tensor(t);
  
  return XPtrTorchIndexTensor(zero_index.get_shared());
}

// tensor_list

SEXP operator_sexp_tensor_list (const XPtrTorchTensorList* self) {
  Rcpp::List out;
  int64_t sze = lantern_TensorList_size(self->get());
  
  for (int i = 0; i < sze; i++)
  {
    void * tmp = lantern_TensorList_at(self->get(), i);
    out.push_back(XPtrTorchTensor(tmp));
  }
  
  return out;
}


