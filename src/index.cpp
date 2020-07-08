#include "torch_types.h"
#include "utils.hpp"

static R_len_t dots_size(SEXP dots) {
  if (dots == R_UnboundValue) {
    // No dots at all in the environment
    return 0;
  } else if (dots == R_MissingArg) {
    // Dots are present, but none were supplied
    return 0;
  } else {
    return Rf_length(dots);
  }
}

// [[Rcpp::export]]
std::vector<Rcpp::RObject> enquos0 (Rcpp::Environment env)
{
  
  SEXP dots = Rf_findVarInFrame3(env, R_DotsSymbol, TRUE);
  std::vector<Rcpp::RObject> out;
  R_len_t size = dots_size(dots);
  
  SEXP el;
  SEXP c;
  SEXP e;
  SEXP node = dots;
  
  for(R_len_t i = 0; i < size; ++i, node = CDR(node)) {
    
    el = CAR(node);
    
    while(true) {
      SEXP code = PRCODE(el);
      if(TYPEOF(code) != PROMSXP) break;
      el = code;
    }
    
    c = PRCODE(el);
    e = PRENV(el);
    
    out.push_back(Rcpp::List::create(c, e));
  }
  
  return out;
}

void list2env (Rcpp::Environment e, Rcpp::List mask)
{
  std::vector<std::string> nms = mask.names();
  for (int i = 0; i < nms.size(); i++)
  {
    e.assign(nms[i], mask[nms[i]]);
  }
}

// [[Rcpp::export]]
std::vector<Rcpp::RObject> evaluate_slices (std::vector<Rcpp::RObject> quosures, Rcpp::List mask)
{
  std::vector<Rcpp::RObject> out;
  
  Rcpp::Environment mask_e;
  Rcpp::List quosure;
  SEXP quosure_e;
  SEXP quosure_c;
  SEXP na = Rcpp::LogicalVector::create(NA_LOGICAL);
  for (int i = 0; i < quosures.size(); i++)
  {
    quosure = quosures[i];
    quosure_c = quosure[0];
    quosure_e = quosure[1];
    
    if (quosure_e == R_NilValue)
    {
      out.push_back(na);
      continue;
    }
    
    mask_e = Rcpp::new_env(quosure_e);
    list2env(mask_e, mask);
    out.push_back(Rf_eval(quosure_c, mask_e));
  }
  
  return out;
}

Rcpp::XPtr<XPtrTorchTensor> cpp_torch_tensor (SEXP x, std::vector<std::int64_t> dim,
                                              Rcpp::XPtr<XPtrTorchTensorOptions> options,
                                              bool requires_grad);

XPtrTorchTensorIndex slices_to_index (std::vector<Rcpp::RObject> slices, bool drop)
{
  XPtrTorchTensorIndex index = lantern_TensorIndex_new();
  
  SEXP slice;
  for (int i = 0; i < slices.size(); i++)
  {
    slice = slices[i];
    
    // a single NA means empty argument which and in turn we must select
    // all elements in that dimension.
    if (TYPEOF(slice) == LGLSXP && LENGTH(slice) == 1 && LOGICAL(slice)[0] == NA_LOGICAL)
    {
      XPtrTorchSlice s = lantern_Slice(lantern_optional_int64_t(0, true), 
                                       lantern_optional_int64_t(0, true),
                                       lantern_optional_int64_t(1, false));
      lantern_TensorIndex_append_slice(index.get(), s.get());
      continue;
    }
    
    // a single numeric scalar to take a single element of a dimension,
    // this slice will drop the dimension so we optionaly add a `none`
    // to add it again.
    if ((TYPEOF(slice) == REALSXP || TYPEOF(slice) == INTSXP) && LENGTH(slice) == 1)
    {
      int s = Rf_asInteger(slice);
      
      if (s > 0)
      {
        s = s-1;
      }
      else if (s == 0)
      {
        Rcpp::stop("Indexing in R is 1-based and found a 0.");
      }
    
      lantern_TensorIndex_append_int64(index.get(), s);
      
      if (!drop)
      {
        lantern_TensorIndex_append_none(index.get());
      }
      
      continue;
    }
    
    // the fill sybol was passed. in this case we add the ellipsis ...
    if (Rf_inherits(slice, "fill"))
    {
      lantern_TensorIndex_append_ellipsis(index.get());
      continue;
    }
    
    // NULL means add an axis.
    if (TYPEOF(slice) == NILSXP)
    {
      lantern_TensorIndex_append_none(index.get());
      continue;
    }
    
    // if it's a slice with start and end values
    if (Rf_inherits(slice, "slice"))
    {
      Rcpp::List s = slice;
      XPtrTorchoptional_int64_t start = lantern_optional_int64_t(s["start"], false);
      XPtrTorchoptional_int64_t end = lantern_optional_int64_t(s["end"], false);
      XPtrTorchoptional_int64_t step = lantern_optional_int64_t(s["step"], false);
      XPtrTorchSlice l = lantern_Slice(start.get(), end.get(), step.get());
      
      lantern_TensorIndex_append_slice(index.get(), l.get());
      continue;
    }
    
    // if it's a numeric vector
    if ((TYPEOF(slice) == REALSXP || TYPEOF(slice) == INTSXP) && LENGTH(slice) > 1)
    {
      
      Rcpp::NumericVector v = slice;
      for (int j = 0; j < v.size(); j++)
      {
        if (v[j] > 0)
        {
          v[j] = v[j] - 1; // make it 0-based.
        }
        else if (v[j] == 0)
        {
          Rcpp::stop("Indexing in R is 1-based and found a 0.");
        }
      }
    
      // Create the integer Tensor
      XPtrTorchTensorOptions options = lantern_TensorOptions();
      options = lantern_TensorOptions_dtype(options.get(), XPtrTorchDtype(lantern_Dtype_int64()).get());
      std::vector<int64_t> dim = {LENGTH(slice)};
      
      Rcpp::XPtr<XPtrTorchTensor> tensor = cpp_torch_tensor(v, dim, make_xptr<XPtrTorchTensorOptions>(options), false);
      
      lantern_TensorIndex_append_tensor(index.get(), tensor->get());
      continue;
    }
    
    if (Rf_inherits(slice, "torch_tensor"))
    {
      Rcpp::Environment e = slice;
      Rcpp::XPtr<XPtrTorchTensor> t = e["ptr"];
      lantern_TensorIndex_append_tensor(index.get(), t->get());
    }
    
  }
  
  return index;
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> Tensor_slice(Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::Environment e,
                                         bool drop, Rcpp::List mask)
{
  auto dots = evaluate_slices(enquos0(e), mask);
  auto index = slices_to_index(dots, drop);
  XPtrTorchTensor out = lantern_Tensor_index(self->get(), index.get());
  return make_xptr<XPtrTorchTensor>(out);
}

Rcpp::XPtr<XPtrTorchScalar> cpp_torch_scalar (SEXP x);

// [[Rcpp::export]]
void Tensor_slice_put(Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::Environment e,
                      SEXP rhs, Rcpp::List mask)
{
  auto dots = evaluate_slices(enquos0(e), mask);
  auto index = slices_to_index(dots, true);
  
  
  if ((TYPEOF(rhs) == REALSXP || TYPEOF(rhs) == INTSXP || TYPEOF(rhs) == LGLSXP ||
      TYPEOF(rhs) == STRSXP) && LENGTH(rhs) == 1)
  {
    auto s = cpp_torch_scalar(rhs);
    lantern_Tensor_index_put_scalar_(self->get(), index.get(), s->get());  
    return;
  }
  
  if (Rf_inherits(rhs, "torch_tensor"))
  {
    Rcpp::Environment e = rhs;
    Rcpp::XPtr<XPtrTorchTensor> t = e["ptr"];
    lantern_Tensor_index_put_tensor_(self->get(), index.get(), t->get());  
    return;
  }
  
  Rcpp::stop("rhs must be a tensor or scalar");
}

