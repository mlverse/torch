#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchStack> cpp_stack_new ()
{
  XPtrTorchStack out = lantern_Stack_new();
  return make_xptr<XPtrTorchStack>(out);
}

// [[Rcpp::export]]
void cpp_stack_push_back_Tensor (Rcpp::XPtr<XPtrTorchStack> self,
                                 Rcpp::XPtr<XPtrTorchTensor> x)
{
  lantern_Stack_push_back_Tensor(self->get(), x->get());  
}

// [[Rcpp::export]]
void cpp_stack_push_back_int64_t (Rcpp::XPtr<XPtrTorchStack> self,
                                  int64_t x)
{
  lantern_Stack_push_back_int64_t(self->get(), x);
}

Rcpp::XPtr<XPtrTorchTensor> cpp_stack_at_Tensor (Rcpp::XPtr<XPtrTorchStack> self,
                             int64_t i)
{
  XPtrTorchTensor out = lantern_Stack_at_Tensor(self->get(), i);
  return make_xptr<XPtrTorchTensor>(out);
}

int64_t cpp_stack_at_int64_t (Rcpp::XPtr<XPtrTorchStack> self,
                             int64_t i)
{
  int64_t out = lantern_Stack_at_int64_t(self->get(), i);
  return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_stack_to_r (Rcpp::XPtr<XPtrTorchStack> self) 
{

  int64_t size = lantern_Stack_size(self->get());
  Rcpp::List output;
  
  std::string type;
  
  for (int i=0; i < size; i++)
  {
    
    type = "Tensor";
    if (lantern_Stack_at_is(self->get(), i, type.c_str()))
    {
      output.push_back(Rcpp::List::create(cpp_stack_at_Tensor(self, i), type));
      continue;
    }
    
    type = "Int";
    if (lantern_Stack_at_is(self->get(), i, type.c_str()))
    {
      output.push_back(Rcpp::List::create(cpp_stack_at_int64_t(self, i), type));
      continue;
    }
    
    Rcpp::stop("Stack contains non supported types");
  }
  
  return output;
}