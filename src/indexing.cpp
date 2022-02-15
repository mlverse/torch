#include <torch.h>

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
std::vector<Rcpp::RObject> enquos0(Rcpp::Environment env) {
  SEXP dots = Rf_findVarInFrame3(env, R_DotsSymbol, TRUE);
  std::vector<Rcpp::RObject> out;
  R_len_t size = dots_size(dots);

  SEXP el;
  SEXP c;
  SEXP e;
  SEXP node = dots;

  for (R_len_t i = 0; i < size; ++i, node = CDR(node)) {
    el = CAR(node);

    while (true) {
      SEXP code = PRCODE(el);
      if (TYPEOF(code) != PROMSXP) break;
      el = code;
    }

    c = PRCODE(el);
    e = PRENV(el);

    out.push_back(Rcpp::List::create(c, e));
  }

  return out;
}

void list2env(Rcpp::Environment e, Rcpp::List mask) {
  std::vector<std::string> nms = mask.names();
  for (int i = 0; i < nms.size(); i++) {
    e.assign(nms[i], mask[nms[i]]);
  }
}

// [[Rcpp::export]]
std::vector<Rcpp::RObject> evaluate_slices(std::vector<Rcpp::RObject> quosures,
                                           Rcpp::List mask) {
  std::vector<Rcpp::RObject> out;

  Rcpp::Environment mask_e;
  Rcpp::List quosure;
  SEXP quosure_e;
  SEXP quosure_c;
  SEXP na = Rcpp::LogicalVector::create(NA_LOGICAL);
  for (int i = 0; i < quosures.size(); i++) {
    quosure = quosures[i];
    quosure_c = quosure[0];
    quosure_e = quosure[1];

    if (quosure_e == R_NilValue) {
      out.push_back(na);
      continue;
    }

    mask_e = Rcpp::new_env(quosure_e);
    list2env(mask_e, mask);
    out.push_back(Rf_eval(quosure_c, mask_e));
  }

  return out;
}

void index_append_empty_slice(XPtrTorchTensorIndex& index) {
  auto s = XPtrTorchSlice(lantern_Slice(
      XPtrTorchoptional_int64_t(lantern_optional_int64_t(nullptr)).get(),
      XPtrTorchoptional_int64_t(lantern_optional_int64_t(nullptr)).get(),
      XPtrTorchoptional_int64_t(
          lantern_optional_int64_t(XPtrTorchint64_t(lantern_int64_t(1)).get()))
          .get()));
  lantern_TensorIndex_append_slice(index.get(), s.get());
}

void index_append_scalar_integer(XPtrTorchTensorIndex& index, SEXP slice) {
  int s = Rf_asInteger(slice);

  if (s > 0) {
    s = s - 1;
  } else if (s == 0) {
    Rcpp::stop("Indexing in R is 1-based and found a 0.");
  }

  lantern_TensorIndex_append_int64(index.get(), s);
}

void index_append_none(XPtrTorchTensorIndex& index) {
  lantern_TensorIndex_append_none(index.get());
}

void index_append_scalar_bool(XPtrTorchTensorIndex& index, SEXP slice) {
  bool s = Rf_asLogical(slice);
  lantern_TensorIndex_append_bool(index.get(), s);
}

void index_append_ellipsis(XPtrTorchTensorIndex& index) {
  lantern_TensorIndex_append_ellipsis(index.get());
}

void index_append_slice(XPtrTorchTensorIndex& index, SEXP slice) {
  Rcpp::List s = slice;
  auto start = Rcpp::as<XPtrTorchoptional_int64_t>(s["start"]);
  auto end = Rcpp::as<XPtrTorchoptional_int64_t>(s["end"]);
  auto step = Rcpp::as<XPtrTorchoptional_int64_t>(s["step"]);
  XPtrTorchSlice l = lantern_Slice(start.get(), end.get(), step.get());

  lantern_TensorIndex_append_slice(index.get(), l.get());
}

void index_append_integer_vector(XPtrTorchTensorIndex& index, SEXP slice) {
  Rcpp::NumericVector v(
      LENGTH(slice));  // tmp variable v to avoid changing slice object
  Rcpp::NumericVector u = slice;  // cast slice to numeric
  for (int j = 0; j < u.size(); j++) {
    if (u[j] > 0) {
      v[j] = u[j] - 1;  // make it 0-based.
    } else if (u[j] == 0) {
      Rcpp::stop("Indexing in R is 1-based and found a 0.");
    } else {
      v[j] = u[j];
    }
  }

  // Create the integer Tensor
  auto tensor = torch_tensor_cpp(v, torch::Dtype(lantern_Dtype_int64()));

  lantern_TensorIndex_append_tensor(index.get(), tensor.get());
}

void index_append_bool_vector(XPtrTorchTensorIndex& index, SEXP slice) {
  Rcpp::LogicalVector v = slice;

  // Create the integer Tensor
  auto tensor = torch_tensor_cpp(v, torch::Dtype(lantern_Dtype_bool()));

  lantern_TensorIndex_append_tensor(index.get(), tensor.get());
}

bool index_append_tensor(XPtrTorchTensorIndex& index, SEXP slice) {
  Rcpp::XPtr<XPtrTorchTensor> t = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(slice);

  auto s =
      lantern_Dtype_type(XPtrTorchDtype(lantern_Tensor_dtype(t->get())).get());
  auto type = std::string(s);
  lantern_const_char_delete(s);

  // is boolean tensor
  if (type == "Bool") {
    lantern_TensorIndex_append_tensor(index.get(), t->get());
  }
  // integer tensor: we need to make it zero based
  else if (type == "Long") {
    bool current_autograd_mode = lantern_autograd_is_enabled();
    lantern_autograd_set_grad_mode(false);
    // check that there's no zeros
    bool zeros = lantern_Tensor_has_any_zeros(t->get());
    if (zeros) {
      lantern_autograd_set_grad_mode(current_autograd_mode);
      Rcpp::stop("Indexing starts at 1 but found a 0.");
    }

    XPtrTorchTensor sign = lantern_Tensor_signbit_tensor(t->get());
    sign = lantern_logical_not_tensor(sign.get());

    // cast from bool to int
    XPtrTorchTensorOptions options = lantern_TensorOptions();
    options = lantern_TensorOptions_dtype(
        options.get(), XPtrTorchDtype(lantern_Dtype_int64()).get());
    sign = lantern_Tensor_to(sign.get(), options.get());

    // create a 1 scalar
    int al = 1;
    XPtrTorchScalar alpha =
        lantern_Scalar((void*)&al, std::string("int").c_str());

    XPtrTorchTensor zero_index = lantern_Tensor_sub_tensor_tensor_scalar(
        t->get(), sign.get(), alpha.get());

    lantern_TensorIndex_append_tensor(index.get(), zero_index.get());
    lantern_autograd_set_grad_mode(current_autograd_mode);
  } else {
    Rcpp::stop("Only long and boolean tensors are supported.");
  }

  return lantern_Tensor_ndimension(t->get()) == 0;
}

struct index_info {
  int dim;      // number of dimensions that are kept in the output
  bool vector;  // vector like?
  bool ellipsis;
};

// returns true if appended a vector like object. We use the boolean vector
// to decide if we should start a new index object.
index_info index_append_sexp(XPtrTorchTensorIndex& index, SEXP slice,
                             bool drop) {
  // a single NA means empty argument which and in turn we must select
  // all elements in that dimension.
  if (TYPEOF(slice) == LGLSXP && LENGTH(slice) == 1 &&
      LOGICAL(slice)[0] == NA_LOGICAL) {
    index_append_empty_slice(index);
    return {1, false, false};
  }

  // a single numeric scalar to take a single element of a dimension,
  // this slice will drop the dimension so we optionaly add a `none`
  // to add it again.
  if ((TYPEOF(slice) == REALSXP || TYPEOF(slice) == INTSXP) &&
      LENGTH(slice) == 1) {
    index_append_scalar_integer(index, slice);
    if (!drop) {
      index_append_none(index);
      return {1, false, false};
    } else {
      return {0, false, false};
    }
  }

  // scalar boolean
  if (TYPEOF(slice) == LGLSXP && LENGTH(slice) == 1) {
    index_append_scalar_bool(index, slice);
    return {1, false, false};
  }

  // the fill sybol was passed. in this case we add the ellipsis ...
  if (Rf_inherits(slice, "fill")) {
    index_append_ellipsis(index);
    return {1, false, true};
  }

  // NULL means add an axis.
  if (TYPEOF(slice) == NILSXP) {
    index_append_none(index);
    return {1, false, false};
  }

  // if it's a slice with start and end values
  if (Rf_inherits(slice, "slice")) {
    index_append_slice(index, slice);
    return {1, false, false};
  }

  // if it's a numeric vector
  if ((TYPEOF(slice) == REALSXP || TYPEOF(slice) == INTSXP) &&
      LENGTH(slice) > 1) {
    index_append_integer_vector(index, slice);
    return {1, true, false};
  }

  if (TYPEOF(slice) == LGLSXP && LENGTH(slice) > 1) {
    index_append_bool_vector(index, slice);
    return {1, true, false};
  }

  if (Rf_inherits(slice, "torch_tensor")) {
    bool is_scalar = index_append_tensor(index, slice);
    if (is_scalar) {
      return {0, false, false};
    } else {
      return {1, true, false};
    }
  }

  Rcpp::stop("Unsupported index.");
}

std::vector<XPtrTorchTensorIndex> slices_to_index(
    std::vector<Rcpp::RObject> slices, bool drop) {
  std::vector<XPtrTorchTensorIndex> output;
  XPtrTorchTensorIndex index = lantern_TensorIndex_new();
  SEXP slice;
  int num_dim = 0;
  bool has_ellipsis = false;
  for (int i = 0; i < slices.size(); i++) {
    slice = slices[i];
    auto info = index_append_sexp(index, slice, drop);

    if (!has_ellipsis && info.ellipsis) {
      has_ellipsis = true;
    }

    num_dim += info.dim;
    if (info.vector) {
      bool last_dim = i >= (slices.size() - 1);
      // we add an ellipsis to get all the other dimensions and append it to the
      // output vector. we only append if it's not the last dimension too.
      if (!last_dim && !has_ellipsis) {
        index_append_ellipsis(index);
      } else if (!last_dim && has_ellipsis) {
        int missing_slices = slices.size() - i - 1;
        for (int j = 0; j < missing_slices; j++) {
          index_append_empty_slice(index);
        }
      }

      output.push_back(index);
      index = lantern_TensorIndex_new();
      // if there's still more slices to go trough, we need to add empty slices
      // in the new index object. the number of empty slices that we need to add
      // is related to the number of dimensions of the resulting intermidiary
      // tensor that is tracked by num_dim.
      if (!last_dim) {
        if (!has_ellipsis) {
          for (int j = 0; j < num_dim; j++) {
            index_append_empty_slice(index);
          }
        } else {
          index_append_ellipsis(index);
        }
      }
    }
  }

  if (!lantern_TensorIndex_is_empty(index.get())) {
    output.push_back(index);
  }

  return output;
}

// [[Rcpp::export]]
XPtrTorchTensor Tensor_slice(XPtrTorchTensor self, Rcpp::Environment e,
                             bool drop, Rcpp::List mask) {
  auto dots = evaluate_slices(enquos0(e), mask);
  auto index = slices_to_index(dots, drop);
  XPtrTorchTensor out = self;
  for (auto& ind : index) {
    out = lantern_Tensor_index(out.get(), ind.get());
  }
  return out;
}

XPtrTorchScalar cpp_torch_scalar(SEXP x);

// [[Rcpp::export]]
void Tensor_slice_put(Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::Environment e,
                      SEXP rhs, Rcpp::List mask) {
  auto dots = evaluate_slices(enquos0(e), mask);
  auto indexes = slices_to_index(dots, true);

  if (indexes.size() > 1) {
    Rcpp::stop(
        "Subset assignment indexing doesn't work with vector like indexing. "
        "Use slices or scalar indexing.");
  }

  auto index = indexes.at(0);

  if ((TYPEOF(rhs) == REALSXP || TYPEOF(rhs) == INTSXP ||
       TYPEOF(rhs) == LGLSXP || TYPEOF(rhs) == STRSXP) &&
      LENGTH(rhs) == 1) {
    auto s = cpp_torch_scalar(rhs);
    lantern_Tensor_index_put_scalar_(self->get(), index.get(), s.get());
    return;
  }

  if (Rf_inherits(rhs, "torch_tensor")) {
    Rcpp::XPtr<XPtrTorchTensor> t = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(rhs);
    lantern_Tensor_index_put_tensor_(self->get(), index.get(), t->get());
    return;
  }

  Rcpp::stop("rhs must be a torch_tensor or scalar value.");
}
