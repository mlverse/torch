#include <torch.h>

// [[Rcpp::export]]
bool cpp_cuda_is_available() { return lantern_cuda_is_available(); }

// [[Rcpp::export]]
int cpp_cuda_device_count() { return lantern_cuda_device_count(); }

// [[Rcpp::export]]
void cpp_cuda_synchronize(int device) {lantern_cuda_synchronize(device); }

// [[Rcpp::export]]
int64_t cpp_cuda_current_device() { return lantern_cuda_current_device(); }

// [[Rcpp::export]]
XPtrTorchvector_int64_t cpp_cuda_get_device_capability(int64_t device) {
  return lantern_cuda_get_device_capability(device);
}

// [[Rcpp::export]]
int64_t cpp_cudnn_runtime_version() { return lantern_cudnn_runtime_version(); }

// [[Rcpp::export]]
bool cpp_cudnn_is_available() { return lantern_cudnn_is_available(); }

// [[Rcpp::export]]
torch::vector::int64_t cpp_cuda_memory_stats(int64_t device) {
  return torch::vector::int64_t(lantern_cuda_device_stats(device));
}

// [[Rcpp::export]]
int cpp_cuda_get_runtime_version() {
  return lantern_cuda_get_runtime_version();
}

// [[Rcpp::export]]
void cpp_cuda_empty_cache () {
  lantern_cuda_empty_cache();
}

// [[Rcpp::export]]
void cpp_cuda_record_memory_history(Rcpp::Nullable<std::string> enabled, Rcpp::Nullable<std::string> context, std::string stacks, size_t max_entries) {
  void* en_ptr = nullptr;
  void* ctx_ptr = nullptr;
  std::string en_str, ctx_str;
  if (enabled.isNotNull()) {
    en_str = Rcpp::as<std::string>(enabled);
    en_ptr = (void*)&en_str;
  }
  if (context.isNotNull()) {
    ctx_str = Rcpp::as<std::string>(context);
    ctx_ptr = (void*)&ctx_str;
  }
  lantern_cuda_record_memory_history(en_ptr, ctx_ptr, (void*)&stacks, max_entries);
}

// [[Rcpp::export]]
Rcpp::RawVector cpp_cuda_memory_snapshot() {
  std::string snapshot = torch::string(lantern_cuda_memory_snapshot());
  Rcpp::RawVector out(snapshot.size());
  std::copy(snapshot.begin(), snapshot.end(), out.begin());
  return out;
}