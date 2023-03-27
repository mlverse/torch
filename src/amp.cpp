#include <torch.h>


// [[Rcpp::export]]
bool cpp_amp_is_autocast_gpu_enabled () {
  return lantern_amp_is_autocast_gpu_enabled();
}

// [[Rcpp::export]]
bool cpp_amp_is_autocast_cpu_enabled () {
  return lantern_amp_is_autocast_cpu_enabled();
}

// [[Rcpp::export]]
void cpp_amp_autocast_set_gpu_enabled (bool enabled) {
  lantern_amp_autocast_set_gpu_enabled(enabled);
}

// [[Rcpp::export]]
void cpp_amp_autocast_set_cpu_enabled (bool enabled) {
  lantern_amp_autocast_set_cpu_enabled(enabled);
}

// [[Rcpp::export]]
void cpp_amp_autocast_set_gpu_dtype (torch::Dtype dtype) {
  lantern_amp_autocast_set_gpu_dtype(dtype.get());
}

// [[Rcpp::export]]
void cpp_amp_autocast_set_cpu_dtype (torch::Dtype dtype) {
  lantern_amp_autocast_set_cpu_dtype(dtype.get());
}

// [[Rcpp::export]]
void cpp_amp_autocast_set_cache_enabled (bool enabled) {
  lantern_amp_autocast_set_cache_enabled(enabled);
}

// [[Rcpp::export]]
bool cpp_amp_autocast_is_cache_enabled () {
  return lantern_amp_autocast_is_cache_enabled();
}

// [[Rcpp::export]]
torch::Dtype cpp_amp_autocast_get_gpu_dtype () {
  return torch::Dtype(lantern_amp_autocast_get_gpu_dtype());
}

// [[Rcpp::export]]
torch::Dtype cpp_amp_autocast_get_cpu_dtype () {
  return torch::Dtype(lantern_amp_autocast_get_cpu_dtype());
}

// [[Rcpp::export]]
void cpp_amp_autocast_increment_nesting () {
  lantern_amp_autocast_increment_nesting();
}

// [[Rcpp::export]]
int cpp_amp_autocast_decrease_nesting () {
  return lantern_amp_autocast_decrement_nesting();
}

// [[Rcpp::export]]
void cpp_amp_autocast_clear_cache () {
  return lantern_amp_autocast_clear_cache();
}

// [[Rcpp::export]]
int cpp_amp_foreach_non_finite_check_and_unscale (torch::TensorList params, torch::Tensor found_inf, torch::Tensor inv_scale) {
  return lantern_amp_foreach_non_finite_check_and_unscale(params.get(), found_inf.get(), inv_scale.get());
}

// [[Rcpp::export]]
void cpp_amp_update_scale_ (torch::Tensor scale, torch::Tensor growth_tracker, torch::Tensor found_inf, double scale_growth_factor, double scale_backoff_factor, torch::int64_t growth_interval) {
  lantern_amp_update_scale_(scale.get(), growth_tracker.get(), found_inf.get(), scale_growth_factor, scale_backoff_factor, growth_interval.get());
}