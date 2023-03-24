#define LANTERN_BUILD
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include "lantern/lantern.h"
#include "lantern/types.h"

bool _lantern_amp_is_autocast_gpu_enabled() {
  LANTERN_FUNCTION_START
  return at::autocast::is_enabled();
  LANTERN_FUNCTION_END
}

bool _lantern_amp_is_autocast_cpu_enabled() {
  LANTERN_FUNCTION_START
  return at::autocast::is_cpu_enabled();
  LANTERN_FUNCTION_END
}

void _lantern_amp_autocast_set_gpu_enabled(bool enabled) {
  LANTERN_FUNCTION_START
  at::autocast::set_enabled(enabled);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_amp_autocast_set_cpu_enabled(bool enabled) {
  LANTERN_FUNCTION_START
  at::autocast::set_cpu_enabled(enabled);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_amp_autocast_set_gpu_dtype (void* dtype) {
    LANTERN_FUNCTION_START
    auto scalar_type = from_raw::ScalarType(dtype);
    at::autocast::set_autocast_gpu_dtype(scalar_type);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_amp_autocast_set_cpu_dtype (void* dtype) {
    LANTERN_FUNCTION_START
    auto scalar_type = from_raw::ScalarType(dtype);
    at::autocast::set_autocast_cpu_dtype(scalar_type);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_amp_autocast_set_cache_enabled(bool enabled) {
  LANTERN_FUNCTION_START
  at::autocast::set_autocast_cache_enabled(enabled);
  LANTERN_FUNCTION_END_VOID
}

bool _lantern_amp_autocast_is_cache_enabled() {
  LANTERN_FUNCTION_START
  return at::autocast::is_autocast_cache_enabled();
  LANTERN_FUNCTION_END
}

void* _lantern_amp_autocast_get_gpu_dtype () {
    LANTERN_FUNCTION_START
    auto scalar_type = at::autocast::get_autocast_gpu_dtype();
    return make_raw::ScalarType(scalar_type);
    LANTERN_FUNCTION_END_VOID
}

void* _lantern_amp_autocast_get_cpu_dtype () {
    LANTERN_FUNCTION_START
    auto scalar_type = at::autocast::get_autocast_cpu_dtype();
    return make_raw::ScalarType(scalar_type);
    LANTERN_FUNCTION_END_VOID
}

void _lantern_amp_autocast_increment_nesting () {
    LANTERN_FUNCTION_START
    at::autocast::increment_nesting();
    LANTERN_FUNCTION_END_VOID
}

int _lantern_amp_autocast_decrement_nesting () {
    LANTERN_FUNCTION_START
    return at::autocast::decrement_nesting();
    LANTERN_FUNCTION_END
}

void _lantern_amp_autocast_clear_cache () {
    LANTERN_FUNCTION_START
    at::autocast::clear_cache();
    LANTERN_FUNCTION_END_VOID
}

int _lantern_amp_foreach_non_finite_check_and_unscale (void* params, void* found_inf, void* inv_scale) 
  {
    LANTERN_FUNCTION_START
    auto params_ = from_raw::TensorList(params);
    auto found_inf_ = from_raw::Tensor(found_inf);
    auto inv_scale_ = from_raw::Tensor(inv_scale);

    int found = 0;
    for (auto& param : params_) {
        auto grad = param.grad();
        auto found_inf__ = found_inf_.to(grad.device());
        auto inv_scale__ = inv_scale_.to(grad.device());
        at::_amp_foreach_non_finite_check_and_unscale_(grad, found_inf__, inv_scale__);
        found += found_inf__.sum().item().toInt();
    } 
    return found;
    LANTERN_FUNCTION_END_VOID
}

void _lantern_amp_update_scale_ (void* self, void* growth_tracker, void* found_inf, double scale_growth_factor, double scale_backoff_factor, void* growth_interval) {
  auto self_ = from_raw::Tensor(self);
  auto growth_tracker_ = from_raw::Tensor(growth_tracker);
  auto found_inf_ = from_raw::Tensor(found_inf);
  auto growth_interval_ = from_raw::int64_t(growth_interval);

  at::_amp_update_scale_(self_, growth_tracker_, found_inf_, scale_growth_factor, scale_backoff_factor, growth_interval_);
}
