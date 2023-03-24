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
