#ifndef TORCH_DELETERS
#define TORCH_DELETERS

void delete_tensor(void* x);
void delete_script_module(void* x);
void delete_script_method(void* x);
void delete_optional_tensor(void* x);
void delete_tensor_list(void* x);
void delete_optional_tensor_list(void* x);
void delete_scalar_type(void* x);
void delete_scalar(void* x);
void delete_tensor_options(void* x);
void delete_device(void* x);
void delete_optional_device(void* x);
void delete_dtype(void* x);
void delete_dimname(void* x);
void delete_dimname_list(void* x);
void delete_jit_named_parameter_list(void* x);
void delete_jit_named_buffer_list(void* x);
void delete_jit_named_module_list(void* x);
void delete_generator(void* x);
void delete_memory_format(void* x);
void delete_function_ptr(void* x);
void delete_optional_int_array_ref(void* x);
void delete_vector_string(void* x);
void delete_string(void* x);
void delete_stack(void* x);
void delete_ivalue(void* x);
void delete_tuple(void* x);
void delete_vector_bool(void* x);
void delete_vector_scalar(void* x);
void delete_vector_int64_t(void* x);
void delete_vector_double(void* x);
void delete_tensor_dict(void* x);
void delete_generic_dict(void* x);
void delete_generic_list(void* x);
void delete_vector_ivalue(void* x);
void delete_named_tuple_helper(void* x);
void delete_compilation_unit(void* x);
void delete_qscheme(void* x);
void delete_double(void* x);
void delete_variable_list(void* x);
void delete_int64_t(void* x);
void delete_bool(void* x);
void delete_layout(void* x);
void delete_tensor_index(void* x);
void delete_optional_int64_t(void* x);
void delete_slice(void* x);
void delete_packed_sequence(void* x);
void delete_storage(void* x);
void delete_jit_module(void* x);
void delete_traceable_function(void* x);
void delete_vector_void(void* x);
void delete_optional_bool(void* x);
void delete_optional_double_array_ref(void* x);
void delete_optional_dimname_list(void* x);
void delete_optional_generator(void* x);
void delete_optional_double(void* x);
void delete_optional_scalar_type(void* x);
void delete_optional_string(void* x);
void delete_optional_scalar(void* x);
void delete_optional_memory_format(void* x);
void delete_string_view(void* x);
void delete_optional_string_view(void* x);

void* fixme_new_string(const char* x);
void* fixme_new_dimname(const char* x);

#endif  // TORCH_DELETERS