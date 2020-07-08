#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"
#include <torch/torch.h>

#include "utils.hpp"
#include "base64.hpp"

std::string size_t_to_string( std::size_t i) {
    std::stringstream ss;
    ss << std::setw(20) << std::setfill('0') << i;
    return ss.str();
}

const char  * _lantern_tensor_save (void* self)
{
    LANTERN_FUNCTION_START
    auto t = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();

    std::ostringstream oss;
    torch::save(t, oss);

    auto str = std::string(macaron::Base64::Encode(oss.str()));
    
    char *cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    return cstr;
    LANTERN_FUNCTION_END
}

std::size_t _lantern_tensor_serialized_size (const char * s)
{
    LANTERN_FUNCTION_START
    std::istringstream iss_size(std::string(s, 20));
    std::size_t size;
    iss_size >> size;
    return size;
    LANTERN_FUNCTION_END_RET(0)
}

void * _lantern_tensor_load (const char * s) 
{
    LANTERN_FUNCTION_START
    std::string str;
    macaron::Base64::Decode(std::string(s), str);
    std::istringstream stream(str);

    torch::Tensor t;
    torch::load(t, stream);
    return (void*) new LanternObject<torch::Tensor>(t);
    LANTERN_FUNCTION_END
}

void* _lantern_test_tensor ()
{
    LANTERN_FUNCTION_START
    return (void*) new LanternObject<torch::Tensor>(torch::ones({5, 5}));
    LANTERN_FUNCTION_END
}

void _lantern_test_print (void * x)
{
    LANTERN_FUNCTION_START
    auto t = reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get();
    std::cout << t << std::endl;
    LANTERN_FUNCTION_END_VOID
}