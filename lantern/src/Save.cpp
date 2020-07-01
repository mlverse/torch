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

const char  * lantern_tensor_save (void* self)
{
    auto t = reinterpret_cast<LanternObject<torch::Tensor> *>(self)->get();

    std::ostringstream oss;
    torch::save(t, oss);

    auto str = std::string(macaron::Base64::Encode(oss.str()));
    
    char *cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    return cstr;
}

std::size_t lantern_tensor_serialized_size (const char * s)
{
    std::istringstream iss_size(std::string(s, 20));
    std::size_t size;
    iss_size >> size;
    return size;
}

void * lantern_tensor_load (const char * s) 
{
    std::string str;
    macaron::Base64::Decode(std::string(s), str);
    std::istringstream stream(str);

    torch::Tensor t;
    torch::load(t, stream);
    return (void*) new LanternObject<torch::Tensor>(t);
}

void* lantern_test_tensor ()
{
    return (void*) new LanternObject<torch::Tensor>(torch::ones({5, 5}));
}

void lantern_test_print (void * x)
{
    auto t = reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get();
    std::cout << t << std::endl;
}