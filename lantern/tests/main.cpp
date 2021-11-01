#include <iostream>
#include <fstream>
#include <vector>

#define LANTERN_HEADERS_ONLY
#define LANTERN_CHECK_LOADED
bool lantern_loaded = false;
#include "lantern/lantern.h"

#include "init.h"

void lantern_host_handler() {
    
}

int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        std::cout << "Usage: lanterntest <path-to-lib>";
        return 1;
    }

    myInit(argv[1]);

    lanternTest();

    void *device = lantern_Device("cpu", 0, false);
    std::cout << "Device: " << lantern_Device_type(device) << ":" << lantern_Device_index(device) << std::endl;

    void *generator = lantern_Generator();
    lantern_Generator_set_current_seed(generator, 123456);
    std::cout << "Seed: " << lantern_Generator_current_seed(generator) << std::endl;

    void *qscheme = lantern_QScheme_per_channel_affine();
    std::cout << "QScheme: " << lantern_QScheme_type(qscheme) << std::endl;

    std::vector<int64_t> x(2, 2);
    void *t = lantern_rand_intarrayref_tensoroptions(lantern_vector_int64_t(&x[0], 2), lantern_TensorOptions());
    std::cout << std::string(lantern_Tensor_StreamInsertion(t)) << std::endl;

    void *r = lantern_max_tensor_intt_bool(t, lantern_int64_t(1), lantern_bool(false));
    std::cout << std::string(lantern_Tensor_StreamInsertion(lantern_vector_get(r, 0))) << std::endl;
    std::cout << std::string(lantern_Tensor_StreamInsertion(lantern_vector_get(r, 1))) << std::endl;

    void *dimnamelist = lantern_DimnameList();
    void *dimname = lantern_Dimname(std::string("a").c_str());
    lantern_DimnameList_push_back(dimnamelist, dimname);

    int64_t size = lantern_DimnameList_size(dimnamelist);
    std::cout << "DimnameList size: " << size << std::endl;

    for (int i = 0; i < size; i++)
    {
        std::cout << std::string(lantern_Dimname_to_string(lantern_DimnameList_at(dimnamelist, i))) << std::endl;
    }

    std::vector<int64_t> sizes(2, 1);
    void *a = lantern_rand_intarrayref_dimnamelist_tensoroptions(lantern_vector_int64_t(&sizes[0], 1), dimnamelist, lantern_TensorOptions());

    // zeros_like
    std::vector<int64_t> x1(2, 2);
    void *t1 = lantern_rand_intarrayref_tensoroptions(lantern_vector_int64_t(&x[0], 2), lantern_TensorOptions());

    _test_grad_fn();

    trace_r_nn_module();


    std::cout << "Success!!" << std::endl;
    return 0;
}