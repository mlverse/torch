#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
std::string cpp_device_type_to_string (Rcpp::XPtr<torch::Device> device) {
  
  torch::DeviceType device_type = device->type();
  
  if (device_type == torch::DeviceType::CPU) {
    return "cpu";
  } else if (device_type == torch::DeviceType::CUDA) {
    return "cuda";
  }
  Rcpp::stop("DeviceType not handled");
}

// [[Rcpp::export]]
std::int64_t cpp_device_index_to_int (Rcpp::XPtr<torch::Device> device) {
  torch::DeviceIndex device_index = device->index();
  return device_index;
}

torch::DeviceType device_type_from_string (std::string x) {
  if (x == "cpu") {
    return torch::DeviceType::CPU;
  } else if (x == "cuda") {
    return torch::DeviceType::CUDA;
  }
  Rcpp::stop("DeviceType not handled");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Device> cpp_torch_device (std::string type, Rcpp::Nullable<std::int64_t> index) {
  
  auto device_type = device_type_from_string(type);
  torch::Device device = torch::Device(device_type);
  
  if (index.isNotNull()) {
    torch::DeviceIndex device_index = Rcpp::as<std::int64_t>(index);
    device = torch::Device(device_type, device_index);
  }
  
  return make_xptr<torch::Device>(device);
}


Rcpp::XPtr<torch::Device> make_device_ptr (torch::Device x) {
  auto * out = new torch::Device(x);
  return Rcpp::XPtr<torch::Device>(out);
}

// Device Index --------

std::int64_t device_index_to_int (torch::DeviceIndex x) {
  return x;
}

torch::DeviceIndex device_index_from_int (std::int64_t x) {
  return x;
}

// Device Type ---------

std::string device_type_to_string (torch::DeviceType x) {
  if (x == torch::DeviceType::CPU) {
    return "cpu";
  } else if (x == torch::DeviceType::CUDA) {
    return "cuda";
  }
  Rcpp::stop("DeviceType not handled");
}

// torch::DeviceType device_type_from_string (std::string x) {
//   if (x == "cpu") {
//     return torch::DeviceType::CPU;
//   } else if (x == "cuda") {
//     return torch::DeviceType::CUDA;
//   }
//   Rcpp::stop("DeviceType not handled");
// }


// Device


// Device attributes

// index

// [[Rcpp::export]]
std::int64_t get_device_index (Rcpp::XPtr<torch::Device> device) {
  return device_index_to_int(device->index());
}

// type

// [[Rcpp::export]]
std::string get_device_type (Rcpp::XPtr<torch::Device> device) {
  return device_type_to_string(device->type());
}

// Device methods

// has_index

// [[Rcpp::export]]
bool device_has_index (Rcpp::XPtr<torch::Device> device) {
  return device->has_index();
}

// is_cuda

// [[Rcpp::export]]
bool device_is_cuda (Rcpp::XPtr<torch::Device> device) {
  return device->is_cuda();
}

// is_cpu

// [[Rcpp::export]]
bool device_is_cpu (Rcpp::XPtr<torch::Device> device) {
  return device->is_cpu();
}

// ==

// [[Rcpp::export]]
bool device_equals (Rcpp::XPtr<torch::Device> device1, Rcpp::XPtr<torch::Device> device2) {
  return (*device1) == (*device2);
}

// set_index

// [[Rcpp::export]]
void device_set_index (Rcpp::XPtr<torch::Device> device, std::int64_t index) {
  device->set_index(device_index_from_int(index));
}

// Create a device

torch::Device device_from_string(std::string device) {
  return torch::Device(device_type_from_string(device));
}

std::string device_to_string (torch::Device x) {
  if (x.is_cpu()) {
    return "CPU";
  } else if (x.is_cuda()){
    return "CUDA";
  };
  Rcpp::stop("not handled");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Device> device_from_r(std::string device, Rcpp::Nullable<std::int64_t> index) {
  if (index.isNull()) {
    return make_device_ptr(torch::Device(device_type_from_string(device)));
  } else {
    return make_device_ptr(torch::Device(device_type_from_string(device), device_index_from_int(Rcpp::as<std::int64_t>(index))));
  }
}
