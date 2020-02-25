#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void* lantern_Device(const char* type, int64_t index, bool useIndex)
{
  std::string deviceName(type);
  torch::DeviceType deviceType = torch::DeviceType::CPU;
  torch::DeviceIndex deviceIndex = index;
  
  if (deviceName == "cuda") {
    deviceType = torch::DeviceType::CUDA;
  } else if (deviceName == "mkldnn") {
    deviceType = torch::DeviceType::MKLDNN;
  } else if (deviceName == "opengl") {
    deviceType = torch::DeviceType::OPENGL;
  } else if (deviceName == "opencl") {
    deviceType = torch::DeviceType::OPENCL;
  } else if (deviceName == "ideep") {
    deviceType = torch::DeviceType::IDEEP;
  } else if (deviceName == "hip") {
    deviceType = torch::DeviceType::HIP;
  } else if (deviceName == "fpga") {
    deviceType = torch::DeviceType::FPGA;
  } else if (deviceName == "msnpu") {
    deviceType = torch::DeviceType::MSNPU;
  } else if (deviceName == "xla") {
    deviceType = torch::DeviceType::XLA;
  } else if (deviceName == "test") {
    deviceType = torch::DeviceType::ONLY_FOR_TEST;
  }
  
  torch::Device device = torch::Device(deviceType);
  
  if (useIndex) {
    device = torch::Device(deviceType, deviceIndex);
  }
  
  return (void *) new LanternPtr<torch::Device>(device);
}

const char* lantern_Device_type(void* device)
{
  torch::Device dev = ((LanternPtr<torch::Device>*)device)->get();
  torch::DeviceType type = dev.type();
  return torch::DeviceTypeName(type, true).c_str();
}

int64_t lantern_Device_index(void* device)
{
  return ((LanternPtr<torch::Device>*)device)->get().index();
}