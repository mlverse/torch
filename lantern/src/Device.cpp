#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void *_lantern_Device(const char *type, int64_t index, bool useIndex) {
  LANTERN_FUNCTION_START
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
  } else if (deviceName == "xla") {
    deviceType = torch::DeviceType::XLA;
  } else if (deviceName == "meta") {
    deviceType = torch::DeviceType::Meta;
  }

  torch::Device device = torch::Device(deviceType);

  if (useIndex) {
    device = torch::Device(deviceType, deviceIndex);
  }

  return make_raw::Device(device);
  LANTERN_FUNCTION_END
}

const char *_lantern_Device_type(void *device) {
  LANTERN_FUNCTION_START
  torch::Device type = from_raw::Device(device).type();

  std::string str;
  if (type == torch::DeviceType::CPU) {
    str = "cpu";
  } else if (type == torch::DeviceType::CUDA) {
    str = "cuda";
  } else if (type == torch::DeviceType::MKLDNN) {
    str = "mkldnn";
  } else if (type == torch::DeviceType::OPENGL) {
    str = "opengl";
  } else if (type == torch::DeviceType::OPENCL) {
    str = "opencl";
  } else if (type == torch::DeviceType::IDEEP) {
    str = "ideep";
  } else if (type == torch::DeviceType::HIP) {
    str = "hip";
  } else if (type == torch::DeviceType::FPGA) {
    str = "fpga";
  } else if (type == torch::DeviceType::XLA) {
    str = "xla";
  } else if (type == torch::DeviceType::Meta) {
    str = "meta";
  } else {
    str = "unknown";
  }

  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

int64_t _lantern_Device_index(void *device) {
  LANTERN_FUNCTION_START
  return from_raw::Device(device).index();
  LANTERN_FUNCTION_END_RET(0)
}