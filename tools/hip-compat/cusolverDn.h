#pragma once
/* cuda -> HIP compat shim (v16): ATen/cuda/CUDAContextLight.h:16 includes
 * <cusolverDn.h> under "#ifdef CUDART_VERSION" (dead here -- the HIP shim
 * defines no CUDART_VERSION, exactly like torch's own ROCm build); kept for
 * robustness. hipSOLVER's dense handle is hipsolverDnHandle_t; no
 * cusolverDnHandle_t alias is shipped. */
#include <hipsolver.h>
using cusolverDnHandle_t = hipsolverDnHandle_t;
