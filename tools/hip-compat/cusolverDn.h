#pragma once
/* cuda -> HIP compat shim (make_rocm_kit.sh v20): ATen/cuda/CUDAContextLight.h:16
 * includes <cusolverDn.h> under "#ifdef CUDART_VERSION" (dead here -- the HIP
 * shims define no CUDART_VERSION, exactly like torch's own ROCm build); kept
 * for robustness in case a future header includes it unguarded. The cusolver
 * types are declared SELF-CONTAINED by the cusparse.h shim (which every
 * consumer includes first) -- alias onto the hipsolver types only when this
 * header is reached STANDALONE (cusparse.h not seen yet). */
#include <hipsolver.h>
#ifndef LANTERN_HIP_COMPAT_CUSOLVERDN_TYPES
using cusolverDnHandle_t = hipsolverDnHandle_t;
using cusolverStatus_t = hipsolverStatus_t;
#endif
