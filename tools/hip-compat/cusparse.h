#pragma once
/* cuda -> HIP compat shim (make_rocm_kit.sh v20): ATen/cuda/CUDAContextLight.h:8
 * includes <cusparse.h> unguarded. hipSPARSE ships no cusparse* compat
 * typedefs (hipsparseHandle_t is void*, verified against ROCm 6.4.4). */
#include <hipsparse.h>
using cusparseHandle_t = hipsparseHandle_t;
/* ATen/cuda/Exceptions.h:71 declares cusparseGetErrorString(cusparseStatus_t);
 * alias the STATUS type too, and CUSPARSE_STATUS_SUCCESS for the
 * AT_CUSPARSE_CHECK macro that downstream TUs expand (hipsparseStatus_t is an
 * enum; HIPSPARSE_STATUS_SUCCESS ships in every hipsparse version, flat 6.x
 * and per-component 7.x alike). */
using cusparseStatus_t = hipsparseStatus_t;
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
/* ATen/cuda/CUDAContextLight.h:95 declares getCurrentCUDASolverDnHandle()
 * returning cusolverDnHandle_t behind
 * "#if defined(CUDART_VERSION) || defined(USE_ROCM)" (USE_ROCM is pinned by
 * the kit) while its own include of <cusolverDn.h> (L16) stays DEAD under
 * "#ifdef CUDART_VERSION" -- and ATen/cuda/Exceptions.h:112 declares
 * cusolverGetErrorMessage(cusolverStatus_t). This shim is included by BOTH
 * torch headers BEFORE <hipsolver/hipsolver.h> (CUDAContextLight.h:8 vs :24,
 * Exceptions.h:4 vs :10), so the hipsolver types are not usable yet: declare
 * the cusolver types SELF-CONTAINED. cusolverDnHandle_t is a plain opaque
 * pointer and cusolverStatus_t a plain int -- ABI-identical to
 * hipsolverDnHandle_t/hipsolverStatus_t (handles are pointers, statuses are
 * C enums), no coupling to the hipsolver struct tags, no ordering hazard.
 * lantern never CALLS getCurrentCUDASolverDnHandle/cusolverGetErrorMessage;
 * only the declarations must compile. */
typedef struct cusolverDnContext *cusolverDnHandle_t;
typedef int cusolverStatus_t;
#define LANTERN_HIP_COMPAT_CUSOLVERDN_TYPES 1
