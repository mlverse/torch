#pragma once
#ifndef __TORCH
#define __TORCH

#include <Rcpp.h>
#include <memory>
#include <string>
#include "torch_types.h"
#include "torch_api.h"
#include "utils.h"
#include "torch_impl.h"

#ifdef IMPORT_TORCH

#include "torch_imports.h"

#else

#define LANTERN_HEADERS_ONLY
#include "lantern/lantern.h"

#ifdef FALSE
#undef FALSE
#endif

#ifdef TRUE
#undef TRUE
#endif

#endif // IMPORT_TORCH
#endif // __TORCH

