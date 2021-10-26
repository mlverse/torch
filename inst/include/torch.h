#pragma once

#include <string>
#include <memory>
#include <RcppCommon.h>
#include <Rcpp.h>
#include "torch_types.h"
#include "torch_api.h"
#include "utils.h"
#include "torch_impl.h"

#ifdef IMPORT_TORCH

#include "torch_imports.h"

#else

#define LANTERN_HEADERS_ONLY
#include "lantern/lantern.h"

#endif // IMPORT_TORCH


