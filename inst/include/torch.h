#pragma once
#ifndef __TORCH
#define __TORCH

#include <Rcpp.h>

#include <memory>
#include <string>

#include "torch_api.h"
#include "torch_impl.h"
#include "torch_types.h"
#include "utils.h"

#ifdef IMPORT_TORCH

#include "torch_imports.h"

#else

// R 4.5+ on Linux defaults to dyn.load(now=TRUE) (RTLD_NOW), which rejects
// undefined symbols at load time. The _lantern_* function pointers declared in
// lantern.h are normally 'extern' (no storage), causing "undefined symbol"
// errors. Weak linkage provides actual BSS definitions that satisfy RTLD_NOW,
// while allowing the linker to merge duplicates across translation units.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#define LANTERN_API __attribute__((weak))
#else
#define LANTERN_API extern
#endif
#define LANTERN_HEADERS_ONLY
#include "lantern/lantern.h"

#ifdef FALSE
#undef FALSE
#endif

#ifdef TRUE
#undef TRUE
#endif

#endif  // IMPORT_TORCH
#endif  // __TORCH
