#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"

bool _lantern_backend_has_mkl() { return at::hasMKL(); }

bool _lantern_backend_has_mkldnn() { return at::hasMKLDNN(); }

bool _lantern_backend_has_openmp() { return at::hasOpenMP(); }

bool _lantern_backend_has_lapack() { return at::hasLAPACK(); }