# torch (development version)

- Fixed bug in `nn_multihead_attention` when q,k,v inputs not all the same. (@jonathanbratt #540)
- Added parameter to multihead attention module to allow output of unaveraged attention weights. (@jonathanbratt #542)
- We now allow `jit_trace` functions with more than 1 argument. (#544)
- Fixed `$copy_` so it correctly respects the src `requires_grad()` when reloading saved models with `torch_load()`. (#545)

# torch 0.3.0

## Breaking changes

- `torch_nonzero` and `tensor$nonzero()` now return 1-based indexes. (#432)
- Breaking change: `torch_arange` returns in the closed interval `[start, end]` instead of the half open `[start, end)`. This makes it behave similar to R's `seq`. (#506)

## New features

- `torch_split` now accepts a list of sizes as well as a fixed size. (#429)
- Added `nn_layer_norm`. (#435)
- Allow `timeout=360` as `install_torch()` parameter for large file download (@cregouby #438)
- Added `install_torch_from_file()` and `get_install_libs_url()`for setup cases where direct download is not possible (@cregouby #439)
- Added `mean.torch_tensor` (#448)
- New arguments `worker_globals` and `worker_packages` allowing to easily pass objects to workers in parallel dataloaders (#449).
- We now call R garbage collector when there's no memory available on GPU, this can help in a few cases when the laziness of the garbage collector allows too many tensors to be on memory even though they are no longer referenced in R. (#456)
- Implemented `nn_group_norm` and fixed a bug in `nnf_group_norm` (#474)
- Added backend functions allowing us to query which optimizations LibTorch was compiled with (#476)
- Added normal distribution (#462)
- Added bernoulli distribution (#484)
- `as.list` for `nn_modules` (#492)
- Enumerate support in Bernoulli distribution (#490)
- Added Poisson Distriibution (#495)
- Allow optional .getbatch in datasets/dataloaders (#498)
- `nn_lstm`, `nn_gru` and `nn_gru` can now use cudnn accelerations when available (#503).
- Added Gamma distribution (#489)
- We now respect the TORCH_HOME env var to automatically install torch. (#522)
- Implement comparison operator `!=` for torch dtypes. (#524)
- Added Chi-square distribution. (#518)
- Added `optimizer` function allowing to easily implement custom optimizers. (#527)

## Bug fixes

- Fixed bug in `optim_lbfgs` that would make model objects exponentially big. (#431)
- Correctly handle `NaN`s in L-BFGS optimizer (#433)
- The default collate function now respects the data type when converting to a tensor (if the dataset returns an R object) (#434)
- Fixed `torch_normal`. (#450)
- Fixed backward compatibility issue when loading models saved in older versions of torch. This bug was introduced in #452 and is now fixed and we also added a regression test. (#458)
- Fixed bug when using RNN's on the GPU (#460)
- Found and fixed some memory leaks, specially when creating datatypes from strings and when saving models with `torch_save`. (#454)
- Fixed bug in `nnf_pad` when using `mode='circular'`. (#471)
- Bugfixes in `nn_multihead_attention` (#496)
- Fixed bug when using packed sequences with `nn_lstm` (#500)
- Fixed bug in the `to` method of `nn_module` that would reset the `requires_grad` attribute of parameters. (#501)
- Added `strong_wolfe` option to `optim_lbfgs`. (#517)
- Fixed default argument of `nn_init_trunc_normal_` initializer function. (#535)

## Documentation

- Added vignette on reading models from Python (#469)

## Internal changes

- Removed the PerformanceReporter from tests to get easier to read stack traces. (#449)
- Internal change in the R7 classes so R7 objects are simple external pointer instead of environments. This might cause breaking change if you relied on saving any kind of state in the Tensor object. (#452)
- Internal refactoring making Rcpp aware of some XPtrTorch* types so making it simpler to return them from Rcpp code. This might cause a breaking change if you are relying on `torch_dtype()` being an R6 class. (#451) 
- Internal changes to auto unwrap arguments from SEXP's in Rcpp. This will make easier to move the dispatcher system to C++ in the future, but already allows us to gain ~30% speedups in small operations. (#454)
- Added a Windows GPU CI workflow (#508).
- Update to LibTorch v1.8 (#513) 
- Moved some parts of the dispatcher to C++ to make it faster. (#520)

# torch 0.2.1

## Breaking changes

- Made `torch_one_hot` and `nnf_one_hot` use 1-based indexing. (#410)
- `nn_module$eval()` and `nn_module$train()` now return a callable `nn_module` instead of a `nn_Module`. (#425)

## New features

- Added a custom CPU allocator to call `gc` when torch might need more memory (#402)
- Updated to LibTorch 1.7.1 (#412)
- Allow listing all nested modules in a `nn_module` (#417)
- Allow modifying the `requires_grad` attribute using the `$<-` operator (#419)
- Added `length` method for the `nn_sequential` container. (#423)
- Added support for CUDA 11 on linux (#424)

## Bug fixes

- Fix support for cuda 9.2 (#398)
- Fixed GPU CI that was skipping tests. (#398)
- Fixed a memory leak when printing tensors (#402)
- Fixed a memory leak when passing integer vectors to lantern. (#402)
- Fixed a few more memory leaks related to autograd context (#405)
- Fixed `nnf_normalize` and `x$norm()` as they were not able to be called (#409)

## Documentation

- Small improvement to `nn_module` documentation (#399).
- The getting started section has been removed from the pkgdown website in favor of the new guide in the landing page (#401)
- Updated the landing page to include a getting started tutorial (#400)

# torch 0.2.0

## Breaking changes

- Dataloaders now returns a `coro::exhausted` intead of raising `stop_iteration_error` when the dataloader exceeds. (#366)
- Fixed bug that would happen with functions that need to transform tensors from
  0-based to 1-based in the GPU. (#317)
- Fixed `torch_argsort` and `x$argsort` to return 1-based indexes (#342)
- Fixed `torch_argmax`, `torch_argmin`, `x$argmax()` and `x$argmin()` return 1-based indexes. (#389)

## New features

- Added `$element_size()` method (@dirkschumacher #322)
- Added `$bool()` method (@dirkschumacher #323)
- `torch__addr` and `torch__addr_` have been removed as they are no longer available in LibTorch 1.7.
- We now check the MD5 hashes of downloaded LibTorch binaries. (@dirkschumacher #325)
- Added a Distribution abstract class (@krzjoa #333)
- Updated to LibTorch 1.7 (#337)
- We now warn when converting `long` tensors to R and there's a chance of an integer overflow. (#347)
- Allow `private` and `active` methods in `nn_module`'s and `dataset`'s. (#349)
- Added `nn_batch_norm3d` (@mattwarkentin #354)
- Added `nn_lstm` and `nn_gru` modules. (#362)
- Added distribution constraints (@krzjoa #364)
- Dataloaders now use the num_workers argument to load data in parallel (#366)
- Added Exponential Family classs to distributions (#373)
- Added Dockerfile and docker compose file with GPU support, with a how-to guide. (#380 #386)
- Added R 3.6 to the CI system and fixed compilation from source with it on Windows (#387)
- Initial support for JIT tracing (#377)
- Added LBFGS optimizer (#392)
- Improved the `nn_module` UI by improving autocomplete support and adding a print method (#391)

## Bug fixes

- Fixed bug when trying to print the `grad_fn` of a Tensor that doesn't have one.
  See (#321)
- Refactored the optimizers code to avoid duplication of parameter checks, etc. (@dirkschumacher #328)
- Fixed `torch_norm` so it can be called with a `dim` argument. (#345)
- Fixed crash when calling `torch_hann_window` with an invalid `NULL` `window_length`. (#351)
- Fixed `torch_stft` calls for LibTorch 1.7 (added the `return_complex` argument) (#355)
- Fixed bug when strides were NULL in some pooling operations. (#361)
- Use `nvcc --version` instead of `nvidia-smi` to find the CUDA version as `nvidia-smi` reports the latest supported version and not the installed one. (#363)
- Corrected URL to download LibTorch under Linux with CUDA 10.2 (#367)
- Fixed handling of integer tensors when indexing tensors (#385)
- Fixed bug when passing length zero vectors to lantern/libtorch. (#388)

# torch 0.1.1

## Bug fixes

- Fixed bug that made `RandomSampler(replacement = TRUE)` to never take the last
  element in the dataset. (84861fa)
- Fixed `torch_topk` and `x$topk` so the returned indexes are 1-based (#280)
- Fixed a bug (#275) that would cause `1 - torch_tensor(1, device = "cuda")` to 
  fail because `1` was created in the CPU. (#279)
- We now preserve names in the `dataloader` output (#286)
- `torch_narrow`, `Tensor$narrow()` and `Tensor$narrow_copy` are now indexed 
  starting at 1. (#294)
- `Tensor$is_leaf` is now an active method. (#295)
- Fixed bug when passing equations to `torch_einsum`. (#296)  
- Fixed `nn_module_list()` to correctly name added modules, otherwise they are not
  returned when doing `state_dict()` on it. (#300)
- Fixed bug related to random number seeds when using in-place methods. (#303)
- Fixed `nn_batchnorm*` so it returns the same results as PyTorch (#302)
- Fixed a bug that made `nn_module$parameter` when there were shared parameters
  between layers. (#306)
- Fixed `$max` and `$min` to return 1-based indexes. (#315)

## New features

- Expanded the `utils_data_default_collate` to support converting R objects to
  torch tensors when needed. (#269) 
- Added an `as.matrix` method for torch Tensors. (#282)
- By default we now truncate the output of `print(totrch_tensor(1:40))` if it
  spans for more than 30 lines. This is useful for not spamming the console or
  taking very long to print when you print a very large tensor. (#283)
- Added the Adadelta optimizer (@krzjoa #284)
- Added support for GPU's on Windows (#281)
- Added the Adagrad optimizer (@krzjoa #289)
- Added RMSprop optimizer (@krzjoa #290)
- Added the Rprop optimizer (@krzjoa #297)
- Added gradient clipping utilities (#299)
- Added `nnf_contrib_sparsemax` and `nn_contrib_sparsemax`. (#309)
- Added ASGD optimizer (@krzjoa #307)
- Getters and setters for the number of threads used by torch (#311)

# torch 0.1.0

- Added many missing losses (#252)
- Implemented the `$<-` and `[[<-` operators for the `nn_module` class. (#253)
- Export `nn_parameter`, `nn_buffer`, and `is_*` auxiliary functions.
- Added a new serialization vignette.
- Added a few learning rate schedulers (#258)

# torch 0.0.2

- Added a `NEWS.md` file to track changes to the package.
- Auto install when loading the package for the first time.
