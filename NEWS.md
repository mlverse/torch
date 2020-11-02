# torch (development version)

- Fixed bug that would happen with functions that need to transform tensors from
  0-based to 1-based in the GPU. (#317)
- Fixed bug when trying to print the `grad_fn` of a Tensor that doesn't have one.
  See (#321)
- Added `$element_size()` method (@dirkschumacher #322)
- Added `$bool()` method (@dirkschumacher #323)
- `torch__addr` and `torch__addr_` have been removed.
- We now check the MD5 hashes of downloaded LibTorch binaries. (@dirkschumacher #325)
- Refactored the optimizers code to avoid duplication of parameter checks, etc. (@dirkschumacher #328)
- Added a Distribution abstract class (@krzjoa #333)
- Updated to LibTorch 1.7 (#337)
- Fixed `torch_argsort` and `x$argsort` to return 1-based indexes (#342)
- Fixed `torch_norm` so it can be called with a `dim` argument. (#345)

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
