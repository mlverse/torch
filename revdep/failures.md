# COTAN

<details>

* Version: 2.6.2
* GitHub: https://github.com/seriph78/COTAN
* Source code: https://github.com/cran/COTAN
* Date/Publication: 2025-01-19
* Number of recursive dependencies: 276

Run `revdepcheck::revdep_details(, "COTAN")` for more info

</details>

## In both

*   checking examples ... ERROR
    ```
    Running examples in ‘COTAN-Ex.R’ failed
    The error most likely occurred in:
    
    > ### Name: COTAN_ObjectCreation
    > ### Title: 'COTAN' shortcuts
    > ### Aliases: COTAN_ObjectCreation COTAN proceedToCoex,COTAN-method
    > ###   proceedToCoex automaticCOTANObjectCreation
    > 
    > ### ** Examples
    > 
    ...
    Cotan genes' coex estimation started
    Warning in canUseTorch(optimizeForSpeed, deviceStr) :
      The `torch` library could not find any `CUDA` device
    Warning in canUseTorch(optimizeForSpeed, deviceStr) :
      Falling back to CPU calculations
    Calculate genes coex (torch) on device cpu: START
    Error in cpp_cuda_empty_cache() : 
      `cuda_empty_cache` is only supported on CUDA runtimes.
    Calls: automaticCOTANObjectCreation ... calculateCoex_Torch -> <Anonymous> -> cpp_cuda_empty_cache
    Execution halted
    ```

*   R CMD check timed out
    

*   checking installed package size ... NOTE
    ```
      installed size is 13.7Mb
      sub-directories of 1Mb or more:
        doc  11.8Mb
    ```

*   checking dependencies in R code ... NOTE
    ```
    'library' or 'require' call to ‘torch’ in package code.
      Please use :: or requireNamespace() instead.
      See section 'Suggested packages' in the 'Writing R Extensions' manual.
    Unexported object imported by a ':::' call: ‘ggplot2:::ggname’
      See the note in ?`:::` about the use of this operator.
    ```

*   checking R code for possible problems ... NOTE
    ```
    mergeUniformCellsClusters : fromMergedName: warning in
      vapply(currentClNames, function(clName, mergedName) {: partial
      argument match of 'FUN.VAL' to 'FUN.VALUE'
    mergeUniformCellsClusters : fromMergedName: warning in
      return(str_detect(mergedName, clName)): partial argument match of
      'FUN.VAL' to 'FUN.VALUE'
    mergeUniformCellsClusters : fromMergedName: warning in }, FUN.VAL =
      logical(1L), mergedClName): partial argument match of 'FUN.VAL' to
      'FUN.VALUE'
    ECDPlot: no visible binding for global variable ‘.’
    ...
      ‘clusterData’
    Undefined global functions or variables:
      . .x CellNumber Cluster Condition ExpGenes GCS GDI PC1 PC2 UDEPLot a
      bGroupGenesPlot cl1 cl2 clName1 clName2 clusterData clusters coex
      condName conditions expectedN expectedNN expectedNY expectedYN
      expectedYY g2 group hk keys lambda means mit.percentage n nu nuPlot
      obj objSeurat observedNN observedNY observedY observedYN observedYY
      pcaCellsPlot permMap rankGenes rawNorm secondaryMarkers sum.raw.norm
      type types useTorch usedMaxResolution values violinwidth width x xmax
      xmaxv xminv y zoomedNuPlot
    ```

# proteus

<details>

* Version: 1.1.4
* GitHub: NA
* Source code: https://github.com/cran/proteus
* Date/Publication: 2023-10-21 17:40:02 UTC
* Number of recursive dependencies: 137

Run `revdepcheck::revdep_details(, "proteus")` for more info

</details>

## In both

*   R CMD check timed out
    

# tabnet

<details>

* Version: 0.6.0
* GitHub: https://github.com/mlverse/tabnet
* Source code: https://github.com/cran/tabnet
* Date/Publication: 2024-06-15 18:20:02 UTC
* Number of recursive dependencies: 173

Run `revdepcheck::revdep_details(, "tabnet")` for more info

</details>

## In both

*   checking tests ...
    ```
      Running ‘spelling.R’
      Running ‘testthat.R’
     ERROR
    Running the tests in ‘tests/testthat.R’ failed.
    Last 13 lines of output:
        4.       └─testthat:::test_files_parallel(...)
        5.         ├─withr::with_dir(...)
        6.         │ └─base::force(code)
        7.         ├─testthat::with_reporter(...)
        8.         │ └─base::tryCatch(...)
    ...
        9.         │   └─base (local) tryCatchList(expr, classes, parentenv, handlers)
       10.         │     └─base (local) tryCatchOne(expr, names, parentenv, handlers[[1L]])
       11.         │       └─base (local) doTryCatch(return(expr), name, parentenv, handler)
       12.         └─testthat:::parallel_event_loop_chunky(queue, reporters, ".")
       13.           └─queue$poll(Inf)
       14.             └─base::lapply(...)
       15.               └─testthat (local) FUN(X[[i]], ...)
       16.                 └─private$handle_error(msg, i)
       17.                   └─rlang::abort(...)
      Execution halted
    ```

*   R CMD check timed out
    

# torchdatasets

<details>

* Version: 0.3.1
* GitHub: https://github.com/mlverse/torchdatasets
* Source code: https://github.com/cran/torchdatasets
* Date/Publication: 2024-06-20 12:40:01 UTC
* Number of recursive dependencies: 71

Run `revdepcheck::revdep_details(, "torchdatasets")` for more info

</details>

## In both

*   R CMD check timed out
    

# torchvisionlib

<details>

* Version: 0.5.0
* GitHub: https://github.com/mlverse/torchvisionlib
* Source code: https://github.com/cran/torchvisionlib
* Date/Publication: 2024-02-15 19:20:02 UTC
* Number of recursive dependencies: 31

Run `revdepcheck::revdep_details(, "torchvisionlib")` for more info

</details>

## Newly broken

*   checking whether package ‘torchvisionlib’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00install.out’ for details.
    ```

## Installation

### Devel

```
* installing *source* package ‘torchvisionlib’ ...
** package ‘torchvisionlib’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
using C++ compiler: ‘Apple clang version 16.0.0 (clang-1600.0.26.3)’
using SDK: ‘MacOSX15.1.sdk’
clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2   -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2   -c exports.cpp -o exports.o
clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2   -c torchvisionlib.cpp -o torchvisionlib.o
clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2   -c torchvisionlib_types.cpp -o torchvisionlib_types.o
...
 .onLoad failed in loadNamespace() for 'torchvisionlib', details:
  call: dyn.load(lib_path("torchvision"), local = FALSE)
  error: unable to load shared object '/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib//lib/libtorchvision.dylib':
  dlopen(/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib//lib/libtorchvision.dylib, 0x000A): Symbol not found: __ZN2at4_ops10zeros_like4callERKNS_6TensorEN3c108optionalINS5_10ScalarTypeEEENS6_INS5_6LayoutEEENS6_INS5_6DeviceEEENS6_IbEENS6_INS5_12MemoryFormatEEE
  Referenced from: <7770DCC4-8CD6-3A42-8643-D57BF69E720D> /Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib/lib/libtorchvision.dylib
  Expected in:     <30A54DCC-A845-368C-8B03-ADCDD3639E86> /Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/lib/libtorch_cpu.dylib
Error: loading failed
Execution halted
ERROR: loading failed
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/torchvisionlib’


```
### CRAN

```
* installing *source* package ‘torchvisionlib’ ...
** package ‘torchvisionlib’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
using C++ compiler: ‘Apple clang version 16.0.0 (clang-1600.0.26.3)’
using SDK: ‘MacOSX15.1.sdk’
clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/torch/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2   -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/torch/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2   -c exports.cpp -o exports.o
clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/torch/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2   -c torchvisionlib.cpp -o torchvisionlib.o
clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/torch/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2   -c torchvisionlib_types.cpp -o torchvisionlib_types.o
...
** testing if installed package can be loaded from temporary location
trying URL 'https://github.com/mlverse/torchvisionlib/releases/download/v0.5.0/torchvisionlib-0.5.0+cpu+arm64-Darwin.zip'
Content type 'application/octet-stream' length 928798 bytes (907 KB)
==================================================
downloaded 907 KB

** checking absolute paths in shared objects and dynamic libraries
** testing if installed package can be loaded from final location
** testing if installed package keeps a record of temporary installation path
* DONE (torchvisionlib)


```
