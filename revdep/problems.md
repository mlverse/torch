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
