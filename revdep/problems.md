# torchvisionlib

<details>

* Version: 0.2.0
* GitHub: NA
* Source code: https://github.com/cran/torchvisionlib
* Date/Publication: 2022-06-14 17:20:02 UTC
* Number of recursive dependencies: 36

Run `revdep_details(, "torchvisionlib")` for more info

</details>

## Newly broken

*   checking whether package ‘torchvisionlib’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00install.out’ for details.
    ```

## Newly fixed

*   checking installed package size ... NOTE
    ```
      installed size is  8.2Mb
      sub-directories of 1Mb or more:
        lib   7.3Mb
    ```

## Installation

### Devel

```
* installing *source* package ‘torchvisionlib’ ...
** package ‘torchvisionlib’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c exports.cpp -o exports.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c torchvisionlib.cpp -o torchvisionlib.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c torchvisionlib_types.cpp -o torchvisionlib_types.o
ccache clang++ -std=gnu++11 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2/Resources/lib -L/usr/local/lib -o torchvisionlib.so RcppExports.o exports.o torchvisionlib.o torchvisionlib_types.o -F/Library/Frameworks/R.framework/Versions/4.2 -framework R -Wl,-framework -Wl,CoreFoundation
installing to /Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib/libs
...
 .onLoad falhou em loadNamespace() para 'torchvisionlib', detalhes:
  chamada: dyn.load(lib_path("torchvision"), local = FALSE)
  erro: impossível carregar objeto compartilhado '/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib//lib/libtorchvision.dylib':
  dlopen(/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib//lib/libtorchvision.dylib, 0x000A): Symbol not found: (__ZN2at14RecordFunctionC1ENS_11RecordScopeEb)
  Referenced from: '/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib/lib/libtorchvision.dylib'
  Expected in: '/Users/dfalbel/Documents/torch/lantern/build/libtorch/lib/libtorch_cpu.dylib'
Erro: loading failed
Execução interrompida
ERROR: loading failed
* removing ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/torchvisionlib’


```
### CRAN

```
* installing *source* package ‘torchvisionlib’ ...
** package ‘torchvisionlib’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/old/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/old/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c exports.cpp -o exports.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/old/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c torchvisionlib.cpp -o torchvisionlib.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/old/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c torchvisionlib_types.cpp -o torchvisionlib_types.o
ccache clang++ -std=gnu++11 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2/Resources/lib -L/usr/local/lib -o torchvisionlib.so RcppExports.o exports.o torchvisionlib.o torchvisionlib_types.o -F/Library/Frameworks/R.framework/Versions/4.2 -framework R -Wl,-framework -Wl,CoreFoundation
installing to /Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/old/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib/libs
...
** testing if installed package can be loaded from temporary location
tentando a URL 'https://github.com/mlverse/torchvisionlib/releases/download/v0.2.0/torchvisionlib-0.2.0+cpu-Darwin.zip'
Content type 'application/octet-stream' length 1508611 bytes (1.4 MB)
==================================================
downloaded 1.4 MB

** checking absolute paths in shared objects and dynamic libraries
** testing if installed package can be loaded from final location
** testing if installed package keeps a record of temporary installation path
* DONE (torchvisionlib)


```
