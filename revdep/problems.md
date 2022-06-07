# torchvisionlib

<details>

* Version: 0.1.0
* GitHub: NA
* Source code: https://github.com/cran/torchvisionlib
* Date/Publication: 2022-03-07 20:10:02 UTC
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
      installed size is  7.5Mb
      sub-directories of 1Mb or more:
        lib   6.6Mb
    ```

## Installation

### Devel

```
* installing *source* package ‘torchvisionlib’ ...
** package ‘torchvisionlib’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c exports.cpp -o exports.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c torchvisionlib.cpp -o torchvisionlib.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c torchvisionlib_types.cpp -o torchvisionlib_types.o
ccache clang++ -std=gnu++11 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Resources/lib -L/usr/local/lib -o torchvisionlib.so RcppExports.o exports.o torchvisionlib.o torchvisionlib_types.o -F/Library/Frameworks/R.framework/.. -framework R -Wl,-framework -Wl,CoreFoundation
installing to /Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib/libs
...
 .onLoad falhou em loadNamespace() para 'torchvisionlib', detalhes:
  chamada: dyn.load(lib_path("torchvision"), local = FALSE)
  erro: impossível carregar objeto compartilhado '/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib//lib/libtorchvision.dylib':
  dlopen(/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib//lib/libtorchvision.dylib, 0x000A): Symbol not found: __ZN5torch2nn6Module9zero_gradEv
  Referenced from: /Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/new/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib/lib/libtorchvision.dylib
  Expected in: /Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/torch/lib/libtorch_cpu.dylib
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
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/old/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/old/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c exports.cpp -o exports.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/old/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c torchvisionlib.cpp -o torchvisionlib.o
ccache clang++ -std=gnu++11 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchvisionlib/Rcpp/include' -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/old/torch/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c torchvisionlib_types.cpp -o torchvisionlib_types.o
ccache clang++ -std=gnu++11 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Resources/lib -L/usr/local/lib -o torchvisionlib.so RcppExports.o exports.o torchvisionlib.o torchvisionlib_types.o -F/Library/Frameworks/R.framework/.. -framework R -Wl,-framework -Wl,CoreFoundation
installing to /Users/dfalbel/Documents/torch/revdep/checks.noindex/torchvisionlib/old/torchvisionlib.Rcheck/00LOCK-torchvisionlib/00new/torchvisionlib/libs
...
  problem creating directory /var/folders/x0/fqbv9_ys1lq55xqjqbqllzqm0000gn/T//RtmpEWNZKW/file72f951cc6ced/torchvisionlib-0.1.0+cpu-Darwin/share: No space left on device
** checking absolute paths in shared objects and dynamic libraries
** testing if installed package can be loaded from final location
tentando a URL 'https://github.com/mlverse/torchvisionlib/releases/download/v0.1.0/torchvisionlib-0.1.0+cpu-Darwin.zip'
Content type 'application/octet-stream' length 1378091 bytes (1.3 MB)
==================================================
downloaded 1.3 MB

** testing if installed package keeps a record of temporary installation path
* DONE (torchvisionlib)


```
