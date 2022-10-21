# lambdaTS

<details>

* Version: 1.1
* GitHub: NA
* Source code: https://github.com/cran/lambdaTS
* Date/Publication: 2022-02-20 20:00:02 UTC
* Number of recursive dependencies: 126

Run `revdep_details(, "lambdaTS")` for more info

</details>

## In both

*   checking whether package ‘lambdaTS’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/lambdaTS/new/lambdaTS.Rcheck/00install.out’ for details.
    ```

## Installation

### Devel

```
* installing *source* package ‘lambdaTS’ ...
** package ‘lambdaTS’ successfully unpacked and MD5 sums checked
** using staged installation
** R
** data
*** moving datasets to lazyload DB
** byte-compile and prepare package for lazy loading
Error in loadNamespace(j <- i[[1L]], c(lib.loc, .libPaths()), versionCheck = vI[[j]]) : 
  there is no package called ‘clue’
Calls: <Anonymous> ... loadNamespace -> withRestarts -> withOneRestart -> doWithOneRestart
Execução interrompida
ERROR: lazy loading failed for package ‘lambdaTS’
* removing ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/lambdaTS/new/lambdaTS.Rcheck/lambdaTS’


```
### CRAN

```
* installing *source* package ‘lambdaTS’ ...
** package ‘lambdaTS’ successfully unpacked and MD5 sums checked
** using staged installation
** R
** data
*** moving datasets to lazyload DB
** byte-compile and prepare package for lazy loading
Error in loadNamespace(j <- i[[1L]], c(lib.loc, .libPaths()), versionCheck = vI[[j]]) : 
  there is no package called ‘clue’
Calls: <Anonymous> ... loadNamespace -> withRestarts -> withOneRestart -> doWithOneRestart
Execução interrompida
ERROR: lazy loading failed for package ‘lambdaTS’
* removing ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/lambdaTS/old/lambdaTS.Rcheck/lambdaTS’


```
# proteus

<details>

* Version: 1.0.0
* GitHub: NA
* Source code: https://github.com/cran/proteus
* Date/Publication: 2021-06-24 11:50:02 UTC
* Number of recursive dependencies: 100

Run `revdep_details(, "proteus")` for more info

</details>

## In both

*   checking whether package ‘proteus’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/proteus/new/proteus.Rcheck/00install.out’ for details.
    ```

## Installation

### Devel

```
* installing *source* package ‘proteus’ ...
** package ‘proteus’ successfully unpacked and MD5 sums checked
** using staged installation
** R
** data
*** moving datasets to lazyload DB
** byte-compile and prepare package for lazy loading
Error in loadNamespace(j <- i[[1L]], c(lib.loc, .libPaths()), versionCheck = vI[[j]]) : 
  there is no package called ‘clue’
Calls: <Anonymous> ... loadNamespace -> withRestarts -> withOneRestart -> doWithOneRestart
Execução interrompida
ERROR: lazy loading failed for package ‘proteus’
* removing ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/proteus/new/proteus.Rcheck/proteus’


```
### CRAN

```
* installing *source* package ‘proteus’ ...
** package ‘proteus’ successfully unpacked and MD5 sums checked
** using staged installation
** R
** data
*** moving datasets to lazyload DB
** byte-compile and prepare package for lazy loading
Error in loadNamespace(j <- i[[1L]], c(lib.loc, .libPaths()), versionCheck = vI[[j]]) : 
  there is no package called ‘clue’
Calls: <Anonymous> ... loadNamespace -> withRestarts -> withOneRestart -> doWithOneRestart
Execução interrompida
ERROR: lazy loading failed for package ‘proteus’
* removing ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/proteus/old/proteus.Rcheck/proteus’


```
# tabnet

<details>

* Version: 0.3.0
* GitHub: https://github.com/mlverse/tabnet
* Source code: https://github.com/cran/tabnet
* Date/Publication: 2021-10-11 17:00:02 UTC
* Number of recursive dependencies: 152

Run `revdep_details(, "tabnet")` for more info

</details>

## In both

*   R CMD check timed out
    

# torchaudio

<details>

* Version: 0.2.0
* GitHub: NA
* Source code: https://github.com/cran/torchaudio
* Date/Publication: 2021-05-05 15:20:07 UTC
* Number of recursive dependencies: 82

Run `revdep_details(, "torchaudio")` for more info

</details>

## In both

*   checking whether package ‘torchaudio’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchaudio/new/torchaudio.Rcheck/00install.out’ for details.
    ```

## Installation

### Devel

```
* installing *source* package ‘torchaudio’ ...
** package ‘torchaudio’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
ccache clang++ -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
ccache clang++ -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c read_mp3.cpp -o read_mp3.o
ccache clang++ -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c read_wav.cpp -o read_wav.o
In file included from read_wav.cpp:2:
./AudioFile.h:56:6: warning: scoped enumerations are a C++11 extension [-Wc++11-extensions]
enum class AudioFileFormat
...
    audioFileFormat = determineAudioFileFormat (fileData);
                      ^
read_wav.cpp:9:7: note: in instantiation of member function 'AudioFile<float>::load' requested here
  wav.load(filepath.c_str());
      ^
28 warnings and 9 errors generated.
make: *** [read_wav.o] Error 1
make: *** Waiting for unfinished jobs....
ERROR: compilation failed for package ‘torchaudio’
* removing ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchaudio/new/torchaudio.Rcheck/torchaudio’


```
### CRAN

```
* installing *source* package ‘torchaudio’ ...
** package ‘torchaudio’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
ccache clang++ -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
ccache clang++ -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c read_mp3.cpp -o read_mp3.o
ccache clang++ -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c read_wav.cpp -o read_wav.o
In file included from read_wav.cpp:2:
./AudioFile.h:56:6: warning: scoped enumerations are a C++11 extension [-Wc++11-extensions]
enum class AudioFileFormat
...
    audioFileFormat = determineAudioFileFormat (fileData);
                      ^
read_wav.cpp:9:7: note: in instantiation of member function 'AudioFile<float>::load' requested here
  wav.load(filepath.c_str());
      ^
28 warnings and 9 errors generated.
make: *** [read_wav.o] Error 1
make: *** Waiting for unfinished jobs....
ERROR: compilation failed for package ‘torchaudio’
* removing ‘/Users/dfalbel/Documents/torch/revdep/checks.noindex/torchaudio/old/torchaudio.Rcheck/torchaudio’


```
# torchdatasets

<details>

* Version: 0.1.0
* GitHub: https://github.com/mlverse/torchdatasets
* Source code: https://github.com/cran/torchdatasets
* Date/Publication: 2021-10-07 16:20:02 UTC
* Number of recursive dependencies: 65

Run `revdep_details(, "torchdatasets")` for more info

</details>

## In both

*   R CMD check timed out
    

*   checking dependencies in R code ... NOTE
    ```
    Namespaces in Imports field not imported from:
      ‘stringr’ ‘torch’ ‘torchvision’ ‘zip’
      All declared Imports should be used.
    ```

# torchvision

<details>

* Version: 0.4.1
* GitHub: https://github.com/mlverse/torchvision
* Source code: https://github.com/cran/torchvision
* Date/Publication: 2022-01-28 20:20:02 UTC
* Number of recursive dependencies: 43

Run `revdep_details(, "torchvision")` for more info

</details>

## In both

*   R CMD check timed out
    

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
