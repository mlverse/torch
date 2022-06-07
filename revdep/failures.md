# torchaudio

<details>

* Version: 0.2.0
* GitHub: NA
* Source code: https://github.com/cran/torchaudio
* Date/Publication: 2021-05-05 15:20:07 UTC
* Number of recursive dependencies: 81

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
ccache clang++ -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
ccache clang++ -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c read_mp3.cpp -o read_mp3.o
ccache clang++ -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torch/new/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c read_wav.cpp -o read_wav.o
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
ccache clang++ -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
ccache clang++ -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c read_mp3.cpp -o read_mp3.o
ccache clang++ -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/torch/revdep/library.noindex/torchaudio/Rcpp/include' -I/usr/local/include   -fPIC  -Wall -g -O2  -c read_wav.cpp -o read_wav.o
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
* Number of recursive dependencies: 69

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
