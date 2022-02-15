# torchaudio

<details>

* Version: 0.2.0
* GitHub: NA
* Source code: https://github.com/cran/torchaudio
* Date/Publication: 2021-05-05 15:20:07 UTC
* Number of recursive dependencies: 79

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

