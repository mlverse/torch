# PLNmodels

<details>

* Version: 1.0.1
* GitHub: https://github.com/pln-team/PLNmodels
* Source code: https://github.com/cran/PLNmodels
* Date/Publication: 2023-02-12 14:42:07 UTC
* Number of recursive dependencies: 147

Run `revdepcheck::revdep_details(, "PLNmodels")` for more info

</details>

## In both

*   checking whether package ‘PLNmodels’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/PLNmodels/new/PLNmodels.Rcheck/00install.out’ for details.
    ```

## Installation

### Devel

```
* installing *source* package ‘PLNmodels’ ...
** package ‘PLNmodels’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c nlopt_wrapper.cpp -o nlopt_wrapper.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c optim_diag_cov.cpp -o optim_diag_cov.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c optim_fixed_cov.cpp -o optim_fixed_cov.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c optim_full_cov.cpp -o optim_full_cov.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c optim_genet_cov.cpp -o optim_genet_cov.o
...
1 warning generated.
1 warning generated.
clang++ -arch arm64 -std=gnu++14 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o PLNmodels.so RcppExports.o nlopt_wrapper.o optim_diag_cov.o optim_fixed_cov.o optim_full_cov.o optim_genet_cov.o optim_rank_cov.o optim_spherical.o packing.o -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRlapack -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRblas -L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1 -L/opt/R/arm64/gfortran/lib -lgfortran -lemutls_w -lquadmath -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1'
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib'
ld: library not found for -lgfortran
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [PLNmodels.so] Error 1
ERROR: compilation failed for package ‘PLNmodels’
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/PLNmodels/new/PLNmodels.Rcheck/PLNmodels’


```
### CRAN

```
* installing *source* package ‘PLNmodels’ ...
** package ‘PLNmodels’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c nlopt_wrapper.cpp -o nlopt_wrapper.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c optim_diag_cov.cpp -o optim_diag_cov.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c optim_fixed_cov.cpp -o optim_fixed_cov.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c optim_full_cov.cpp -o optim_full_cov.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/PLNmodels/nloptr/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c optim_genet_cov.cpp -o optim_genet_cov.o
...
1 warning generated.
1 warning generated.
clang++ -arch arm64 -std=gnu++14 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o PLNmodels.so RcppExports.o nlopt_wrapper.o optim_diag_cov.o optim_fixed_cov.o optim_full_cov.o optim_genet_cov.o optim_rank_cov.o optim_spherical.o packing.o -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRlapack -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRblas -L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1 -L/opt/R/arm64/gfortran/lib -lgfortran -lemutls_w -lquadmath -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1'
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib'
ld: library not found for -lgfortran
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [PLNmodels.so] Error 1
ERROR: compilation failed for package ‘PLNmodels’
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/PLNmodels/old/PLNmodels.Rcheck/PLNmodels’


```
# proteus

<details>

* Version: 1.1.0
* GitHub: NA
* Source code: https://github.com/cran/proteus
* Date/Publication: 2023-03-08 09:20:05 UTC
* Number of recursive dependencies: 123

Run `revdepcheck::revdep_details(, "proteus")` for more info

</details>

## In both

*   R CMD check timed out
    

# scDHA

<details>

* Version: 1.2.1
* GitHub: https://github.com/duct317/scDHA
* Source code: https://github.com/cran/scDHA
* Date/Publication: 2023-04-04 12:10:02 UTC
* Number of recursive dependencies: 62

Run `revdepcheck::revdep_details(, "scDHA")` for more info

</details>

## In both

*   checking whether package ‘scDHA’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/scDHA/new/scDHA.Rcheck/00install.out’ for details.
    ```

## Installation

### Devel

```
* installing *source* package ‘scDHA’ ...
** package ‘scDHA’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -DARMA_64BIT_WORD=1 -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppParallel/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppAnnoy/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -DARMA_64BIT_WORD=1 -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppParallel/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppAnnoy/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c Support.cpp -o Support.o
clang++ -arch arm64 -std=gnu++14 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o scDHA.so RcppExports.o Support.o -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRlapack -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRblas -L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1 -L/opt/R/arm64/gfortran/lib -lgfortran -lemutls_w -lquadmath -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1'
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib'
ld: library not found for -lgfortran
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [scDHA.so] Error 1
ERROR: compilation failed for package ‘scDHA’
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/scDHA/new/scDHA.Rcheck/scDHA’


```
### CRAN

```
* installing *source* package ‘scDHA’ ...
** package ‘scDHA’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -DARMA_64BIT_WORD=1 -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppParallel/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppAnnoy/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -DARMA_64BIT_WORD=1 -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppArmadillo/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppParallel/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/scDHA/RcppAnnoy/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c Support.cpp -o Support.o
clang++ -arch arm64 -std=gnu++14 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o scDHA.so RcppExports.o Support.o -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRlapack -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRblas -L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1 -L/opt/R/arm64/gfortran/lib -lgfortran -lemutls_w -lquadmath -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1'
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib'
ld: library not found for -lgfortran
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [scDHA.so] Error 1
ERROR: compilation failed for package ‘scDHA’
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/scDHA/old/scDHA.Rcheck/scDHA’


```
# sits

<details>

* Version: 1.3.0
* GitHub: https://github.com/e-sensing/sits
* Source code: https://github.com/cran/sits
* Date/Publication: 2023-03-17 18:10:02 UTC
* Number of recursive dependencies: 207

Run `revdepcheck::revdep_details(, "sits")` for more info

</details>

## In both

*   checking whether package ‘sits’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/sits/new/sits.Rcheck/00install.out’ for details.
    ```

## Installation

### Devel

```
* installing *source* package ‘sits’ ...
** package ‘sits’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c combine_data.cpp -o combine_data.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c kernel.cpp -o kernel.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c label_class.cpp -o label_class.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c linear_interp.cpp -o linear_interp.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c nnls_solver.cpp -o nnls_solver.o
...
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c smooth_whit.cpp -o smooth_whit.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c uncertainty.cpp -o uncertainty.o
clang++ -arch arm64 -std=gnu++14 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o sits.so RcppExports.o combine_data.o kernel.o label_class.o linear_interp.o nnls_solver.o normalize_data.o normalize_data_0.o sampling_window.o smooth.o smooth_sgp.o smooth_whit.o uncertainty.o -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRlapack -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRblas -L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1 -L/opt/R/arm64/gfortran/lib -lgfortran -lemutls_w -lquadmath -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1'
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib'
ld: library not found for -lgfortran
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [sits.so] Error 1
ERROR: compilation failed for package ‘sits’
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/sits/new/sits.Rcheck/sits’


```
### CRAN

```
* installing *source* package ‘sits’ ...
** package ‘sits’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c combine_data.cpp -o combine_data.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c kernel.cpp -o kernel.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c label_class.cpp -o label_class.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c linear_interp.cpp -o linear_interp.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c nnls_solver.cpp -o nnls_solver.o
...
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c smooth_whit.cpp -o smooth_whit.o
clang++ -arch arm64 -std=gnu++14 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/sits/RcppArmadillo/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c uncertainty.cpp -o uncertainty.o
clang++ -arch arm64 -std=gnu++14 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o sits.so RcppExports.o combine_data.o kernel.o label_class.o linear_interp.o nnls_solver.o normalize_data.o normalize_data_0.o sampling_window.o smooth.o smooth_sgp.o smooth_whit.o uncertainty.o -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRlapack -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRblas -L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1 -L/opt/R/arm64/gfortran/lib -lgfortran -lemutls_w -lquadmath -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1'
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib'
ld: library not found for -lgfortran
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [sits.so] Error 1
ERROR: compilation failed for package ‘sits’
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/sits/old/sits.Rcheck/sits’


```
# SPQR

<details>

* Version: 0.1.0
* GitHub: https://github.com/stevengxu/SPQR
* Source code: https://github.com/cran/SPQR
* Date/Publication: 2022-05-02 20:02:03 UTC
* Number of recursive dependencies: 65

Run `revdepcheck::revdep_details(, "SPQR")` for more info

</details>

## In both

*   checking whether package ‘SPQR’ can be installed ... ERROR
    ```
    Installation failed.
    See ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/SPQR/new/SPQR.Rcheck/00install.out’ for details.
    ```

## Installation

### Devel

```
* installing *source* package ‘SPQR’ ...
** package ‘SPQR’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/SPQR/RcppArmadillo/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/SPQR/RcppArmadillo/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2  -c mcmc_export.cpp -o mcmc_export.o
clang++ -arch arm64 -std=gnu++11 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o SPQR.so RcppExports.o mcmc_export.o -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRlapack -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRblas -L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1 -L/opt/R/arm64/gfortran/lib -lgfortran -lemutls_w -lquadmath -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1'
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib'
ld: library not found for -lgfortran
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [SPQR.so] Error 1
ERROR: compilation failed for package ‘SPQR’
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/SPQR/new/SPQR.Rcheck/SPQR’


```
### CRAN

```
* installing *source* package ‘SPQR’ ...
** package ‘SPQR’ successfully unpacked and MD5 sums checked
** using staged installation
** libs
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/SPQR/RcppArmadillo/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG  -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/SPQR/RcppArmadillo/include' -I/opt/R/arm64/include    -fPIC  -falign-functions=64 -Wall -g -O2  -c mcmc_export.cpp -o mcmc_export.o
clang++ -arch arm64 -std=gnu++11 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o SPQR.so RcppExports.o mcmc_export.o -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRlapack -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -lRblas -L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1 -L/opt/R/arm64/gfortran/lib -lgfortran -lemutls_w -lquadmath -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib/gcc/aarch64-apple-darwin20.6.0/12.0.1'
ld: warning: directory not found for option '-L/opt/R/arm64/gfortran/lib'
ld: library not found for -lgfortran
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [SPQR.so] Error 1
ERROR: compilation failed for package ‘SPQR’
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/SPQR/old/SPQR.Rcheck/SPQR’


```
# targets

<details>

* Version: 0.14.3
* GitHub: https://github.com/ropensci/targets
* Source code: https://github.com/cran/targets
* Date/Publication: 2023-03-08 13:40:02 UTC
* Number of recursive dependencies: 170

Run `revdepcheck::revdep_details(, "targets")` for more info

</details>

## Newly broken

*   R CMD check timed out
    

# torchdatasets

<details>

* Version: 0.3.0
* GitHub: https://github.com/mlverse/torchdatasets
* Source code: https://github.com/cran/torchdatasets
* Date/Publication: 2023-02-14 11:00:02 UTC
* Number of recursive dependencies: 70

Run `revdepcheck::revdep_details(, "torchdatasets")` for more info

</details>

## In both

*   R CMD check timed out
    

# torchvision

<details>

* Version: 0.5.0
* GitHub: https://github.com/mlverse/torchvision
* Source code: https://github.com/cran/torchvision
* Date/Publication: 2023-03-15 12:10:02 UTC
* Number of recursive dependencies: 43

Run `revdepcheck::revdep_details(, "torchvision")` for more info

</details>

## Newly broken

*   checking examples ...sh: line 1: 72641 Abort trap: 6           LANGUAGE=en _R_CHECK_INTERNALS2_=1 '/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/R' --vanilla > 'torchvision-Ex.Rout' 2>&1 < 'torchvision-Ex.R'
    ```
     ERROR
    Running examples in ‘torchvision-Ex.R’ failed
    The error most likely occurred in:
    
    > ### Name: draw_bounding_boxes
    > ### Title: Draws bounding boxes on image.
    > ### Aliases: draw_bounding_boxes
    > 
    > ### ** Examples
    > 
    > if (torch::torch_is_installed()) {
    + image <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
    + x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
    + y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
    + boxes <- torch::torch_cat(c(x, y, x + 20, y +  10), dim = 2)
    + bboxed <- draw_bounding_boxes(image, boxes, colors = "black", fill = TRUE)
    + tensor_image_browse(bboxed)
    + }
    Error : R: UnableToReadFont `helvetica' @ error/annotate.c/RenderFreetype/1396
    ```

*   checking dependencies in R code ... WARNING
    ```
    Missing or unexported object: ‘torch::torch_lstsq’
    ```

## Newly fixed

*   checking examples ...sh: line 1: 71953 Abort trap: 6           LANGUAGE=en _R_CHECK_INTERNALS2_=1 '/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/R' --vanilla > 'torchvision-Ex.Rout' 2>&1 < 'torchvision-Ex.R'
    ```
     ERROR
    Running examples in ‘torchvision-Ex.R’ failed
    The error most likely occurred in:
    
    > ### Name: draw_bounding_boxes
    > ### Title: Draws bounding boxes on image.
    > ### Aliases: draw_bounding_boxes
    > 
    > ### ** Examples
    > 
    ...
    > if (torch::torch_is_installed()) {
    + image <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
    + x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
    + y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
    + boxes <- torch::torch_cat(c(x, y, x + 20, y +  10), dim = 2)
    + bboxed <- draw_bounding_boxes(image, boxes, colors = "black", fill = TRUE)
    + tensor_image_browse(bboxed)
    + }
    Unable to revert mtime: /Library/Fonts
    Error : R: UnableToReadFont `helvetica' @ error/annotate.c/RenderFreetype/1396
    ```

## In both

*   R CMD check timed out
    

# torchvisionlib

<details>

* Version: 0.3.0
* GitHub: NA
* Source code: https://github.com/cran/torchvisionlib
* Date/Publication: 2022-10-31 22:10:02 UTC
* Number of recursive dependencies: 36

Run `revdepcheck::revdep_details(, "torchvisionlib")` for more info

</details>

## In both

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
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c exports.cpp -o exports.o
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c torchvisionlib.cpp -o torchvisionlib.o
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/new/torch/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c torchvisionlib_types.cpp -o torchvisionlib_types.o
clang++ -arch arm64 -std=gnu++11 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o torchvisionlib.so RcppExports.o exports.o torchvisionlib.o torchvisionlib_types.o -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: -undefined dynamic_lookup may not work with chained fixups
...
Warning in download.file(url = url, destfile = file) :
  cannot open URL 'https://github.com/mlverse/torchvisionlib/releases/download/v0.3.0/torchvisionlib-0.3.0+cpu+arch64-Darwin.zip': HTTP status was '404 Not Found'
Error: package or namespace load failed for ‘torchvisionlib’:
 .onLoad failed in loadNamespace() for 'torchvisionlib', details:
  call: download.file(url = url, destfile = file)
  error: cannot open URL 'https://github.com/mlverse/torchvisionlib/releases/download/v0.3.0/torchvisionlib-0.3.0+cpu+arch64-Darwin.zip'
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
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/torch/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c RcppExports.cpp -o RcppExports.o
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/torch/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c exports.cpp -o exports.o
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/torch/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c torchvisionlib.cpp -o torchvisionlib.o
clang++ -arch arm64 -std=gnu++11 -I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/include" -DNDEBUG -I../inst/include/ -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/Rcpp/include' -I'/Users/dfalbel/Documents/posit/torch/revdep/library.noindex/torch/old/torch/include' -I/opt/R/arm64/include   -fPIC  -falign-functions=64 -Wall -g -O2  -c torchvisionlib_types.cpp -o torchvisionlib_types.o
clang++ -arch arm64 -std=gnu++11 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib -L/opt/R/arm64/lib -o torchvisionlib.so RcppExports.o exports.o torchvisionlib.o torchvisionlib_types.o -F/Library/Frameworks/R.framework/Versions/4.2-arm64 -framework R -Wl,-framework -Wl,CoreFoundation
ld: warning: -undefined dynamic_lookup may not work with chained fixups
...
Warning in download.file(url = url, destfile = file) :
  cannot open URL 'https://github.com/mlverse/torchvisionlib/releases/download/v0.3.0/torchvisionlib-0.3.0+cpu+arch64-Darwin.zip': HTTP status was '404 Not Found'
Error: package or namespace load failed for ‘torchvisionlib’:
 .onLoad failed in loadNamespace() for 'torchvisionlib', details:
  call: download.file(url = url, destfile = file)
  error: cannot open URL 'https://github.com/mlverse/torchvisionlib/releases/download/v0.3.0/torchvisionlib-0.3.0+cpu+arch64-Darwin.zip'
Error: loading failed
Execution halted
ERROR: loading failed
* removing ‘/Users/dfalbel/Documents/posit/torch/revdep/checks.noindex/torchvisionlib/old/torchvisionlib.Rcheck/torchvisionlib’


```
