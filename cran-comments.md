
We found a bug in the `Makevars` that could explain the errors with parallel
make that motivated removing this package from CRAN.
This release hopefully fixes the issue, however we are not able to reproduce
this outside of CRAN so we can further debug.

## Test environments

* local R installation, R 4.0.2
* local mac OS install, R 4.0.0
* ubuntu 16.04 (on github actions), R 4.0.0
* mac OS 10.15.4 (on github actions) R 4.0.0
* Microsoft Windows Server 2019 10.0.17763 (on github actions) R 4.0.0
* win-builder (devel)

## R CMD check results

0 errors | 0 warnings | 1 note

installed size is 23.1Mb
sub-directories of 1Mb or more:
    R      3.1Mb
    help   2.6Mb
    libs  17.2Mb

