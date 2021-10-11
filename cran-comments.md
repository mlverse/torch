Fixed valgrind errors. The vignettes shouldn't be evaluated in that context and
a bug in a recent knitr version cause R CMD CHECK to ignore the global handlers
that avoid the code to be evaluated. We have now pinned the knitr version to 
avoid that.


