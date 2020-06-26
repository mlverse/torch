cd torch

Rscript -e "install.packages(c('remotes', 'rcmdcheck'))"
Rscript -e "remotes::install_deps(dependencies = TRUE)"

# check -----------
Rscript -e 'rcmdcheck::rcmdcheck(args = c("--no-manual", "--no-multiarch", "--no-build-vignettes"), build_args = c("--no-build-vignettes"), error_on = "error", check_dir = "check")'
