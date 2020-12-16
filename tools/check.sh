cd torch

Rscript -e "install.packages(c('remotes', 'rcmdcheck'))"
Rscript -e "remotes::install_deps(dependencies = TRUE)"

# check -----------
Rscript -e 'rcmdcheck::rcmdcheck(args = c("--no-manual", "--no-multiarch"), error_on = "error", check_dir = "check")'

# install ---------
Rscript -e 'torch_package <- dir("check", full.names = TRUE, pattern = "torch_");install.packages(torch_package, repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")'

# run tests -------
Rscript -e 'setwd("tests"); source("testthat.R")'
