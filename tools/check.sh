cd torch

export TORCH_TEST=1
export TORCH_INSTALL=1

Rscript -e "install.packages(c('remotes', 'rcmdcheck'))"
Rscript -e "remotes::install_deps(dependencies = TRUE)"

# check -----------
Rscript -e 'rcmdcheck::rcmdcheck(args = c("--no-manual", "--no-multiarch"), error_on = "error", check_dir = "check")'

echo "Done"