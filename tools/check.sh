cd torch

export TORCH_TEST=1
export TORCH_INSTALL=1

Rscript -e "install.packages(c('remotes', 'rcmdcheck', 'markdown'))"
Rscript -e "remotes::install_deps(dependencies = TRUE)"

if "$BUILD_LANTERN" = "true"; then
  Rscript tools/buildlantern.R
fi

# check -----------
Rscript -e 'check <- rcmdcheck::rcmdcheck(args = c("--no-manual", "--no-multiarch"), error_on = "never")' \
        -e 'quit(status = check$status)'

exit $?
