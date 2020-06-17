cd torch

# install dependencies --------
Rscript -e "install.packages(c('remotes'))" -e "remotes::install_github('r-hub/sysreqs')"
sysreqs=$(Rscript -e "cat(sysreqs::sysreq_commands('DESCRIPTION'))")
-s eval "$sysreqs"

Rscript -e "install.packages(c('remotes', 'rcmdcheck'))" -e "remotes::install_deps(dependencies = TRUE)"

# build lantern and torch -----
Rscript tools/buildlantern.R

# check -----------
Rscript -e 'rcmdcheck::rcmdcheck(args = c("--no-manual", "--no-multiarch", "--no-build-vignettes"), build_args = c("--no-build-vignettes"), error_on = "error", check_dir = "check")'

find . -name testthat.Rout -exec cat '{}' ';'