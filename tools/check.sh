cd torch

# install dependencies --------
apt-get install -y libcurl4-openssl-dev libssl-dev

Rscript -e "install.packages(c('remotes', 'rcmdcheck'))"
Rscript -e "remotes::install_deps(dependencies = TRUE)"

# build lantern and torch -----
Rscript tools/buildlantern.R

# check -----------
Rscript -e 'rcmdcheck::rcmdcheck(args = c("--no-manual", "--no-multiarch", "--no-build-vignettes"), build_args = c("--no-build-vignettes"), error_on = "error", check_dir = "check")'
