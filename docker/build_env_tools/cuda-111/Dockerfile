FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ARG ROOT_PASSWD
ARG PASSWD

# Set up R(Reference rocker/r-ver)
ARG R_VERSION_FULL=4.0.5
ARG R_VERSION=4
ENV CRAN=${CRAN:-https://cran.rstudio.com}
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8 
ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive   

RUN apt-get update && apt-get install -y --no-install-recommends bash-completion libbz2-dev \
    ca-certificates file fonts-texgyre g++ gfortran gsfonts libblas-dev libbz2-1.0 libcurl4 \
    libcurl4-openssl-dev libjpeg-dev liblzma-dev libopenblas-dev libpangocairo-1.0-0 libpcre2-dev \
    libpcre3 libpng16-16 libreadline-dev libtiff5 liblzma5 locales make unzip wget xorg-dev zip zlib1g libpq5 \
  && BUILDDEPS="curl default-jdk libbz2-dev libcairo2-dev libcurl4-openssl-dev libpango1.0-dev libjpeg-dev \
    libicu-dev libpcre3-dev libpng-dev libreadline-dev libtiff5-dev liblzma-dev libx11-dev libxt-dev perl \
    tcl8.6-dev tk8.6-dev texinfo texlive-extra-utils texlive-fonts-recommended texlive-fonts-extra \
    texlive-latex-recommended x11proto-core-dev xauth xfonts-base xvfb zlib1g-dev" \
  && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen en_US.utf8 && /usr/sbin/update-locale LANG=en_US.UTF-8 \
  && apt-get install -y --no-install-recommends $BUILDDEPS \
  && cd tmp/ \
## Download source code
  &&  wget https://cran.r-project.org/src/base/R-${R_VERSION}/R-${R_VERSION_FULL}.tar.gz \
## Extract source code
  && tar -xf R-${R_VERSION_FULL}.tar.gz \
  && cd R-${R_VERSION_FULL} \
## Set compiler flags
  && R_PAPERSIZE=letter \
    R_BATCHSAVE="--no-save --no-restore" R_BROWSER=xdg-open PAGER=/usr/bin/pager PERL=/usr/bin/perl R_UNZIPCMD=/usr/bin/unzip \
    R_ZIPCMD=/usr/bin/zip R_PRINTCMD=/usr/bin/lpr LIBnn=lib AWK=/usr/bin/awk \
    CFLAGS="-g -O2 -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -g" \
    CXXFLAGS="-g -O2 -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -g" \
## Configure options
    ./configure --enable-R-shlib --enable-memory-profiling --with-readline --with-blas --with-tcltk --disable-nls --with-recommended-packages \
## Build and install
  && make \
  && make install \
## Add a library directory (for user-installed packages)
  && mkdir -p /usr/local/lib/R/site-library \
  && chown root:staff /usr/local/lib/R/site-library \
  && chmod g+ws /usr/local/lib/R/site-library \
## Fix library path
  && sed -i '/^R_LIBS_USER=.*$/d' /usr/local/lib/R/etc/Renviron \
  && echo "R_LIBS_USER=\${R_LIBS_USER-'/usr/local/lib/R/site-library'}" >> /usr/local/lib/R/etc/Renviron \
  && echo "R_LIBS=\${R_LIBS-'/usr/local/lib/R/site-library:/usr/local/lib/R/library:/usr/lib/R/library'}" >> /usr/local/lib/R/etc/Renviron \
## Set configured CRAN mirror
  && if [ -z "$BUILD_DATE" ]; then MRAN=$CRAN; \
    else MRAN=https://mran.microsoft.com/snapshot/${BUILD_DATE}; fi \
  && echo MRAN=$MRAN >> /etc/environment \
  && echo "options(repos = c(CRAN='$MRAN'), download.file.method = 'libcurl')" >> /usr/local/lib/R/etc/Rprofile.site \
## Use littler installation scripts
  && Rscript -e "install.packages(c('littler', 'docopt'), repo = '$CRAN')" \
  && ln -s /usr/local/lib/R/site-library/littler/examples/install2.r /usr/local/bin/install2.r \
  && ln -s /usr/local/lib/R/site-library/littler/examples/installGithub.r /usr/local/bin/installGithub.r \
  && ln -s /usr/local/lib/R/site-library/littler/bin/r /usr/local/bin/r \
## Clean up from R source install
  && cd / \
  && rm -rf /tmp/* \
  && rm -rf /var/lib/apt/lists/*

# Set up RStudio(Reference rocker/rstudio)
ARG RSTUDIO_VERSION
ENV RSTUDIO_VERSION=${RSTUDIO_VERSION:-1.3.1093}
ARG S6_VERSION
ARG PANDOC_TEMPLATES_VERSION
ENV S6_VERSION=${S6_VERSION:-v2.2.0.3}
ENV S6_BEHAVIOUR_IF_STAGE2_FAILS=2
ENV PATH=/usr/lib/rstudio-server/bin:$PATH
ENV PANDOC_TEMPLATES_VERSION=${PANDOC_TEMPLATES_VERSION:-2.14.0.3}

## Download and install RStudio server & dependencies
## Attempts to get detect latest version, otherwise falls back to version given in $VER
## Symlink pandoc, pandoc-citeproc so they are available system-wide
RUN apt-get update && apt-get install -y --no-install-recommends git libapparmor1 libclang-dev libedit2 libssl-dev \
  lsb-release multiarch-support psmisc procps python-setuptools sudo \
  && if [ -z "$RSTUDIO_VERSION" ]; \
    then RSTUDIO_URL="https://www.rstudio.org/download/latest/stable/server/bionic/rstudio-server-latest-amd64.deb"; \
    else RSTUDIO_URL="http://download2.rstudio.org/server/bionic/amd64/rstudio-server-${RSTUDIO_VERSION}-amd64.deb"; fi \
  && wget -q $RSTUDIO_URL \
  && dpkg -i rstudio-server-*-amd64.deb \
  && rm rstudio-server-*-amd64.deb \
## Symlink pandoc & standard pandoc templates for use system-wide
  && ln -s /usr/lib/rstudio-server/bin/pandoc/pandoc /usr/local/bin \
  && ln -s /usr/lib/rstudio-server/bin/pandoc/pandoc-citeproc /usr/local/bin \
  && git clone --recursive --branch ${PANDOC_TEMPLATES_VERSION} https://github.com/jgm/pandoc-templates \
  && mkdir -p /opt/pandoc/templates \
  && cp -r pandoc-templates*/* /opt/pandoc/templates && rm -rf pandoc-templates* \
  && mkdir /root/.pandoc && ln -s /opt/pandoc/templates /root/.pandoc/templates \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/ \
## RStudio wants an /etc/R, will populate from $R_HOME/etc
  && mkdir -p /etc/R \
## Write config files in $R_HOME/etc
  && echo '\n\
    \n# Configure httr to perform out-of-band authentication if HTTR_LOCALHOST \
    \n# is not set since a redirect to localhost may not work depending upon \
    \n# where this Docker container is running. \
    \nif(is.na(Sys.getenv("HTTR_LOCALHOST", unset=NA))) { \
    \n  options(httr_oob_default = TRUE) \
    \n}' >> /usr/local/lib/R/etc/Rprofile.site \
  && echo "PATH=${PATH}" >> /usr/local/lib/R/etc/Renviron \
## Need to configure non-root user for RStudio
  && useradd rstudio \
  && echo "rstudio:rstudio" | chpasswd \
  && mkdir /home/rstudio \
  && chown rstudio:rstudio /home/rstudio \
  && addgroup rstudio staff \
## Prevent rstudio from deciding to use /usr/bin/R if a user apt-get installs a package
  &&  echo 'rsession-which-r=/usr/local/bin/R' >> /etc/rstudio/rserver.conf \
## use more robust file locking to avoid errors when using shared volumes:
  && echo 'lock-type=advisory' >> /etc/rstudio/file-locks \
## configure git not to request password each time
  && git config --system credential.helper 'cache --timeout=3600' \
  && git config --system push.default simple \
## Set up S6 init system
  && wget -P /tmp/ https://github.com/just-containers/s6-overlay/releases/download/${S6_VERSION}/s6-overlay-amd64.tar.gz \
  && tar xzf /tmp/s6-overlay-amd64.tar.gz -C / \
  && mkdir -p /etc/services.d/rstudio \
  && echo '#!/usr/bin/with-contenv bash \
          \n## load /etc/environment vars first: \
  		  \n for line in $( cat /etc/environment ) ; do export $line ; done \
          \n exec /usr/lib/rstudio-server/bin/rserver --server-daemonize 0' \
          > /etc/services.d/rstudio/run \
  && echo '#!/bin/bash \
          \n rstudio-server stop' \
          > /etc/services.d/rstudio/finish \
  && mkdir -p /home/rstudio/.rstudio/monitored/user-settings \
  && echo 'alwaysSaveHistory="0" \
          \nloadRData="0" \
          \nsaveAction="0"' \
          > /home/rstudio/.rstudio/monitored/user-settings/user-settings \
  && chown -R rstudio:rstudio /home/rstudio/.rstudio

RUN wget -P /etc/cont-init.d/ -O userconf https://github.com/rocker-org/rocker-versioned/blob/master/rstudio/userconf.sh

## running with "-e ADD=shiny" adds shiny server
RUN wget -P /etc/cont-init.d/ -O add https://github.com/rocker-org/rocker-versioned/blob/master/rstudio/add_shiny.sh
RUN wget -P /etc/rstudio/ https://github.com/rocker-org/rocker-versioned/blob/master/rstudio/disable_auth_rserver.conf
RUN wget -P /usr/lib/rstudio-server/bin/ -O pam-helper https://github.com/rocker-org/rocker-versioned/blob/master/rstudio/pam-helper.sh

EXPOSE 8787

## automatically link a shared volume for kitematic users
VOLUME /home/rstudio/kitematic

# Install some requires(Reference rocker/tidyverse)
RUN apt-get update -qq && apt-get -y --no-install-recommends install \
  libxml2-dev libcairo2-dev libsqlite-dev libmariadbd-dev libmariadbclient-dev \
  libpq-dev libssh2-1-dev unixodbc-dev libsasl2-dev \
  && Rscript -e "install.packages(c('tidyverse', 'dplyr', 'devtools', 'formatR', \
        'remotes', 'selectr', 'caTools', 'BiocManager'))"

# Set up localize(Reference tokyor/rstudio)
# Change environment to Japanese(Character and DateTime)
# If you would like to use Japanese version of this env
# ENV LANG ja_JP.UTF-8
# ENV LC_ALL ja_JP.UTF-8
# RUN sed -i '$d' /etc/locale.gen && echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen \
#   && locale-gen ja_JP.UTF-8 && /usr/sbin/update-locale LANG=ja_JP.UTF-8 LANGUAGE="ja_JP:ja"
# RUN /bin/bash -c "source /etc/default/locale"
# RUN ln -sf  /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# Install ipaexfont and some requires
RUN apt-get update && apt-get install -y fonts-ipaexfont vim curl

# Install packages
RUN Rscript -e "install.packages(c('githubinstall', 'ranger'))"

# Install torch for R
RUN Rscript -e "install.packages('torch')"
RUN Rscript -e "remotes::install_github('mlverse/torchvision')"
RUN echo 'root:'${ROOT_PASSWD} | chpasswd
RUN echo 'rstudio:'${PASSWD} | chpasswd
RUN echo "rstudio ALL=(ALL:ALL) ALL" >> /etc/sudoers
RUN gpasswd -a rstudio sudo

CMD ["/init"]
