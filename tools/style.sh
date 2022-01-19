#!/bin/bash

if ! command -v clang-format &> /dev/null
then
  echo "Error: clang-format could not be found!"
  exit 1
fi

# Style/format R code
Rscript -e "if (!require('styler')) install.packages('styler')"
Rscript -e 'styler::style_pkg(exclude_files = list.files("./R", pattern = "^gen-*.*|^RcppExports.*", full.names = TRUE))'


# Style/format C/C++ code
find . -type f -regex '.*\.\(h\|hpp\|c\|cc\|cpp\|cxx\)' \
  -not -path "./build/*" \
  -not -path "./check/*" \
  -not -path "./inst/include/*/*" \
  -not -path "./lantern/build/*" \
  -not -path "./lantern/headers/build/*" \
  -not -path "./revdep/*" \
  -not -path "*/gen-*.*" \
  -not -path "*/lantern.*" \
  -not -path "*/RcppExports.*" \
  -exec clang-format -style=Google --verbose -i {} \;
git diff --stat

# Remove whitespaces
-find . -type f \( -name 'DESCRIPTION' -o -name "*.R" \) \
  -not -path "./build/*" \
  -not -path "./check/*" \
  -not -path "./inst/include/*/*" \
  -not -path "./lantern/build/*" \
  -not -path "./lantern/headers/build/*" \
  -not -path "./revdep/*" \
  -not -path "*/gen-*.*" \
  -not -path "*/lantern.*" \
  -not -path "*/RcppExports.*" \
  -printf '%p\n' -exec perl -pi -e 's/[ \t]*$//' {} \;
find . -type f -regex '.*\.\(h\|hpp\|c\|cc\|cpp\|cxx\)' \
  -not -path "./build/*" \
  -not -path "./check/*" \
  -not -path "./inst/include/*/*" \
  -not -path "./lantern/build/*" \
  -not -path "./lantern/headers/build/*" \
  -not -path "./revdep/*" \
  -not -path "*/gen-*.*" \
  -not -path "*/lantern.*" \
  -not -path "*/RcppExports.*" \
  -printf '%p\n' -exec perl -pi -e 's/[ \t]*$//' {} \;
git diff --stat

# Render documents
Rscript -e "if (!require('rmarkdown')) install.packages('rmarkdown')"
Rscript -e "if (!require('roxygen2')) install.packages('roxygen2')"
Rscript tools/document.R
if [ -f README.Rmd ]; then
  Rscript -e 'rmarkdown::render("README.Rmd", output_format = "github_document")'
fi
git diff --stat
