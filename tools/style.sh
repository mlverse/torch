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
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) \
  ! -path "*/gen-*.*" ! -path "*/lantern.*"  ! -path "*/RcppExports.*" ! -path "./check/*" \
  -not -path './lantern/build/*' \
  -not -path './lantern/headers/build/*' \
  -not -path "./inst/include/*/*" \
  -not -path "./revdep/*" \
  -exec clang-format -style=Google --verbose -i {} \;
git diff --stat

# Remove whitespaces
find . -type f \( -name 'DESCRIPTION' -o -name "*.R" \) ! -path "*/gen-*.*" ! -path "*/RcppExports.*" ! -path "./check/*" -printf '%p\n' -exec perl -pi -e 's/[ \t]*$//' {} \;
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) ! -path "*/gen-*.*" ! -path "*/lantern.*"  ! -path "*/RcppExports.*" ! -path "./check/*" -printf '%p\n' -exec perl -pi -e 's/[ \t]*$//' {} \;
git diff --stat

# Render documents
Rscript -e "if (!require('rmarkdown')) install.packages('rmarkdown')"
Rscript -e "if (!require('roxygen2')) install.packages('roxygen2')"
Rscript tools/document.R
if [ -f README.Rmd ]; then
  Rscript -e 'rmarkdown::render("README.Rmd", output_format = "md_document")'
fi
git diff --stat
