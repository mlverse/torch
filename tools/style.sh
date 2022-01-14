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
find -not -path "./check/*" . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) ! -path "*/gen-*.*" ! -path "*/lantern.*"  ! -path "*/RcppExports.*" -exec clang-format -style=Google --verbose -i {} \;

# Remove whitespaces
find -not -path "./check/*" . -type f \( -name 'DESCRIPTION' -o -name "*.R" \) ! -path "*/gen-*.*" ! -path "*/RcppExports.*" -exec perl -pi -e 's/[ \t]*$//' {} \;
find -not -path "./check/*" . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) ! -path "*/gen-*.*" ! -path "*/lantern.*"  ! -path "*/RcppExports.*" -exec perl -pi -e 's/[ \t]*$//' {} \;
