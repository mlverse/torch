---
name: torch-prepare-release
description: |
  Prepare a new CRAN release of torch
---

- Create a branch named cran/v*.*.*
- Bump the version in the description to *.*.*
- Update the `branch` variable in the R souce code to cran/v*.*.*
- Uncomment the commented lines in .RBuildignore
- Push the branch and wait for the lantern workflow to finish, so we have lantern binaries uploaded.
- Create a PR and wait for it to pass
- Polish NEWS bullets
