lantern_sync <- function(sync_lib = FALSE) {
  lib_dest <- "inst/"
  dir.create(lib_dest, showWarnings = FALSE, recursive = TRUE)

  if (!all(tools::md5sum(dir("lantern/include/lantern/", full.names = TRUE)) %in%
    tools::md5sum(dir("inst/include/lantern/", full.names = TRUE)))) {
    dir.create("inst/include/lantern/", showWarnings = FALSE, recursive = TRUE)
    files <- dir("lantern/include/lantern/", full.names = TRUE)
    file.copy(
      files, 
      file.path("inst/include/lantern/", basename(files)), 
      overwrite = TRUE
    )
  }

  if (sync_lib) {
    lib_src <- "lantern/build/liblantern"

    if (file.exists(paste0(lib_src, ".dylib"))) {
      path <- paste0(lib_src, ".dylib")
    } else if (file.exists(paste0(lib_src, ".so"))) {
      path <- paste0(lib_src, ".so")
    } else if (file.exists("lantern/build/Release/lantern.dll")) {
      path <- list.files("lantern/build/Release/", full.names = TRUE)
    }

    dir.create("inst/lib", showWarnings = FALSE, recursive = TRUE)

    file.copy(
      path,
      file.path(lib_dest, "lib"),
      overwrite = TRUE
    )
  }
}
