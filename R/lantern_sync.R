lantern_sync <- function(sync_lib = FALSE) {
  if (!dir.exists("src/lantern")) dir.create("src/lantern")
  file.copy(dir("lantern/include/lantern/", full.names = TRUE), "src/lantern/", overwrite = TRUE)
  
  if (sync_lib) {
    lib_dest <- file.path(system.file("", package = "torch"), "deps")
    lib_dest <- "deps/"
    suppressWarnings(dir.create(lib_dest))
    lib_src <- "lantern/build/liblantern"
    file.copy(normalizePath(
      if (file.exists(paste0(lib_src, ".dylib")))
        paste0(lib_src, ".dylib")
        else
          paste0(lib_src, ".so")),
      lib_dest,
      overwrite = TRUE)
  }
}
