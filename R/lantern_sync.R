lantern_sync <- function(sync_lib = FALSE) {
  if (!dir.exists("src/lantern")) dir.create("src/lantern")
  file.copy(dir("../lantern/include/lantern/", full.names = TRUE), "src/lantern/", overwrite = TRUE)
  
  if (sync_lib) {
    lib_dest <- file.path(system.file("", package = "torch"), "deps")
    file.copy(normalizePath("../lantern/build/liblantern.dylib"), lib_dest, overwrite = TRUE)
  }
}
