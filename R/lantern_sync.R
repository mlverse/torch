lantern_sync <- function(sync_lib = FALSE) {
  if (!dir.exists("src/lantern")) dir.create("src/lantern")
  
  if (!all(tools::md5sum(dir("lantern/include/lantern/", full.names = TRUE)) %in% 
          tools::md5sum(dir("src/lantern/", full.names = TRUE)))) {
    file.copy(dir("lantern/include/lantern/", full.names = TRUE), "src/lantern/", overwrite = TRUE)  
  }
  
  if (sync_lib) {
    lib_dest <- file.path(system.file("", package = "torch"), "deps")
    lib_dest <- "deps/"
    suppressWarnings(dir.create(lib_dest))
    lib_src <- "lantern/build/liblantern"
    
    if (file.exists(paste0(lib_src, ".dylib")))
      path <- paste0(lib_src, ".dylib")
    else if (file.exists(paste0(lib_src, ".so")))
      path <- paste0(lib_src, ".so")
    else if (file.exists("lantern/build/Release/lantern.dll"))
      path <- "lantern/build/Release/lantern.dll"
    
    file.copy(
      path,
      lib_dest,
      overwrite = TRUE
    )
  }
}
