branch <- "master"

install_config <- list(
  "1.5.0" = list(
    "cpu" = list(
      "darwin" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.0.zip",
          path = "libtorch/lib",
          filter = ".dylib"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/lantern-builds/refs/heads/%s/latest/macOS-cpu.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.5.0.zip",
          path = "libtorch/lib",
          filter = ".dll"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/lantern-builds/refs/heads/%s/latest/Windows-cpu.zip", branch)
      ),
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcpu.zip",
          path = "libtorch/lib"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/lantern-builds/refs/heads/%s/latest/Linux-cpu.zip", branch)
      )
    ),
    "10.1" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcu101.zip",
          path = "libtorch/lib"
        )
      )
    )
  )
)

#' @keywords internal
#' @export
install_path <- function() {
  normalizePath(file.path(system.file("", package = "torch"), "deps"), mustWork = FALSE)
}

install_exists <- function() {
  dir.exists(install_path())
}

lib_installed <- function(library_name, install_path) {
  x <- list.files(install_path)
  
  if (library_name == "liblantern")
    any(grepl("lantern", x))
  else if (library_name == "libtorch")
    any(grepl("torch", x))
}

lantern_install_lib <- function(library_name, library_url, install_path, source_path, filter) {
  library_extension <- paste0(".", tools::file_ext(library_url))
  temp_file <- tempfile(fileext = library_extension)
  temp_path <- tempfile()
  
  download.file(library_url, temp_file)
  
  uncompress <- if (identical(library_extension, "tgz")) untar else unzip
  
  uncompress(temp_file, exdir = temp_path)
  source_files <- dir(file.path(temp_path, source_path), full.names = T)
  
  if (!is.null(filter)) source_files <- Filter(filter, source_files)
  
  file.copy(source_files, install_path, recursive = TRUE)
}

lantern_install_libs <- function(version, type, install_path) {
  current_os <- tolower(Sys.info()[["sysname"]])
  
  if (!version %in% names(install_config))
    stop("Version ", version, " is not available, available versions: ",
         paste(names(install_config), collapse = ", "))
  
  if (!type %in% names(install_config[[version]]))
    stop("The ", type, " installation type is currently unsupported.")
  
  if (!current_os %in% names(install_config[[version]][[type]]))
    stop("The ", current_os, " operating system is currently unsupported.")
  
  install_info <- install_config[[version]][[type]][[current_os]]
  
  for (library_name in names(install_info)) {
    
    if (lib_installed(library_name, install_path)) {
      next
    }
    
    library_info <- install_info[[library_name]]
    
    if (!is.list(library_info)) library_info <- list(url = library_info, filter = "", path = "")
    if (is.null(library_info$filter)) library_info$filter <- ""
    
    lantern_install_lib(library_name = library_name,
                        library_url = library_info$url,
                        install_path = install_path,
                        source_path = library_info$path,
                        filter = function(e) grepl(library_info$filter, e))
  }
  
  invisible(install_path)
}

#' Install Torch
#' 
#' Installs Torch and its dependencies.
#' 
#' @param version The Torch version to install. 
#' @param type The installation type for Torch. Valid values are \code{"cpu"} or the 'CUDA' version.
#' @param reinstall Re-install Torch even if its already installed?
#' @param path Optional path to install or check for an already existing installation.
#' 
#' @export
install_torch <- function(version = "1.5.0", type = Sys.getenv("CUDA", unset = "cpu"), reinstall = FALSE,
                          path = install_path()) {
  if (reinstall) {
    unlink(path, recursive = TRUE)
  }
  
  if (!dir.exists(path)) {
    dir.create(path, showWarnings = FALSE)
  }
  
  lantern_install_libs(version, type, path)
}
