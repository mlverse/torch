install_config <- list(
  "1.3.0" = list(
    "cpu" = list(
      "darwin" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.3.0.zip",
          path = "libtorch/lib",
          filter = ".dylib"
        ),
        "liblantern" = "https://github.com/mlverse/lantern/releases/download/v0.0.6/macOS.zip"
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.3.0.zip",
          path = "libtorch/lib",
          filter = ".dll"
        ),
        "liblantern" = "https://github.com/mlverse/lantern/releases/download/v0.0.6/windows.zip"
      ),
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.3.0%2Bcpu.zip",
          path = "libtorch/lib"
        ),
        "liblantern" = "https://github.com/mlverse/lantern/releases/download/v0.0.6/linux.zip"
      )
    )
  )
)

lantern_install_path <- function() {
  normalizePath(file.path(system.file("", package = "lantern"), "deps"), mustWork = FALSE)
}

lantern_installed <- function() {
  dir.exists(lantern_install_path())
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

#' @export
lantern_install <- function(version = "1.3.0", type = "cpu", reinstall = FALSE) {
  if (reinstall) {
    unlink(lantern_install_path(), recursive = TRUE)
  }
  
  if (lantern_installed()) {
    stop("Lantern is already installed.")
  }
  
  install_path <- lantern_install_path()
  dir.create(install_path)
  
  lantern_install_libs(version, type, install_path)
}
