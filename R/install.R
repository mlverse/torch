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
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/macOS-cpu.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.5.0.zip",
          path = "libtorch/lib",
          filter = ".dll"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-cpu.zip", branch)
      ),
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcpu.zip",
          path = "libtorch/lib"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-cpu.zip", branch)
      )
    ),
    "10.1" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcu101.zip",
          path = "libtorch/lib"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-101.zip", branch)
      )
    ),
    "10.2" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcu102.zip",
          path = "libtorch/lib"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-102.zip", branch)
      )
    ),
    "9.2" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcu92.zip",
          path = "libtorch/lib"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu.zip", branch)
      )
    )
  )
)

#' @keywords internal
#' @export
install_path <- function(version = "1.5.0") {
  path <- Sys.getenv("TORCH_HOME")
  if (nchar(path) > 0) {
    if (!dir.exists(path)) {
      warning("The TORCH_HOME path does not exists.")
      path <- ""
    }
    else {
      install_info <- install_config[[version]][["cpu"]][[install_os()]]
      for (library_name in names(install_info)) {
        if (!lib_installed(library_name, path)) {
          warning("The TORCH_HOME path is missing the '", library_name, "' library.")
          path <- ""
        }
      }
    }
  }
  
  if (nchar(path) > 0) {
    path
  }
  else {
    normalizePath(file.path(system.file("", package = "torch"), "deps"), mustWork = FALSE)
  }
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
  
  utils::download.file(library_url, temp_file)
  
  uncompress <- if (identical(library_extension, "tgz")) utils::untar else utils::unzip
  
  uncompress(temp_file, exdir = temp_path)
  source_files <- dir(file.path(temp_path, source_path), full.names = T)
  
  if (!is.null(filter)) source_files <- Filter(filter, source_files)
  
  file.copy(source_files, install_path, recursive = TRUE)
}

install_os <- function() {
  tolower(Sys.info()[["sysname"]])
}

lantern_install_libs <- function(version, type, install_path) {
  current_os <- install_os()
  
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

#' @keywords internal
#' @export
install_type <- function(version) {
  if (nchar(Sys.getenv("CUDA")) > 0) return(Sys.getenv("CUDA"))
  if (install_os() != "linux") return("cpu")
  
  cuda_version <- NULL
  cuda_home <- Sys.getenv("CUDA_HOME")
  
  if (nchar(cuda_home) > 0) {
    versions_file <- file.path(cuda_home, "version.txt")
    if (file.exists(versions_file)) {
      cuda_version <- gsub("CUDA Version |\\.[0-9]+$", "", readLines(versions_file))
    }
  }
  
  if (is.null(cuda_version)) {
    versions_file <- "/usr/local/cuda/version.txt"
    if (file.exists(versions_file)) {
      cuda_version <- gsub("CUDA Version |\\.[0-9]+$", "", readLines(versions_file))
    }
  }
  
  if (is.null(cuda_version)) {
    smi <- tryCatch(system2("nvidia-smi", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
    if (!is.null(smi)) {
      cuda_version <- gsub(".*CUDA Version: | +\\|", "", smi[grepl("CUDA", smi)])
    }
  }
  
  if (is.null(cuda_version)) return("cpu")
  
  versions_available <- names(install_config[[version]])
  
  if (!cuda_version %in% versions_available) {
    message("Cuda ", cuda_version, " detected but torch only supports: ", paste(versions_available, collapse = ", "))
    return("cpu")
  }
  
  cuda_version
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
#' @details 
#' 
#' When using \code{path} to install in a specific location, make sure the \code{TORCH_HOME} environment
#' variable is set to this same path to reuse this installation. The \code{TORCH_INSTALL} environment
#' variable can be set to \code{0} to prevent auto-installing torch and \code{TORCH_LOAD} set to \code{0}
#' to avoid loading dependencies automatically. These environment variables are meant for advanced use
#' cases and troubleshootinng only.
#' 
#' @export
install_torch <- function(version = "1.5.0", type = install_type(version = version), reinstall = FALSE,
                          path = install_path(), ...) {
  if (reinstall) {
    unlink(path, recursive = TRUE)
  }
  
  if (!dir.exists(path)) {
    dir.create(path, showWarnings = FALSE)
  }
  
  lantern_install_libs(version, type, path)
  
  # reinitialize lantern, might happen if installation fails on load and manual install required
  if (!identical(list(...)$load, FALSE))
    lantern_start(reload = TRUE)
}
