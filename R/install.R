branch <- "libtorch-v1.8"  


install_config <- list(
  "1.8.0" = list(
    "cpu" = list(
      "darwin" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.0.zip",
          path = "libtorch/lib",
          filter = ".dylib",
          md5hash = "23f07569e0c942260c8b13fa8c3289b8"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/macOS-cpu.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.8.0%2Bcpu.zip",
          path = "libtorch/lib",
          filter = ".dll",
          md5hash = "f1ef43cbda67461f357a1463ac176a97"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-cpu.zip", branch)
      ),
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.0%2Bcpu.zip",
          path = "libtorch/lib",
          md5hash = "82969e58ae8cabee16746711850d6fd9"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-cpu.zip", branch)
      )
    ),
    "10.1" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.8.0%2Bcu101.zip",
          path = "libtorch/lib",
          md5hash = "ca8957f8cda1c82164192037b9411714"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-101.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.8.0%2Bcu101.zip",
          path = "libtorch/lib",
          filter = ".dll",
          md5hash = "8351770ef9b2909d63353157c5d723c9"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-gpu-101.zip", branch)
      )
    ),
    "10.2" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.0.zip",
          path = "libtorch/lib",
          md5hash = "041594059e03381d8ef4d9f63c8e2c47"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-102.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.8.0.zip",
          path = "libtorch/lib",
          filter = ".dll",
          md5hash = "fa3d3c05f9fef22e7b67bcf16cc6729e"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-gpu-102.zip", branch)
      )
    ),
    "11.1" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip",
          path = "libtorch/lib",
          md5hash = "f6ab838b62fba8f875ccbceeeb71c6cd"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-111.zip", branch)
      )
    )
  )
)

#' @keywords internal
install_path <- function(version = "1.8.0") {
  path <- Sys.getenv("TORCH_HOME")
  if (nzchar(path)) {
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
  
  if (nzchar(path)) {
    path
  }
  else {
    normalizePath(file.path(system.file("", package = "torch"), "deps"), mustWork = FALSE)
  }
}

install_exists <- function() {
  dir.exists(install_path())
}

#' Verifies if torch is installed
#'
#' @export
torch_is_installed <- function() {
  install_exists()
}

lib_installed <- function(library_name, install_path) {
  x <- list.files(install_path)
  
  if (library_name == "liblantern")
    any(grepl("lantern", x))
  else if (library_name == "libtorch")
    any(grepl("torch", x))
}

lantern_install_lib <- function(library_name, library_url, 
                                install_path, source_path, filter, md5hash) {
  library_extension <- paste0(".", tools::file_ext(library_url))
  temp_file <- tempfile(fileext = library_extension)
  temp_path <- tempfile()
  
  utils::download.file(library_url, temp_file)
  on.exit(try(unlink(temp_file)))

  if (!is.null(md5hash) && is.character(md5hash) && length(md5hash) == 1) {
    hash <- tools::md5sum(temp_file)
    if (hash != md5hash) {
      stop(
        "The file downloaded from '", library_url,
        "' does not match the expected md5 hash '",
        md5hash, "'. The observed hash is '", hash,
        "'. Due to security reasons the installation is stopped."
      )
    }
  }
  
  uncompress <- if (identical(library_extension, "tgz")) utils::untar else utils::unzip
  
  uncompress(temp_file, exdir = temp_path)
  source_files <- dir(file.path(temp_path, source_path), full.names = T)
  
  if (!is.null(filter)) source_files <- Filter(filter, source_files)
  
  file.copy(source_files, install_path, recursive = TRUE)
}

install_os <- function() {
  tolower(Sys.info()[["sysname"]])
}

lantern_install_libs <- function(version, type, install_path, install_config) {
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
    
    lantern_install_lib(
      library_name = library_name,
      library_url = library_info$url,
      install_path = install_path,
      source_path = library_info$path,
      filter = function(e) grepl(library_info$filter, e),
      md5hash = library_info$md5hash
    )
  }
  
  invisible(install_path)
}

install_type_windows <- function(version) {
  
  cuda_version <- NULL
  cuda_path <- Sys.getenv("CUDA_PATH")
  
  if (nzchar(cuda_path)) {
    versions_file <- file.path(cuda_path, "version.txt")
    if (file.exists(versions_file)) {
      cuda_version <- gsub("CUDA Version |\\.[0-9]+$", "", readLines(versions_file))
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

#' @keywords internal
install_type <- function(version) {
  if (nzchar(Sys.getenv("CUDA"))) return(Sys.getenv("CUDA"))
  if (install_os() == "windows") return(install_type_windows(version))
  
  if (install_os() != "linux") return("cpu") # macOS
  
  # Detect cuda version on Linux
  
  cuda_version <- NULL
  cuda_home <- Sys.getenv("CUDA_HOME")
  
  if (nzchar(cuda_home)) {
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
    nvcc <- tryCatch(system2("nvcc", "--version", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
    if (!is.null(nvcc)) {
      cuda_version <- gsub(".*release |, V.*", "", nvcc[grepl("release", nvcc)])
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
#' @param timeout Optional timeout in seconds for large file download.
#' @param ... other optional arguments (like \code{`load`} for manual installation).
#' 
#' @details 
#' 
#' When using \code{path} to install in a specific location, make sure the \code{TORCH_HOME} environment
#' variable is set to this same path to reuse this installation. The \code{TORCH_INSTALL} environment
#' variable can be set to \code{0} to prevent auto-installing torch and \code{TORCH_LOAD} set to \code{0}
#' to avoid loading dependencies automatically. These environment variables are meant for advanced use
#' cases and troubleshooting only.
#' When timeout error occurs during library archive download, or length of downloaded files differ from 
#' reported length, an increase of the \code{timeout} value should help.
#' 
#' @export
install_torch <- function(version = "1.8.0", type = install_type(version = version), reinstall = FALSE,
                          path = install_path(), timeout = 360, ...) {
  
  if (reinstall) {
    unlink(path, recursive = TRUE)
  }
  
  if (!dir.exists(path)) {
    dir.create(path, showWarnings = FALSE)
  }
  if (!is.null(list(...)$install_config) && is.list(list(...)$install_config))
    install_config <- list(...)$install_config
  
  withr::with_options(list(timeout = timeout),
                      lantern_install_libs(version, type, path, install_config))
    
  # reinitialize lantern, might happen if installation fails on load and manual install is required
  if (!identical(list(...)$load, FALSE))
    lantern_start(reload = TRUE)
  
}
                     
#' Install Torch from files
#' 
#' Installs Torch and its dependencies from files.
#' 
#' @param version The Torch version to install. 
#' @param type The installation type for Torch. Valid values are \code{"cpu"} or the 'CUDA' version.
#' @param libtorch The installation archive file to use for Torch. Shall be a \code{"file://"} URL scheme.
#' @param liblantern The installation archive file to use for Lantern. Shall be a \code{"file://"} URL scheme.
#' @param ... other parameters to be passed to \code{"install_torch()"} 
#' 
#' @details 
#' 
#' When \code{"install_torch()"} initiated download is not possible, but installation archive files are
#' present on local filesystem, \code{"install_torch_from_file()"} can be used as a workaround to installation issue.
#' \code{"libtorch"} is the archive containing all torch modules, and \code{"liblantern"} is the C interface to libtorch
#' that is used for the R package. Both are highly dependent, and should be checked through \code{"get_install_libs_url()"}
#' 
#' 
#' @export
install_torch_from_file <- function(version = "1.7.1", type = install_type(version = version), libtorch, liblantern, ...) {
  stopifnot(inherits(url(libtorch), "file"))
  stopifnot(inherits(url(liblantern), "file"))

  install_config[[version]][[type]][[install_os()]][["libtorch"]][["url"]] <- libtorch
  install_config[[version]][[type]][[install_os()]][["liblantern"]] <- liblantern

  install_torch(version = version, type = type, install_config = install_config, ...)
}
                     
#' List of files to download
#' 
#' List the Torch and Lantern files to download as local files in order to proceed with install_torch_from_file().
#' 
#' @param version The Torch version to install. 
#' @param type The installation type for Torch. Valid values are \code{"cpu"} or the 'CUDA' version.
#' 
#' 
#' @export
get_install_libs_url <- function(version = "1.7.1", type = install_type(version = version)) {

  libtorch <- install_config[[version]][[type]][[install_os()]][["libtorch"]][["url"]]
  liblantern <- install_config[[version]][[type]][[install_os()]][["liblantern"]]
  list(libtorch = libtorch, liblantern = liblantern)
}

