branch <- "cuda11.2"  


install_config <- list(
  "1.7.1" = list(
    "cpu" = list(
      "darwin" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip",
          path = "libtorch/lib",
          filter = ".dylib",
          md5hash = "d1d0561fa91326afcb7f0abe0241e704"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/macOS-cpu.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.7.1%2Bcpu.zip",
          path = "libtorch/lib",
          filter = ".dll",
          md5hash = "6618d6cf35fa07224ddcef5de6d9c43c"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-cpu.zip", branch)
      ),
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcpu.zip",
          path = "libtorch/lib",
          md5hash = "185c956c1b09afba574b83d6a7fba634"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-cpu.zip", branch)
      )
    ),
    "10.1" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcu101.zip",
          path = "libtorch/lib",
          md5hash = "fc8dd4ba0cef95ae272de6a7b3600d59"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-101.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.7.1%2Bcu101.zip",
          path = "libtorch/lib",
          filter = ".dll",
          md5hash = "602a980d4bbfdf2e03742df75050405b"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-gpu-101.zip", branch)
      )
    ),
    "10.2" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.7.1.zip",
          path = "libtorch/lib",
          md5hash = "6ef47fd76f1b5959bf807b06f0803aa9"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-102.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.7.1.zip",
          path = "libtorch/lib",
          filter = ".dll",
          md5hash = "2ffa6d627007574ce231d93964c965fa"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-gpu-102.zip", branch)
      )
    ),
    "9.2" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcu92.zip",
          path = "libtorch/lib",
          md5hash = "bceb125610b9905e777089c2528000a4"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-902.zip", branch)
      )
    ),
    "11.0" = list(
      "linux" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cu110/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcu110.zip",
          path = "libtorch/lib",
          md5hash = "75d171cd4c84fa33caa46651a74a2fe1"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-110.zip", branch)
      )
    )
  )
)

#' @keywords internal
install_path <- function(version = "1.7.1") {
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
#' @param ... other optional arguments (like `load` for manual installation.)
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
install_torch <- function(version = "1.7.1", type = install_type(version = version), reinstall = FALSE,
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
