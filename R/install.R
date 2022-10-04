branch <- "m1-mac"

install_config <- list(
  "1.12.0" = list(
    "cpu" = list(
      "darwin" = list(
        x86_64 = list(
          "libtorch" = list(
            url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.0.zip",
            path = "libtorch/",
            filter = ".dylib",
            md5hash = "b01b0c32221fc81ea1c480eab589cac6"
          ),
          "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/macOS-cpu.zip", branch)  
        ),
        aarch64 = list(
          libtorch = list(
            url = "https://github.com/mlverse/libtorch-mac-m1/releases/download/LibTorch/libtorch-v1.12.0.zip",
            path = "libtorch/",
            filter = ".dylib",
            md5hash = "bddd13cfaabaf43a66b760df7229c238"
          ),
          "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/macOSArm64-cpu.zip", branch)  
        )
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.12.0%2Bcpu.zip",
          path = "libtorch/",
          filter = ".dll",
          md5hash = "5b61a53594bc3cb017002edc5a2f0131"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-cpu.zip", branch)
      ),
      "linux" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcpu.zip",
          md5hash = "a0400de625916051176b69f7aa524fc3"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-cpu.zip", branch)
      )
    ),
    "10.2" = list(
      "linux" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu102.zip",
          md5hash = "e424e58a49ed153f7670d9acb533c3fc"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-102.zip", branch)
      )
    ),
    "11.3" = list(
      "linux" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu113.zip",
          md5hash = "7fbb79c163c0482181b180e10c66ce4d"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-113.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-1.12.0%2Bcu113.zip",
          filter = ".dll",
          md5hash = "fd5c98d323e00e32ecbad831e722cae7"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-gpu-113.zip", branch)
      )
    )
  )
)

install_path <- function(version = "1.12.0") {
  path <- Sys.getenv("TORCH_HOME")
  if (nzchar(path)) {
    normalizePath(path, mustWork = FALSE)
  } else {
    normalizePath(file.path(system.file("", package = "torch")), mustWork = FALSE)
  }
}

#' A simple exported version of install_path
#' Returns the torch installation path.
#' @export
torch_install_path <- function() {
  install_path()
}

install_exists <- function() {
  if (!dir.exists(install_path())) {
    return(FALSE)
  }

  if (!length(list.files(file.path(install_path(), "lib"), "torch")) > 0) {
    return(FALSE)
  }

  if (!length(list.files(file.path(install_path(), "lib"), "lantern")) > 0) {
    return(FALSE)
  }

  TRUE
}

#' Verifies if torch is installed
#'
#' @export
torch_is_installed <- function() {
  install_exists()
}

lib_installed <- function(library_name, install_path) {
  x <- list.files(file.path(install_path, "lib"))

  if (library_name == "liblantern") {
    any(grepl("lantern", x))
  } else if (library_name == "libtorch") {
    any(grepl("torch", x))
  }
}

lantern_install_lib <- function(library_name, library_url,
                                install_path, source_path, filter, md5hash,
                                inst_path) {
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

  file.copy(
    from = dir(file.path(temp_path, source_path), full.names = TRUE),
    to = file.path(install_path, inst_path),
    recursive = TRUE
  )

  # if (!is.null(filter)) source_files <- Filter(filter, source_files)
  # file.copy(source_files, install_path, recursive = TRUE)
  #
  #
  # header_files <- dir(file.path(temp_path, "libtorch", "include"), full.names = T)
  # if (length(header_files) > 0) {
  #   # also install the include files.
  #   include_dir <- file.path(system.file("", package = "torch"), "include/")
  #   dir.create(include_dir, showWarnings = FALSE, recursive = TRUE)
  #   file.copy(header_files, include_dir, recursive = TRUE)
  # }
}

install_os <- function() {
  tolower(Sys.info()[["sysname"]])
}

lantern_install_libs <- function(version, type, install_path, install_config) {
  current_os <- install_os()

  if (!version %in% names(install_config)) {
    stop(
      "Version ", version, " is not available, available versions: ",
      paste(names(install_config), collapse = ", ")
    )
  }

  if (!type %in% names(install_config[[version]])) {
    stop("The ", type, " installation type is currently unsupported.")
  }

  if (!current_os %in% names(install_config[[version]][[type]])) {
    stop("The ", current_os, " operating system is currently unsupported.")
  }

  install_info <- install_config[[version]][[type]][[current_os]]
  
  if (current_os == "darwin") {
    install_info <- install_info[[R.version$arch]]
  }

  for (library_name in names(install_info)) {
    if (lib_installed(library_name, install_path)) {
      next
    }

    library_info <- install_info[[library_name]]

    if (!is.list(library_info)) {
      library_info <- list(url = library_info, filter = "", path = "", inst_path = "lib")
    }
    if (is.null(library_info$filter)) library_info$filter <- ""
    if (is.null(library_info$inst_path)) library_info$inst_path <- ""
    
    library_info <- maybe_switch_precxx11abi(library_info)

    lantern_install_lib(
      library_name = library_name,
      library_url = library_info$url,
      install_path = install_path,
      source_path = library_info$path,
      filter = function(e) grepl(library_info$filter, e),
      md5hash = library_info$md5hash,
      inst_path = library_info$inst_path
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

  # Query nvcc from cuda in cuda_path.
  if (nzchar(cuda_path) && is.null(cuda_version)) {
    nvcc_path <- file.path(cuda_path, "bin", "nvcc.exe")
    cuda_version <- nvcc_version_from_path(nvcc_path)
  }

  if (is.null(cuda_version)) {
    return("cpu")
  }

  versions_available <- names(install_config[[version]])

  if (!cuda_version %in% versions_available) {
    message("Cuda ", cuda_version, " detected but torch only supports: ", paste(versions_available, collapse = ", "))
    return("cpu")
  }

  cuda_version
}

nvcc_version_from_path <- function(nvcc) {
  suppressWarnings(
    nvcc <- tryCatch(system2(nvcc, "--version", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
  )

  if (is.null(nvcc) || !any(grepl("release", nvcc))) {
    return(NULL)
  }

  gsub(".*release |, V.*", "", nvcc[grepl("release", nvcc)])
}

#' @keywords internal
install_type <- function(version) {
  if (nzchar(Sys.getenv("CUDA"))) {
    return(Sys.getenv("CUDA"))
  }
  if (install_os() == "windows") {
    return(install_type_windows(version))
  }

  if (install_os() != "linux") {
    return("cpu")
  } # macOS

  # Detect cuda version on Linux

  cuda_version <- NULL
  cuda_home <- Sys.getenv("CUDA_HOME")

  # This file no longer exists with cuda >= 11
  if (nzchar(cuda_home)) {
    versions_file <- file.path(cuda_home, "version.txt")
    if (file.exists(versions_file)) {
      cuda_version <- gsub("CUDA Version |\\.[0-9]+$", "", readLines(versions_file))
    }
  }

  # Query nvcc from cuda in cuda_home path.
  if (nzchar(cuda_home) && is.null(cuda_version)) {
    nvcc_path <- file.path(cuda_home, "bin", "nvcc")
    cuda_version <- nvcc_version_from_path(nvcc_path)
  }

  # Try to find in conventional location.
  if (is.null(cuda_version)) {
    versions_file <- "/usr/local/cuda/version.txt"
    if (file.exists(versions_file)) {
      cuda_version <- gsub("CUDA Version |\\.[0-9]+$", "", readLines(versions_file))
    }
  }

  # Query nvcc from conventional location
  if (is.null(cuda_version)) {
    cuda_version <- nvcc_version_from_path("/usr/local/cuda/bin/nvcc")
  }

  if (is.null(cuda_version)) {
    cuda_version <- nvcc_version_from_path("nvcc")
  }

  if (is.null(cuda_version)) {
    return("cpu")
  }

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
#' Setting the environmanet variable `PRECXX11ABI=1` will trigger the installation of
#' a Pre-cxx11 ABI installation of LibTorch. This can be useful in environments with
#' older versions of GLIBC like CentOS7 and older Debian/Ubuntu versions.
#'
#' @export
install_torch <- function(version = "1.12.0", type = install_type(version = version), reinstall = FALSE,
                          path = install_path(), timeout = 360, ...) {
  if (reinstall) {
    unlink(file.path(path, "lib"), recursive = TRUE)
  }

  if (!dir.exists(path)) {
    ok <- dir.create(path, showWarnings = FALSE, recursive = TRUE)
    if (!ok) {
      rlang::abort(c(
        "Failed creating directory",
        paste("Check that you can write to: ", path)
      ))
    }
  }

  # check for write permission
  if (file.access(path, 2) < 0) {
    rlang::abort(c(
      "No write permissions to install torch.",
      paste("Check that you can write to:", path),
      "Or set the TORCH_HOME env var to a path with write permissions."
    ))
  }

  if (!is.null(list(...)$install_config) && is.list(list(...)$install_config)) {
    install_config <- list(...)$install_config
  }

  withr::with_options(
    list(timeout = timeout),
    lantern_install_libs(version, type, path, install_config)
  )

  # reinitialize lantern, might happen if installation fails on load and manual install is required
  if (!identical(list(...)$load, FALSE)) {
    lantern_start(reload = TRUE)
  }
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
install_torch_from_file <- function(version = "1.12.0", type = install_type(version = version), libtorch, liblantern, ...) {
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
get_install_libs_url <- function(version = "1.12.0", type = install_type(version = version)) {
  
  config <- install_config[[version]][[type]][[install_os()]]
  
  if (grepl("darwin", install_os())) {
    config <- config[[R.version$arch]]
  }

  libtorch <- config[["libtorch"]][["url"]]
  liblantern <- config[["liblantern"]]
  list(
    libtorch = maybe_get_pre_cxx11_abi_url(libtorch), 
    liblantern = maybe_get_pre_cxx11_abi_url(liblantern)
  )
}

maybe_get_pre_cxx11_abi_url <- function(url) {
  if (identical(Sys.getenv("PRECXX11ABI", unset = "0"), "1")) {
    if (grepl("lantern", url)) {
      url <- sub("Linux", "LinuxNonABI", url)
    } else if (grepl("libtorch", url)) {
      url <- sub("libtorch-cxx11-abi-shared", "libtorch-shared", url)
    }
  }
  url
}

maybe_switch_precxx11abi <- function(library_info) {
  url <- library_info$url
  library_info$url <- maybe_get_pre_cxx11_abi_url(library_info$url)
  # if the URL changed, we can't keep the md5 hash
  if (!identical(library_info$url, url)) {
    library_info$md5hash <- NULL
  }
  library_info
}

