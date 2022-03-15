branch <- "main"

install_config <- list(
  "1.10.2" = list(
    "cpu" = list(
      "darwin" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.2.zip",
          path = "libtorch/",
          filter = ".dylib",
          md5hash = "96ebbf1e2e44f30ee80bf3c8e4a31e15"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/macOS-cpu.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.10.2%2Bcpu.zip",
          path = "libtorch/",
          filter = ".dll",
          md5hash = "c49ddfd07ba65e0ff4a54e041ed22c42"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-cpu.zip", branch)
      ),
      "linux" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcpu.zip",
          md5hash = "99d16043865716f5e38a8d15480b61c6"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-cpu.zip", branch)
      )
    ),
    "10.2" = list(
      "linux" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu102.zip",
          md5hash = "ea32cd0be31f75078208fd484b3982e8"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-102.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.10.2%2Bcu102.zip",
          filter = ".dll",
          md5hash = "8a3496640497cbec833e7fc12ae91c94"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-gpu-102.zip", branch)
      )
    ),
    "11.1" = list(
      "linux" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu111.zip",
          md5hash = "5e0afdd052fa25d150c2ee5bdb64e6fb"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-111.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-1.10.2%2Bcu111.zip",
          filter = ".dll",
          md5hash = "23d56f370a306e9d480a8d60513d9e67"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-gpu-111.zip", branch)
      )
    ),
    "11.3" = list(
      "linux" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu113.zip",
          md5hash = "b7e28b3d8edb8ebb7d5428f0c8fb928d"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Linux-gpu-113.zip", branch)
      ),
      "windows" = list(
        "libtorch" = list(
          path = "libtorch/",
          url = "https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-1.10.2%2Bcu113.zip",
          filter = ".dll",
          md5hash = "d3711140709c64764db837040921b19f"
        ),
        "liblantern" = sprintf("https://storage.googleapis.com/torch-lantern-builds/refs/heads/%s/latest/Windows-gpu-113.zip", branch)
      )
    )
  )
)

install_path <- function(version = "1.10.2") {
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
#' @export
install_torch <- function(version = "1.10.2", type = install_type(version = version), reinstall = FALSE,
                          path = install_path(), timeout = 360, ...) {
  if (reinstall) {
    unlink(path, recursive = TRUE)
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
install_torch_from_file <- function(version = "1.10.2", type = install_type(version = version), libtorch, liblantern, ...) {
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
get_install_libs_url <- function(version = "1.10.2", type = install_type(version = version)) {
  libtorch <- install_config[[version]][[type]][[install_os()]][["libtorch"]][["url"]]
  liblantern <- install_config[[version]][[type]][[install_os()]][["liblantern"]]
  list(libtorch = libtorch, liblantern = liblantern)
}
