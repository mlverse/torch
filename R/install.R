branch <- "main"
torch_version <- "2.7.1"

#' Install Torch
#'
#' Installs Torch and its dependencies.
#'
#' @param reinstall Re-install Torch even if its already installed?
#' @param ... Currently unused.
#' @param .inform_restart if `TRUE` and running in an `interactive()` session, after
#'   installation it will print a message to inform the user that the session must
#'   be restarted for torch to work correctly.
#' 
#' @details
#' This function is mainly controlled by environment variables that can be used
#' to override the defaults:
#' 
#' - `TORCH_HOME`: the installation path. By default dependencies are installed
#'    within the package directory. Eg what's given by `system.file(package="torch")`.
#' - `TORCH_URL`: A URL, path to a ZIP file or a directory containing a LibTorch version.
#'    Files will be installed/copied to the `TORCH_HOME` directory.
#' - `LANTERN_URL`: Same as `TORCH_URL` but for the Lantern library.
#' - `TORCH_INSTALL_DEBUG`: Setting it to 1, shows debug log messages during installation.
#' - `PRECXX11ABI`: DEPRECATED. No longer has effects.
#'    Setting it to `1` will will trigger the installation of
#'    a Pre-cxx11 ABI installation of LibTorch. This can be useful in environments with
#'    older versions of GLIBC like CentOS7 and older Debian/Ubuntu versions.
#' - `LANTERN_BASE_URL`: The base URL for lantern files. This allows passing a directory
#'   where lantern binaries are located. The filename is then constructed as usual.
#' - `TORCH_COMMIT_SHA`: torch repository commit sha to be used when querying lantern
#'   uploads. Set it to `'none'` to avoid looking for build for that commit and 
#'   use the latest build for the branch.
#' - `CUDA`: We try to automatically detect the CUDA version installed in your system,
#'   but you might want to manually set it here. You can also disable CUDA installation
#'   by setting it to 'cpu'.
#' - `TORCH_R_VERSION`: The R torch version. It's unlikely that you need to change it,
#'   but it can be useful if you don't have the R package installed, but want to
#'   install the dependencies.
#' 
#' The \code{TORCH_INSTALL} environment
#' variable can be set to \code{0} to prevent auto-installing torch and \code{TORCH_LOAD} set to \code{0}
#' to avoid loading dependencies automatically. These environment variables are meant for advanced use
#' cases and troubleshooting only.
#' When timeout error occurs during library archive download, or length of downloaded files differ from
#' reported length, an increase of the \code{timeout} value should help.
#' 
#' @export
install_torch <- function(reinstall = FALSE, ..., .inform_restart = TRUE) {
  have_installed <- !torch_is_installed() || reinstall
  
  liblantern <- lantern_url()
  libtorch <- libtorch_url()

  install_lib("torch", libtorch, reinstall)
  install_lib("lantern", liblantern, reinstall)
  
  if (.inform_restart && have_installed && interactive()) {
    cli_inform(c(
      v = "torch dependencies have been installed.",
      i = "You must restart your session to use {.pkg torch} correctly."
    ))
  }
  
  return(invisible(TRUE))
}

#' A simple exported version of install_path
#' Returns the torch installation path.
#' @param check_writable If `TRUE`, checks if the installation path is writable.
#' @export
torch_install_path <- function(check_writable = FALSE) {
 normalizePath(inst_path(check_writable = check_writable), mustWork = FALSE)
}

#' Verifies if torch is installed
#'
#' @export
torch_is_installed <- function() {
  lib_is_installed("lantern", torch_install_path()) && 
    lib_is_installed("torch", torch_install_path())
}

install_lib <- function(libname, url, reinstall = FALSE) {
  inst_path <- torch_install_path(check_writable = TRUE)
  installer_message(c(
    "We are now proceeding to download and installing lantern and torch.",
    "The installation path is: {.path {inst_path}}"
  ))
  
  if (lib_is_installed(libname, inst_path) && !reinstall) {
    installer_message(c(
      "An installation of {.strong {libname}} already exists.",
      "Found file at {.path {inst_path}}."
    ))
    return(invisible(TRUE))
  }
  
  # The library URL can be 3 different things:
  # - real URL
  # - path to a zip file containing the library
  # - path to a directory containing the files to be installed. 
  if (is_url(url)) {
    tmp <- tempfile(fileext = ".zip")
    file.create(tmp)
    on.exit({file.remove(tmp)}, add = TRUE)
    
    download_file(url = url, destfile = tmp)
    url <- tmp
  }
  
  if (grepl("\\.zip$", url) && file.exists(url)) {
    tmp_ex <- tempfile()
    dir.create(tmp_ex)
    on.exit({unlink(tmp_ex)}, add = TRUE)
    
    utils::unzip(url, exdir = tmp_ex)
    url <- tmp_ex
  }

  if (dir.exists(url)) {
    # sometimes the extracted dir includes another directory that contains the
    # library within it.
    if (!lib_is_installed(libname, url)) {
      dirs <- list.files(url, full.names = TRUE)
      if (length(dirs) == 1) {
        url <- dirs
      }
    }
    
    # this where the installation actually happens
    if (lib_is_installed(libname, url)) {
      if (!dir.exists(inst_path)) {
        dir.create(inst_path, recursive = TRUE)
      }
      
      file.copy(
        from = dir(url, full.names = TRUE),
        to = file.path(inst_path, ""),
        recursive = TRUE
      )
    }
  }
  
  if (lib_is_installed(libname, inst_path)) {
    return(invisible(TRUE))
  } 
  
  cli_abort(c(
    "Installation failed.",
    "Could not install {.strong {libname}} from {.val {url}}."
  ))
}

lib_is_installed <- function(libname, install_path) {
  if (file.exists(file.path(install_path, "lib", lib_name(libname))))
    return(TRUE)
  
  if (file.exists(file.path(install_path, "lib64", lib_name(libname))))
    return(TRUE)
  
  if (file.exists(file.path(install_path, "bin", lib_name(libname))))
    return(TRUE)
  
  FALSE
}

inst_path <- function(check_writable = FALSE) {
  install_path <- Sys.getenv("TORCH_HOME")
  if (nzchar(install_path)) {    
    if (!check_writable) {
      return(install_path)
    }

    if (can_write_into(install_path)) {
      return(install_path)
    } else {
      cli_abort(c(
        "x" = "{.pkg torch} cannot write into configured {.var TORCH_HOME} {.path {install_path}}.",
        "i" = "Please configure {.var TORCH_HOME} to be one of {.val {.libPath()}}"
      ))
    }
  }
  install_path <- system.file("", package = "torch")
  if (!check_writable) {
    return(install_path)
  }
  
  if (can_write_into(install_path)) {
    return(install_path)
  } else {
    cli_abort(c("x" = "{.pkg torch} cannot write into {.path {install_path}}.",
                "i" = "Please configure a {.var TORCH_HOME} variable with a writable folder",
                " " = "and run {.fn install_torch()} again",
                "i" = "Or run R under the {.emph root} user {.strong once} to perform the {.fn install_torch()} ",
                " " = "if you use system level package manager like {.pkg r2u}"))
  }
}

libtorch_url <- function() {
  url <- Sys.getenv("TORCH_URL", "")
  
  if (nzchar(url))
    return(url)
  
  if (is_macos()) {
    arch <- architecture()
    url <- glue::glue("https://github.com/mlverse/libtorch-mac-m1/releases/download/LibTorch-for-R/libtorch-{arch}-v{torch_version}.zip") 
  }
  kind <- installation_kind()
  if (is_windows()) {
    url <- glue::glue("https://download.pytorch.org/libtorch/{kind}/libtorch-win-shared-with-deps-{torch_version}%2B{kind}.zip")
  }
  if (is_linux()) {
    url <- glue::glue("https://download.pytorch.org/libtorch/{kind}/libtorch-cxx11-abi-shared-with-deps-{torch_version}%2B{kind}.zip")
  }
  
  installer_message(c(
    "LibTorch will be downloaded from:",
    "{.url {url}}"
  ))
  
  url
}

lantern_url <- function() {
  url <- Sys.getenv("LANTERN_URL", "")
  
  # If a `LANTERN_URL` is set we use it for the download.
  if (url != "")
    return(url)
  
  # Otherwise we construct it from available information
  # file name we want to download has the following format:
  # lantern-<pkg-version>+<cpu|cu113>+<arch>+<precxx11>-<os>.zip
  pkg_version <- torch_r_version()
  kind <- installation_kind()
  arch <- architecture()
  os <- os_name()
  
  fname <- paste0("lantern-", pkg_version, "+", kind)
  if (is_linux() || is_macos()) {
    fname <- paste0(fname, "+", arch)
  }
  
  fname <- paste0(fname, "-", os, ".zip")
  
  # we now query the base URL for that file name. There are 2 cases:
  # the package has been installed with remotes::install_github()
  # in this case the RemoteSha is stored in the package description and
  # we can install directly from it.
  # In the other cases, we download the latest version of the 'branch' variable.
  base_url <- Sys.getenv("LANTERN_BASE_URL", "")

  if (!nzchar(base_url)) {
    base_url <- "https://torch-cdn.mlverse.org/binaries/"
  
    remote_sha <- Sys.getenv("TORCH_COMMIT_SHA", "")
    if (remote_sha == "none") {
      remote_sha <- NA # if the user explicitly set it to none, we won't search for the SHA
    } else if (!nzchar(remote_sha)) {
      remote_sha <- desc::desc(package = "torch")$get("RemoteSha")  
      
      # pak adds the package version as the remote_sha when installing from
      # CRAN which breaks our assumption that it's always a commit sha. 
      # We identify this case and manually remove it.
      if (is_package_version(remote_sha)) {
        remote_sha <- NA
      }
    }
    
    if (is.na(remote_sha)) {
      installer_message(c(
        "Could not find the SHA of the commit that installed the package.",
        "Using the latest build for the specified branch: {.val {branch}}."
      ))
      base_url <- paste0(base_url, "refs/heads/", branch, "/latest/")
    } else {
      installer_message(c(
        "Could find the SHA of the commit that installed the package.",
        "SHA: {.val {remote_sha}}."
      ))
      base_url <- paste0(base_url, remote_sha, "/")
    }
  }

  final_url <- paste0(base_url, fname)

  if (is_url(final_url)) {
    final_url <- utils::URLencode(final_url)
  }

  installer_message(c(
    "Lantern will be downloaded from the following URL:",
    "{.url {final_url}}"
  ))
  
  final_url
}

torch_r_version <- function() {
  version <- Sys.getenv("TORCH_R_VERSION", unset = "")
  if (version != "") return(version)
  
  as.character(utils::packageVersion("torch"))
}

os_name <- function() {
  os <- Sys.info()["sysname"]
  if (!grepl('windows', os, ignore.case = TRUE)) {
    os
  } else {
    "win64"
  }
}

architecture <- function() {
  arch <- Sys.info()["machine"]

  if (!is_x86_64(arch) && (!is_macos())) {
    cli_abort("Architecture {.val {arch}} is not supported in this OS.")
  }
  
  if ((!is_arm64(arch)) && (!is_x86_64(arch))) {
    cli_abort("Unsupported architecture {.val {arch}}.")
  }
  
  installer_message("Architecture is {.val {arch}}")
  arch
}

is_x86_64 <- function(x) {
  x %in% c("x86_64", "x86-64")
}

is_arm64 <- function(x) {
  x %in% c("arm64")
}

installation_kind <- function() {
  cu <- cuda_version()
  if (is.null(cu)) {
    installer_message("No cuda instalation has been found. Using {.val cpu}.")
    return("cpu")
  } else if (cu == "cpu") {
    installer_message("{.envvar CUDA} is set to {.val cpu}, so using the {.val cpu}.")
    return("cpu")
  } else {
    cu <- paste0("cu", gsub(".", "", cu, fixed = TRUE))
    installer_message("Installation kind will be {.val {cu}}.")
    return(cu)
  }
}

cuda_version <- function() {
  
  version <- Sys.getenv("CUDA", "")
  if (version == "") {
    version <- NULL
  }
  
  if (!is.null(version)) {
    installer_message("{.envvar CUDA} has been specified. The CUDA version is {.strong {version}}")
    return(version)
  }
    
  if (is_windows()) {
    return(cuda_version_windows())
  }
  
  if (is_linux()) {
    return(cuda_version_linux())
  }
  
  installer_message("Not on Windows or Linux. No CUDA installation supported.")
  return(NULL)
}

cuda_version_linux <- function() {
  
  cuda_version <- NULL
  cuda_home <- Sys.getenv("CUDA_HOME")
  
  if (nzchar(cuda_home)) {
    installer_message("{.envvar CUDA_HOME}={.path {cuda_home}} is specified.")
  } else {
    installer_message("{.envvar CUDA_HOME} is not specified. Looking in conventional locations.")
  }
  
  # This file no longer exists with cuda >= 11
  if (nzchar(cuda_home)) {
    versions_file <- file.path(cuda_home, "version.txt")
    cuda_version <- cuda_version_from_version_txt_file(versions_file)
  }
  
  # Query nvcc from cuda in cuda_home path.
  if (nzchar(cuda_home) && is.null(cuda_version)) {
    nvcc_path <- file.path(cuda_home, "bin", "nvcc")
    cuda_version <- nvcc_version_from_path(nvcc_path)
  }
  
  # Try to find in conventional location.
  if (is.null(cuda_version)) {
    versions_file <- "/usr/local/cuda/version.txt"
    cuda_version <- cuda_version_from_version_txt_file(versions_file)
  }
  
  # Query nvcc from conventional location
  if (is.null(cuda_version)) {
    cuda_version <- nvcc_version_from_path("/usr/local/cuda/bin/nvcc")
  }
  
  if (is.null(cuda_version)) {
    cuda_version <- nvcc_version_from_path("nvcc")
  }
  
  check_supported_cuda_version_linux(cuda_version)
  
  cuda_version
}

cuda_version_windows <- function() {
  cuda_version <- NULL
  cuda_path <- Sys.getenv("CUDA_PATH")
  
  if (nzchar(cuda_path)) {
    installer_message(c(
      "{.envvar CUDA_PATH}={.path {cuda_path}}.", 
      "Trying to find CUDA in this path."
    ))
  } else {
    installer_message(c(
      "{.envvar CUDA_PATH} is not specified.", 
      "Searching for installation in conventional locations."
    ))
  }
  
  if (nzchar(cuda_path)) {
    versions_file <- file.path(cuda_path, "version.txt")
    cuda_version <- cuda_version_from_version_txt_file(versions_file)
  }
  
  # Query nvcc from cuda in cuda_path.
  if (nzchar(cuda_path) && is.null(cuda_version)) {
    nvcc_path <- file.path(cuda_path, "bin", "nvcc.exe")
    cuda_version <- nvcc_version_from_path(nvcc_path)
  }
  
  if (is.null(cuda_version)) {
    installer_message("Trying to use the nvcc version that might be on your path.")
    cuda_version <- nvcc_version_from_path("nvcc")
  }
  
  check_supported_cuda_version_windows(cuda_version)
  
  cuda_version
}

check_supported_cuda_version_windows <- function(version) {
  supported_versions <- c("12.6", "12.8")
  check_supported_version(version, supported_versions)
}

check_supported_cuda_version_linux <- function(version) {
  supported_versions <- c("12.6", "12.8")
  check_supported_version(version, supported_versions)
}

check_supported_version <- function(version, supported_versions) {
  if (!is.null(version)) {
    if (!version %in% supported_versions) {
      cli_abort(c(
        x = "Unsupported CUDA version {.val {version}}",
        i = "Currently supported versions are: {.val {supported_versions}}."
      ))
    }
  }
}

is_macos <- function() {
  grepl("darwin", Sys.info()["sysname"], ignore.case = TRUE)
}

is_windows <- function() {
  grepl("windows", Sys.info()["sysname"], ignore.case = TRUE)
}

is_linux <- function() {
  grepl("linux", Sys.info()["sysname"], ignore.case = TRUE)
}

cuda_version_from_version_txt_file <- function(versions_file) {
  cuda_version <- NULL
  if (file.exists(versions_file)) {
    cuda_version <- gsub("CUDA Version |\\.[0-9]+$", "", readLines(versions_file))
    installer_message(c(
      "Found CUDA version {.strong {cuda_version}}.",
      "This version was specified in {.path {versions_file}}"
    ))
  } else {
    installer_message(c(
      "Could not find a CUDA version in {.path {versions_file}}."
    ))
  }
  cuda_version
}

nvcc_version_from_path <- function(nvcc_path) {
  suppressWarnings(
    nvcc <- tryCatch(system2(nvcc_path, "--version", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
  )
  
  if (is.null(nvcc) || !any(grepl("release", nvcc))) {
    installer_message(c(
      "Tried to query nvcc from {.path {nvcc_path}}, but was unable to find a CUDA version."
    )) 
    return(NULL)
  }
  
  version <- gsub(".*release |, V.*", "", nvcc[grepl("release", nvcc)])
  installer_message(c(
    "Found CUDA version {.strong {version}}.",
    "It was found by querying nvcc at {.path {nvcc_path}}."
  ))
  
  version
}

installer_message <- function(msg) {
  if (!is_truthy(Sys.getenv("TORCH_INSTALL_DEBUG", FALSE)))
    return(invisible(msg))
  names(msg) <- rep("i", length(msg))
  cli_inform(msg, env = parent.frame(), class = "torch_install")
}

is_truthy <- function(x) {
  if (length(x) == 0) {
    return(FALSE)
  }
  
  if (length(x) > 1) {
    value_error("Unexpected value")
  }
  
  if (x == "") {
    return(FALSE)
  }
  
  if (x == "1") {
    return(TRUE)
  }
    
  (toupper(x) == TRUE)
}

lib_name <- function(name = "torch") {
  if (.Platform$OS.type == "unix") {
    paste0("lib", name, lib_ext())
  } else {
    paste0(name, lib_ext())
  }
}

lib_ext <- function() {
  if (grepl("darwin", version$os))
    ".dylib"
  else if (grepl("linux", version$os))
    ".so"
  else
    ".dll"
}

is_url <- function(x) {
  grepl("^https", x) || grepl("^http", x)
}

#' Install Torch from files
#'
#' List the Torch and Lantern libraries URLs to download as local files in order to proceed with  \code{install_torch_from_file()}.
#'
#' @inheritParams install_torch
#' @param version Not used
#' @param type Not used. This function is deprecated.
#'
#' @rdname install_torch_from_file
#' @export
get_install_libs_url <- function(version = NA, type = NA) {
  if (!is.na(type)) {
    cli_abort("Please use the env vars described in {.fn install_torch} to configure the installation type.")
  }
  if (!is.na(version)) {
    cli_abort("It's not possible to configure the libtorch version.")
  }
  out <- list(
    libtorch = libtorch_url(), 
    liblantern = lantern_url()
  )
  
  if (interactive()) {
    cli_inform(c(
      "Follow the links to download the dependencies, then set the {.envvar TORCH_URL} and {.envvar LANTERN_URL} env vars to the file paths.",
      "LibTorch: {.url {out$libtorch}}",
      "LibLantern: {.url {out$liblantern}}"
    ))
  }
  invisible(out)
}

#' Install Torch from files
#'
#' Installs Torch and its dependencies from files.
#'
#' @inheritParams install_torch
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
#' @examples
#' \dontrun{
#' # on a linux CPU platform 
#' get_install_libs_url()
#' # then after making both files available into /tmp/
#' Sys.setenv(TORCH_URL="/tmp/libtorch-v1.13.1.zip")
#' Sys.setenv(LANTERN_URL="/tmp/lantern-0.9.1.9001+cpu+arm64-Darwin.zip")
#' torch::install_torch()
#' }
#' @export
install_torch_from_file <- function(version = NA, type = NA, libtorch, liblantern, ...) {
  cli_abort(c(
    "This function is now deprecated. The same results can be achieved with {.fn install_torch}.",
    i = "Use the envvars {.envvar TORCH_URL} and {.envvar LANTERN_URL} to set the file locations."
  ))
}

download_file <- function(url, destfile) {
  withr::local_options(timeout = max(600, getOption("timeout", default = 60)))
  tryCatch({
    utils::download.file(url = url, destfile = destfile)  
  }, 
  error = function(e) {
    
    additional_messages <- c()
    if (grepl("torch-lantern-builds", url)) {
      if (!grepl("refs", url)) {
        additional_messages <- c(
          i = "If you installed from GitHub, there might be no lantern build for that commit. Try setting {.envvar TORCH_COMMIT_SHA} to {.val none} to install from the latest commit."
        )
      }
    }
    cli_abort(c(
      x = "Unable to download from {.url {url}}",
      i = "Please verify that the URL is not blocked by your firewall. See also {.url https://torch.mlverse.org/docs/articles/installation.html#file-based-download}",
      additional_messages
    ), parent = e)
  })
}

is_package_version <- function(x) {
  # .standard_regexps()$valid_numeric_version
  regex <- "([[:digit:]]+[.-])*[[:digit:]]+"
  grepl(regex, x)
}

can_write_into <- function(path) {
  file.access(path, 2 ) >= 0
}
