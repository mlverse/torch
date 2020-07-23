
IMG_EXTENSIONS <-  c('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp')

has_file_allowed_extension <- function(filename, extensions) {
  tolower(fs::path_ext(filename)) %in% tolower(extensions )
}

is_image_file <- function(filename) {
  has_file_allowed_extension(filename, IMG_EXTENSIONS)
}

folder_make_dataset <- function(directory, class_to_idx, extensions = NULL, is_valid_file = NULL) {
  directory <- normalizePath(directory)
  
  both_none <- is.null(extensions) && is.null(is_valid_file)
  both_something <- !is.null(extensions) && ! is.null(is_valid_file)
  
  if (both_none || both_something)
    value_error("Both extensions and is_valid_file cannot be None or not None at the same time")
  
  if (!is.null(extensions)) {
    is_valid_file <- function(filename) {
      has_file_allowed_extension(filename, extensions)
    }
  }
  
  paths <- c()
  indexes <- c()
  
  for (target_class in sort(names(class_to_idx))) {
    
    class_index <- class_to_idx[target_class]
    target_dir <- fs::path_join(c(directory, target_class))
    
    if (!fs::is_dir(target_dir))
      next
    
    fnames <- fs::dir_ls(target_dir, recurse = TRUE)
    fnames <- fnames[is_valid_file(fnames)]
   
    paths <- c(paths, fnames)
    indexes <- c(indexes, rep(class_index, length(fnames)))
  }
    
  list(
    paths,
    indexes
  )
}

folder_dataset <- dataset(
  name = "folder",
  initialize = function(root, loader, extensions = NULL, transform = NULL, 
                        target_transform = NULL, is_valid_file = NULL) {
    
    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    
    class_to_idx <- self$.find_classes(root)
    samples <- folder_make_dataset(self$root, class_to_idx, extensions, is_valid_file)
    
    if (length(samples[[1]]) == 0) {
      
      msg <- glue::glue("Found 0 files in subfolders of {self$root}")
      if (!is.null(extensions)) {
        msg <- paste0(msg, glue::glue("\nSupported extensions are {paste(extensions, collapse=',')}"))
      }
      
      runtime_error(msg)
    }
    
    self$loader <- loader
    self$extensions <- extensions
    
    self$classes <- names(class_to_idx)
    self$class_to_idx <- class_to_idx
    self$samples <- samples
    self$targets <- samples[[2]]
      
  },
  .find_classes = function(dir) {
    dirs <- fs::dir_ls(dir, recurse = FALSE, type = "directory")
    dirs <- sapply(fs::path_split(dirs), function(x) tail(x, 1))
    class_too_idx <- seq_along(dirs)
    names(class_too_idx) <- sort(dirs)
    class_too_idx
  },
  .getitem = function(index) {
    
    path <- self$samples[[1]][index]
    target <- self$samples[[2]][index]
    
    sample <- self$loader(path)
    
    if (!is.null(self$transform))
      sample <- self$transform(sample)
    
    if (!is.null(self$target_transform))
      target <- self$target_transform(target)
    
    list(sample, target)
  },
  .length = function() {
    length(self$samples[[1]])
  }
)

magick_loader <- function(path) {
  
  if (!requireNamespace("magick"))
    runtime_error("The `magick` package must be installed to load images.")
  
  magick::image_read(path)
}

image_folder_dataset <- dataset(
  "image_folder",
  inherit = folder_dataset,
  initialize = function(root, transform=NULL, target_transform=NULL,
                        loader=magick_loader, is_valid_file=NULL) {
    
    if (!is.null(is_valid_file))
      extensions <- NULL
    else
      extensions <- IMG_EXTENSIONS
    
    super$initialize(root, loader, extensions, transform=transform,
                     target_transform=target_transform,
                     is_valid_file=is_valid_file)
    self$imgs <- self$samples
  }
)