tiny_imagenet_dataset <- dataset(
  "tiny_imagenet",
  inherit = image_folder_dataset,
  tar_name = "tiny-imagenet-200",
  url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
  initialize = function(root, split='train', download = FALSE, ...) {
    
    root <- normalizePath(root)
    
    if (!fs::dir_exists(root))
      fs::dir_create(root)
    
    self$root <- root
    
    if (download)
      self$download()
    
    super$initialize(root = fs::path_join(c(root, self$tar_name, split)), ...)
    
  },
  download = function() {
    
    p <- fs::path_join(c(self$root, self$tar_name))
    
    if (fs::dir_exists(p))
      return(NULL)
    
    raw_path <- fs::path_join(c(self$root, "tiny-imagenet-200.zip"))
    
    inform("Downloding tiny imagenet dataset!")
    
    download.file(self$url, raw_path)
    unzip(raw_path, exdir = self$root)
    
  }
)