# targets

<details>

* Version: 1.1.3
* GitHub: https://github.com/ropensci/targets
* Source code: https://github.com/cran/targets
* Date/Publication: 2023-05-23 14:10:02 UTC
* Number of recursive dependencies: 177

Run `revdepcheck::revdep_details(, "targets")` for more info

</details>

## Newly broken

*   R CMD check timed out
    

# torchvision

<details>

* Version: 0.5.1
* GitHub: https://github.com/mlverse/torchvision
* Source code: https://github.com/cran/torchvision
* Date/Publication: 2023-04-14 10:00:02 UTC
* Number of recursive dependencies: 43

Run `revdepcheck::revdep_details(, "torchvision")` for more info

</details>

## Newly broken

*   checking examples ...sh: line 1: 12985 Abort trap: 6           LANGUAGE=en _R_CHECK_INTERNALS2_=1 '/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/R' --vanilla > 'torchvision-Ex.Rout' 2>&1 < 'torchvision-Ex.R'
    ```
     ERROR
    Running examples in ‘torchvision-Ex.R’ failed
    The error most likely occurred in:
    
    > ### Name: draw_bounding_boxes
    > ### Title: Draws bounding boxes on image.
    > ### Aliases: draw_bounding_boxes
    > 
    > ### ** Examples
    > 
    > if (torch::torch_is_installed()) {
    + image <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
    + x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
    + y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
    + boxes <- torch::torch_cat(c(x, y, x + 20, y +  10), dim = 2)
    + bboxed <- draw_bounding_boxes(image, boxes, colors = "black", fill = TRUE)
    + tensor_image_browse(bboxed)
    + }
    Error : R: UnableToReadFont `helvetica' @ error/annotate.c/RenderFreetype/1396
    ```

## Newly fixed

*   checking examples ...sh: line 1: 12283 Abort trap: 6           LANGUAGE=en _R_CHECK_INTERNALS2_=1 '/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/R' --vanilla > 'torchvision-Ex.Rout' 2>&1 < 'torchvision-Ex.R'
    ```
     ERROR
    Running examples in ‘torchvision-Ex.R’ failed
    The error most likely occurred in:
    
    > ### Name: draw_bounding_boxes
    > ### Title: Draws bounding boxes on image.
    > ### Aliases: draw_bounding_boxes
    > 
    > ### ** Examples
    > 
    > if (torch::torch_is_installed()) {
    + image <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
    + x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
    + y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
    + boxes <- torch::torch_cat(c(x, y, x + 20, y +  10), dim = 2)
    + bboxed <- draw_bounding_boxes(image, boxes, colors = "black", fill = TRUE)
    + tensor_image_browse(bboxed)
    + }
    Error : R: UnableToReadFont `helvetica' @ error/annotate.c/RenderFreetype/1396
    ```

## In both

*   R CMD check timed out
    

# torchvisionlib

<details>

* Version: 0.4.0
* GitHub: NA
* Source code: https://github.com/cran/torchvisionlib
* Date/Publication: 2023-05-30 13:10:02 UTC
* Number of recursive dependencies: 36

Run `revdepcheck::revdep_details(, "torchvisionlib")` for more info

</details>

## Newly broken

*   checking tests ...
    ```
      Running ‘testthat.R’
     ERROR
    Running the tests in ‘tests/testthat.R’ failed.
    Last 13 lines of output:
      =============================================
      downloaded 68.1 MB
      
      [ FAIL 1 | WARN 2 | SKIP 0 | PASS 13 ]
      
      ══ Failed tests ════════════════════════════════════════════════════════════════
      ── Error ('test-vision.R:5:3'): We can load a detection model ──────────────────
      Error in `download.file(url, destfile = tmp, mode = "wb")`: download from 'https://storage.googleapis.com/torch-lantern-builds/testing-models/fasterrcnn_mobilenet_v3_large_320_fpn.pt' failed
      Backtrace:
          ▆
       1. └─utils::download.file(url, destfile = tmp, mode = "wb") at test-vision.R:5:2
      
      [ FAIL 1 | WARN 2 | SKIP 0 | PASS 13 ]
      Error: Test failures
      Execution halted
    ```

