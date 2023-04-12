# luz

<details>

* Version: 0.3.1
* GitHub: https://github.com/mlverse/luz
* Source code: https://github.com/cran/luz
* Date/Publication: 2022-09-06 07:10:02 UTC
* Number of recursive dependencies: 94

Run `revdepcheck::revdep_details(, "luz")` for more info

</details>

## Newly broken

*   checking tests ...
    ```
      Running ‘testthat.R’
     ERROR
    Running the tests in ‘tests/testthat.R’ failed.
    Last 13 lines of output:
       13.   ├─rlang::with_handlers(...)
       14.   │ └─base::tryCatch(.expr, interrupt = `<fn>`)
       15.   │   └─base (local) tryCatchList(expr, classes, parentenv, handlers)
       16.   │     └─base (local) tryCatchOne(expr, names, parentenv, handlers[[1L]])
       17.   │       └─base (local) doTryCatch(return(expr), name, parentenv, handler)
       18.   ├─coro::loop(...)
       19.   │ └─rlang::eval_bare(loop, env)
       20.   └─luz (local) step()
       21.     └─luz:::default_step(ctx)
       22.       └─luz:::fit_one_batch(ctx)
       23.         └─ctx$model$backward(ctx$loss_grad)
      
      [ FAIL 1 | WARN 50 | SKIP 8 | PASS 151 ]
      Error: Test failures
      Execution halted
    ```

# targets

<details>

* Version: 0.14.3
* GitHub: https://github.com/ropensci/targets
* Source code: https://github.com/cran/targets
* Date/Publication: 2023-03-08 13:40:02 UTC
* Number of recursive dependencies: 170

Run `revdepcheck::revdep_details(, "targets")` for more info

</details>

## Newly broken

*   R CMD check timed out
    

# torchvision

<details>

* Version: 0.5.0
* GitHub: https://github.com/mlverse/torchvision
* Source code: https://github.com/cran/torchvision
* Date/Publication: 2023-03-15 12:10:02 UTC
* Number of recursive dependencies: 43

Run `revdepcheck::revdep_details(, "torchvision")` for more info

</details>

## Newly broken

*   checking examples ...sh: line 1: 72641 Abort trap: 6           LANGUAGE=en _R_CHECK_INTERNALS2_=1 '/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/R' --vanilla > 'torchvision-Ex.Rout' 2>&1 < 'torchvision-Ex.R'
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

*   checking dependencies in R code ... WARNING
    ```
    Missing or unexported object: ‘torch::torch_lstsq’
    ```

## Newly fixed

*   checking examples ...sh: line 1: 71953 Abort trap: 6           LANGUAGE=en _R_CHECK_INTERNALS2_=1 '/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/R' --vanilla > 'torchvision-Ex.Rout' 2>&1 < 'torchvision-Ex.R'
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
    ...
    > if (torch::torch_is_installed()) {
    + image <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
    + x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
    + y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
    + boxes <- torch::torch_cat(c(x, y, x + 20, y +  10), dim = 2)
    + bboxed <- draw_bounding_boxes(image, boxes, colors = "black", fill = TRUE)
    + tensor_image_browse(bboxed)
    + }
    Unable to revert mtime: /Library/Fonts
    Error : R: UnableToReadFont `helvetica' @ error/annotate.c/RenderFreetype/1396
    ```

## In both

*   R CMD check timed out
    

