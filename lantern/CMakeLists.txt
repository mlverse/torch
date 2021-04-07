cmake_minimum_required(VERSION 3.16)

project(lantern)

############################################################
# Helpers
############################################################

function(download_file url filename)
    if(NOT EXISTS ${filename})
        message("-- Downloading ${filename}: ${url}")
        file(DOWNLOAD  ${url} ${filename}
             TIMEOUT 300
             TLS_VERIFY ON
        )
    endif()
endfunction()

function(retrieve_lib url name)
    download_file("${url}" "${CMAKE_CURRENT_BINARY_DIR}/${name}.zip")

    if(NOT EXISTS "${name}")
        message("-- Extracting ${name}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${name}.zip
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endfunction()

############################################################
# Torch
############################################################

if(APPLE)
    if(DEFINED ENV{CUDA})
        message(FATAL_ERROR, "CUDA not supported on macos")
    else()
        retrieve_lib("https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip" "libtorch")
    endif()
elseif(WIN32)
    if(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '10.1')
        message(FATAL_ERROR, "CUDA 10.1 is no longer supported")
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '10.2')
        add_compile_definitions(CUDA102)
        retrieve_lib("https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.8.1.zip" "libtorch")
    else()
        retrieve_lib("https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.8.1%2Bcpu.zip" "libtorch")
    endif()
elseif(UNIX)
    if(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '9.2')
        message(FATAL_ERROR, "CUDA 9.2 is no longer supported")
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '10.1')
        message(FATAL_ERROR, "CUDA 10.1 is no longer supported")
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '10.2')
        add_compile_definitions(CUDA102)
        retrieve_lib("https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.1.zip" "libtorch")
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '11.1')
        add_compile_definitions(CUDA111)
        retrieve_lib("https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip" "libtorch")
    else()
        retrieve_lib("https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip" "libtorch")
    endif()
else()
    message(FATAL_ERROR, "OS not supported.")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_BINARY_DIR}/libtorch/share/cmake/Torch")


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# see https://github.com/pytorch/pytorch/issues/18607#issuecomment-480665557
if(WIN32)
  add_compile_options("/permissive")
  add_compile_options("/permissive-")
endif()


############################################################
# Library
############################################################

set(LANTERN_SRC 
    src/lantern.cpp
    src/TensorOptions.cpp
    src/Dtype.cpp
    src/Tensor.cpp
    src/Device.cpp
    src/utils.cpp
    src/MemoryFormat.cpp
    src/Generator.cpp
    src/QScheme.cpp
    src/TensorList.cpp
    src/Scalar.cpp
    src/Dimname.cpp
    src/Delete.cpp
    src/Reduction.cpp
    src/Quantization.cpp
    src/Autograd.cpp
    src/Function.cpp
    src/Layout.cpp
    src/Indexing.cpp
    src/Cuda.cpp
    src/NNUtilsRnn.cpp
    src/Storage.cpp
    src/Save.cpp
    src/Contrib/Sparsemax.cpp
    src/Threads.cpp
    src/Trace.cpp
    src/Stack.cpp
    src/Allocator.cpp
    src/Backends.cpp
)


if(DEFINED ENV{CUDA})
  set(LANTERN_SRC 
      ${LANTERN_SRC} 
      src/AllocatorCuda.cpp
  )
endif()

add_library(lantern SHARED ${LANTERN_SRC})
add_library(lantern::library ALIAS lantern)

target_include_directories(lantern PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(lantern ${TORCH_LIBRARIES})
set_property(TARGET lantern PROPERTY CXX_STANDARD 17)

############################################################
# Tests
############################################################

add_executable(lanterntest
    tests/init.cpp
    tests/main.cpp
)

target_include_directories(lanterntest PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    tests/
)

target_link_libraries(lanterntest ${CMAKE_DL_LIBS})
