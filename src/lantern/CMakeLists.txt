cmake_minimum_required(VERSION 3.19.2)
project(lantern)

set(TORCH_VERSION "1.13.1")

if (NOT DEFINED TORCH_PATH)
    if (DEFINED ENV{TORCH_PATH})
        set(TORCH_PATH "ENV{TORCH_PATH}")
    else()
        set(TORCH_PATH "${CMAKE_CURRENT_BINARY_DIR}/libtorch/")
    endif()
endif()

message(STATUS "TORCH_PATH: ${TORCH_PATH}")

if (DEFINED ENV{CUDA} AND NOT "$ENV{CUDA}" STREQUAL "")
    string(REPLACE "\." "" CUDA_VERSION_NUMBER "$ENV{CUDA}")
    set(CUDA_VERSION "$ENV{CUDA}")
    message(STATUS "CUDA VERSION: $ENV{CUDA} | ${CUDA_VERSION} | ${CUDA_VERSION_NUMBER}")
endif()


if (DEFINED ENV{PRECXX11ABI} AND '$ENV{PRECXX11ABI}' STREQUAL '1')
    if (NOT UNIX)
        message(FATAL_ERROR "PRECXX11ABI is only supported on UNIX.")
    endif()
    message(STATUS "Building using -D_GLIBCXX_USE_CXX11_ABI=0")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    set(PRECXX11ABI 1)
endif()

if (NOT EXISTS "${TORCH_PATH}")

    message(STATUS "TORCH_PATH doesn't exists yet. Downloading a pre-built binary.")

    if(DEFINED ENV{TORCH_URL})
        set(TORCH_URL "ENV{TORCH_URL}")
    endif()

    # find an appropriate torch url
    if (NOT DEFINED TORCH_URL)
        message(STATUS "No `TORCH_URL` is defined. Will use the default URL.")

        set(PRECXX11ABI_URL "cxx11-abi-")
        if(DEFINED PRECXX11ABI)
            set(PRECXX11ABI_URL "")
        endif()

        if (APPLE)
            if ('${CMAKE_HOST_SYSTEM_PROCESSOR}' STREQUAL 'x86_64')
                set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${TORCH_VERSION}.zip")
            elseif('${CMAKE_HOST_SYSTEM_PROCESSOR}' STREQUAL 'arm64')    
                set(TORCH_URL "https://github.com/mlverse/libtorch-mac-m1/releases/download/LibTorch-for-R/libtorch-v${TORCH_VERSION}.zip")
            endif()
        elseif(WIN32)
            if (DEFINED CUDA_VERSION_NUMBER)    
                set(TORCH_URL "https://download.pytorch.org/libtorch/cu${CUDA_VERSION_NUMBER}/libtorch-win-shared-with-deps-${TORCH_VERSION}%2Bcu${CUDA_VERSION_NUMBER}.zip")
            else()
                set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
            endif()
        elseif(UNIX)
            if (DEFINED CUDA_VERSION_NUMBER)
                set(TORCH_URL "https://download.pytorch.org/libtorch/cu${CUDA_VERSION_NUMBER}/libtorch-${PRECXX11ABI_URL}shared-with-deps-${TORCH_VERSION}%2Bcu${CUDA_VERSION_NUMBER}.zip")
            else()
                set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-${PRECXX11ABI_URL}shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
            endif()
        endif()
    endif()

    # now download torch
    message(STATUS "Downloading LibTorch from URL: ${TORCH_URL}")
    set(TORCH_DOWNLOAD_PATH "${CMAKE_CURRENT_BINARY_DIR}/libtorch.zip")
    file(DOWNLOAD "${TORCH_URL}" "${TORCH_DOWNLOAD_PATH}")
    file(ARCHIVE_EXTRACT INPUT "${TORCH_DOWNLOAD_PATH}"  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    file(REMOVE "${TORCH_DOWNLOAD_PATH}") # cleanup the zip after extraction to save some space

endif()

if (DEFINED CUDA_VERSION_NUMBER)
    if (WIN32)
        find_package(CUDAToolkit)
    endif()
    enable_language(CUDA)
    add_compile_definitions("CUDA${CUDA_VERSION_NUMBER}")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${TORCH_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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
    src/JITTypes.cpp
    src/ScriptModule.cpp
    src/IValue.cpp
    src/Compile.cpp
    src/Amp.cpp
)

if(APPLE)
  if('${CMAKE_HOST_SYSTEM_PROCESSOR}' STREQUAL 'arm64')
    set(LANTERN_SRC ${LANTERN_SRC} src/AllocatorMPS.cpp)
  endif()
endif()

if(DEFINED ENV{CUDA} AND NOT '$ENV{CUDA}' STREQUAL '')
 
  set(LANTERN_SRC 
      ${LANTERN_SRC} 
      src/AllocatorCuda.cpp
      src/Contrib/SortVertices/sort_vert_kernel.cu
      src/Contrib/SortVertices/sort_vert.cpp
  )
  
  set_source_files_properties(src/Cuda.cpp PROPERTIES COMPILE_DEFINITIONS __NVCC__)
  set_source_files_properties(src/Amp.cpp  PROPERTIES COMPILE_DEFINITIONS __NVCC__)
  
  add_library(lantern SHARED ${LANTERN_SRC})

  if (WIN32)  
    set_property(TARGET lantern PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET lantern PROPERTY CUDA_STANDARD 17)
  else()
    set_property(TARGET lantern PROPERTY CUDA_STANDARD 14)
  endif()
  
else()
  set(LANTERN_SRC 
      ${LANTERN_SRC} 
      src/Contrib/SortVertices/sort_vert_cpu.cpp
  )
  add_library(lantern SHARED ${LANTERN_SRC})
endif()

add_library(lantern::library ALIAS lantern)

target_include_directories(lantern PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

if (APPLE)
    set_property(TARGET lantern
        PROPERTY INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_property(TARGET lantern
        PROPERTY INSTALL_RPATH "$ORIGIN"
    )
endif()

set_property(TARGET lantern
    PROPERTY PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/include/lantern/lantern.h
)

target_link_libraries(lantern ${TORCH_LIBRARIES})

set_property(TARGET lantern PROPERTY CXX_STANDARD 17)

## CPack

set(CPACK_GENERATOR ZIP)

# query version from the R package DESCRIPTION
file(READ "${PROJECT_SOURCE_DIR}/../../DESCRIPTION" DESC_CONTENTS)
string(REGEX MATCH "Version: ([0-9\\\\.]+)" DESC_CONTENTS "${DESC_CONTENTS}")
string(REGEX MATCH "[0-9\\\\.]+" CPACK_PACKAGE_VERSION "${DESC_CONTENTS}")
string(STRIP "${CPACK_PACKAGE_VERSION}" CPACK_PACKAGE_VERSION)

if(DEFINED CUDA_VERSION_NUMBER)
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cu${CUDA_VERSION_NUMBER})
else()
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cpu)
endif()

if (NOT WIN32)
    set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+${CMAKE_SYSTEM_PROCESSOR})
endif()

if (DEFINED PRECXX11ABI)
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}+pre-cxx11")
endif()

include(CPack)

install(
    TARGETS lantern
    # LibTorch currently doesn't use the CPack defaults and installs
    # everything into `lib` even on Windows. We try to match that so
    # lantern.dll ends up in the same directory as torch.dll and friends.
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/lantern
)

option(BUNDLE_DEPENDENCIES "On or off" OFF)
if (BUNDLE_DEPENDENCIES)
    install(DIRECTORY ${TORCH_PATH} DESTINATION .)
    install(TARGETS lantern RUNTIME_DEPENDENCY_SET lantern_deps)
    install(RUNTIME_DEPENDENCY_SET lantern_deps 
		        DESTINATION lib
		        DIRECTORIES "${TORCH_PATH}/lib/" "${CUDA_TOOLKIT_ROOT_DIR}/bin/"
		        UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
		)
endif()

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
