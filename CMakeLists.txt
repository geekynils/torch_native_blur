# Invoke cmake with -DCMAKE_PREFIX_PATH to set the path to libtorch
# Like this: cmake -DCMAKE_PREFIX_PATH=/home/nils/libtorch ..

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_native_blur)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Sanitize w/o xcode
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address,undefined")
# set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address,undefined")

set(STB_HEADERS CACHE FILEPATH "")
include_directories(${STB_HEADERS})

add_executable(torch_native_blur torch_native_blur.cpp)
target_link_libraries(torch_native_blur "${TORCH_LIBRARIES}")
set_property(TARGET torch_native_blur PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)