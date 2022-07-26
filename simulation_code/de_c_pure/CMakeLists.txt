cmake_minimum_required(VERSION 3.12)
project(de_c_pure)

list(APPEND CMAKE_PREFIX_PATH "/apps/software/standard/compiler/gcc/7.1.0/armadillo/8.300.3")

find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

add_executable(de_c_pure main.cpp vec3.hpp)
target_link_libraries(de_c_pure ${LAPACK_LIBRARIES} ${BLAS_LIBARIES} ${ARMADILLO_LIBRARIES} ${TORCH_LIBRARIES})