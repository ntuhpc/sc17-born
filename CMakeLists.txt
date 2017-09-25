project(QTCUBE)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)




  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Ofast")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
        message(SEND_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

enable_language(CXX)
find_library(LIBTBB tbb REQUIRED)

FIND_PACKAGE(hypercube NO_MODULE REQUIRED)
INCLUDE_DIRECTORIES(${hypercube_DIRNAME_include})

FIND_PACKAGE(genericIO NO_MODULE REQUIRED)
INCLUDE_DIRECTORIES(${genericIO_DIRNAME_include})





add_subdirectory(base)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/base)

ADD_EXECUTABLE(RTM3D rtm.cc)
ADD_EXECUTABLE(Model3D model.cc)
TARGET_LINK_LIBRARIES(RTM3D base genericCpp hypercube ${LIBTBB})
TARGET_LINK_LIBRARIES(Model3D base genericCpp hypercube ${LIBTBB})


add_subdirectory(util)
install(TARGETS Model3D RTM3D DESTINATION bin)
