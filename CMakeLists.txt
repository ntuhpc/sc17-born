project(QTCUBE)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
        message(SEND_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

enable_language(CXX)
find_library(LIBTBB tbb REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

FIND_PACKAGE(hypercube NO_MODULE REQUIRED)
INCLUDE_DIRECTORIES(${hypercube_DIRNAME_include})

FIND_PACKAGE(genericIO NO_MODULE REQUIRED)
INCLUDE_DIRECTORIES(${genericIO_DIRNAME_include})

option(ALLINEA "Turn on profiling support" OFF)
if(ALLINEA)
	set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")
	find_library(LIBMAP map-sampler REQUIRED)
	find_library(LIBMAPMPI map-sampler-pmpi REQUIRED)
endif()

add_subdirectory(base)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/base)

ADD_EXECUTABLE(RTM3D rtm.cc)
ADD_EXECUTABLE(Model3D model.cc)
TARGET_LINK_LIBRARIES(RTM3D base genericCpp hypercube ${LIBTBB})
TARGET_LINK_LIBRARIES(Model3D base genericCpp hypercube ${LIBTBB})
if(ALLINEA)
	TARGET_LINK_LIBRARIES(RTM3D ${LIBMAPMPI} ${LIBMAP})
endif()


add_subdirectory(util)
install(TARGETS Model3D RTM3D DESTINATION bin)
