cmake_minimum_required(VERSION 3.10.2)
project(3ddfa_v2)
option(DEBUG "build tests or not" ON)

IF(DEBUG)
    message("build for debug")
    SET(CMAKE_CXX_FLAGS "-Wall -g -fPIC")
else()
    SET(CMAKE_CXX_FLAGS "-Wall -O3 -g3 -fPIC")
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_definitions(-fPIC)
#find_package(Threads REQUIRED)

set(Root ${CMAKE_SOURCE_DIR}/..)

set(TNN_LIB_PATH ${Root}/third_party/TNN/libs
            )
set(TNN_INC_DIR ${Root}/third_party/TNN/include
             )

include_directories(${TNN_INC_DIR})
include_directories(${Root}/third_party/stb)
include_directories(base)
include_directories(utils)


link_libraries(-Wl,--whole-archive -Wl,--no-whole-archive)

add_subdirectory(${Root}/third_party/gflags ${Root}/third_party/gflags)
get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
link_libraries(gflags)

link_libraries( ${TNN_LIB_PATH}/libTNN.so )

message(${CMAKE_SOURCE_DIR})
MESSAGE("lib dir:" ${TNN_LIB_PATH})
MESSAGE("INC_DIR:" ${TNN_INC_DIR})


file(GLOB FLAG_SRC "${CMAKE_SOURCE_DIR}/*.cpp")

file(GLOB_RECURSE BASE_SRC "${CMAKE_SOURCE_DIR}/base/*.cpp")

file(GLOB_RECURSE UTIL_SRC "${CMAKE_SOURCE_DIR}/utils/*.cpp")


add_executable(tnn_demo ${FLAG_SRC} ${BASE_SRC} ${UTIL_SRC})
target_link_libraries(tnn_demo PRIVATE  ${CMAKE_THREAD_LIBS_INIT} )