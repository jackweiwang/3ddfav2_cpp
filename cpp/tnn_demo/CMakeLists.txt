cmake_minimum_required(VERSION 3.10.2)
project(3ddfa_v2)
option(DEBUG "build tests or not" OFF)

IF(DEBUG)
    message("build for debug")
    SET(CMAKE_CXX_FLAGS "-Wall -g -fPIC")
else()
    SET(CMAKE_CXX_FLAGS "-Wall -O3 -g3 -fPIC")
ENDIF()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_definitions(-fPIC)
find_package(Threads REQUIRED)
set(LIB_DIR ../third_party/TNN/libs)
set(INC_DIR ../third_party/TNN/include ../third_party/imageHelper )

MESSAGE("lib dir:" ${LIB_DIR})
MESSAGE("INC_DIR:" ${INC_DIR})

#add_library(facedetect SHARED src/main/cpp/Ultra_jni.cpp
#                              src/main/cpp/UltraFace.cpp
#                              src/main/cpp/net.cpp)

# add_subdirectory(../third_party/gflags )
# get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
# include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
# link_libraries(gflags)


FILE(GLOB SRC *.cpp
              utils/*.cpp )

include_directories(
    # ${OpenCV_INCLUDE_DIRS}
    ${INC_DIR}
    .
    utils
)

LINK_DIRECTORIES(
    ${LIB_DIR}
)

add_executable(tnn_demo ${SRC})
target_link_libraries(tnn_demo PRIVATE  ${CMAKE_THREAD_LIBS_INIT}  TNN)