cmake_minimum_required(VERSION 3.22)
project(sat VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
message(${CMAKE_SYSTEM_NAME})
set(CMAKE_BUILD_TYPE "Release")
if (APPLE)
    message("apple")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3    ")

elseif(UNIX)
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -lm -static -ggdb -fPIC ")
endif()



include_directories(${PROJECT_SOURCE_DIR}/include)
# message(${PROJECT_SOURCE_DIR})
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/src MAIN)
# ========================================SAT
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/sat SAT)
message(${SAT})
add_executable(sat ${SAT} ${MAIN})
