cmake_minimum_required(VERSION 3.16)
project(jarvis_whisper_bridge LANGUAGES CXX)

set(WHISPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../whisper.cpp")
set(WHISPER_BUILD_DIR "${WHISPER_ROOT}/build")
set(WHISPER_LIBRARY "${WHISPER_BUILD_DIR}/bin/libwhisper.dylib")

if(NOT EXISTS "${WHISPER_LIBRARY}")
    message(FATAL_ERROR "Build the official whisper.cpp source before this bridge")
endif()

add_library(whisper SHARED IMPORTED)
set_target_properties(whisper PROPERTIES IMPORTED_LOCATION "${WHISPER_LIBRARY}")

add_library(jarvis_whisper SHARED whisper_bridge.cpp)
target_compile_features(jarvis_whisper PRIVATE cxx_std_11)
target_include_directories(
    jarvis_whisper
    PRIVATE
        "${WHISPER_ROOT}/include"
        "${WHISPER_ROOT}/ggml/include"
)
target_link_libraries(jarvis_whisper PRIVATE whisper)
set_target_properties(
    jarvis_whisper
    PROPERTIES
        BUILD_RPATH "${WHISPER_BUILD_DIR}/bin"
        INSTALL_RPATH "${WHISPER_BUILD_DIR}/bin"
)
