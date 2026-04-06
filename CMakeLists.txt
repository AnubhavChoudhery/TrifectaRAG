cmake_minimum_required(VERSION 3.16)
project(TrifectaRAG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast /W4)
else()
    add_compile_options(-O2 -march=native -ffast-math -funroll-loops
                        -ftree-vectorize -Wall -Wextra)
endif()

# =============================================================================
# Core static libraries
# =============================================================================

add_library(trifecta_core STATIC
    src/trifecta_core.cpp
    src/lexical_index.cpp
    src/knowledge_graph.cpp
)
target_include_directories(trifecta_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(hnsw STATIC src/hnsw_index.cpp)
target_include_directories(hnsw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hnsw PUBLIC trifecta_core)

# Phase 5 orchestration layer
add_library(trifecta_engine STATIC src/trifecta_engine.cpp)
target_include_directories(trifecta_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(trifecta_engine PUBLIC trifecta_core hnsw)

# =============================================================================
# C++ unit tests (Phases 1–4)
# =============================================================================

enable_testing()

add_executable(test_phase1 tests/test_phase1.cpp)
target_link_libraries(test_phase1 PRIVATE trifecta_core)
add_test(NAME Phase1 COMMAND test_phase1)

add_executable(test_phase2 tests/test_phase2.cpp)
target_link_libraries(test_phase2 PRIVATE hnsw)
add_test(NAME Phase2 COMMAND test_phase2)

add_executable(test_phase3_4 tests/test_phase3_4.cpp)
target_link_libraries(test_phase3_4 PRIVATE trifecta_core)
add_test(NAME Phase3_4 COMMAND test_phase3_4)

# =============================================================================
# pybind11 Python extension — Phase 5
# =============================================================================

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(trifecta_py src/bindings.cpp)
target_link_libraries(trifecta_py PRIVATE trifecta_engine)
target_include_directories(trifecta_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Place the .pyd / .so inside trifecta/ so the Python package is self-contained
set_target_properties(trifecta_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trifecta
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trifecta
)

# On Windows with MinGW toolchain, copy required runtime DLLs next to the .pyd
# so Python can locate them without modifying the system PATH.
if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_MINGW_BIN "$ENV{MSYS_ROOT}/ucrt64/bin")
    if (NOT EXISTS "${_MINGW_BIN}/libgcc_s_seh-1.dll")
        set(_MINGW_BIN "C:/msys64/ucrt64/bin")
    endif()
    foreach(_dll libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
        if (EXISTS "${_MINGW_BIN}/${_dll}")
            add_custom_command(TARGET trifecta_py POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_MINGW_BIN}/${_dll}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/trifecta/${_dll}")
        endif()
    endforeach()
endif()
