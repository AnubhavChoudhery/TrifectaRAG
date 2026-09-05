cmake_minimum_required(VERSION 3.16)
project(TrifectaRAG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# SKBUILD is set by scikit-build-core when building a pip wheel.
# Wheels must be CPU-portable: no -march=native / /arch:AVX2.
if (DEFINED SKBUILD)
    if (MSVC)
        add_compile_options(/O2 /fp:fast /W4)
    else()
        add_compile_options(-O2 -ffast-math -funroll-loops
                            -ftree-vectorize -Wall -Wextra)
    endif()
else()
    if (MSVC)
        add_compile_options(/O2 /arch:AVX2 /fp:fast /W4)
    else()
        add_compile_options(-O2 -march=native -ffast-math -funroll-loops
                            -ftree-vectorize -Wall -Wextra)
    endif()
endif()

option(TRIFECTA_BUILD_TESTS "Build C++ unit tests" ON)
if (DEFINED SKBUILD)
    set(TRIFECTA_BUILD_TESTS OFF)
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

add_library(trifecta_engine STATIC src/trifecta_engine.cpp)
target_include_directories(trifecta_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(trifecta_engine PUBLIC trifecta_core hnsw)

# =============================================================================
# C++ unit tests (Phases 1–4) — skipped for pip wheels
# =============================================================================

if (TRIFECTA_BUILD_TESTS)
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
endif()

# =============================================================================
# pybind11 Python extension
# =============================================================================

if (DEFINED SKBUILD)
    find_package(pybind11 CONFIG REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(trifecta_py src/bindings.cpp)
target_link_libraries(trifecta_py PRIVATE trifecta_engine)
target_include_directories(trifecta_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (DEFINED SKBUILD)
    install(TARGETS trifecta_py
            LIBRARY DESTINATION trifecta
            RUNTIME DESTINATION trifecta
            ARCHIVE DESTINATION trifecta)
else()
    # Place the .pyd / .so inside trifecta/ so a source checkout works
    # without `pip install` (matches the historical in-tree workflow).
    set_target_properties(trifecta_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trifecta
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trifecta
    )
endif()

# On Windows with MinGW, ship the runtime DLLs next to the extension so
# `import trifecta.trifecta_py` works without a system PATH change.
if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_MINGW_BIN "$ENV{MSYS_ROOT}/ucrt64/bin")
    if (NOT EXISTS "${_MINGW_BIN}/libgcc_s_seh-1.dll")
        set(_MINGW_BIN "C:/msys64/ucrt64/bin")
    endif()
    foreach(_dll libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
        if (EXISTS "${_MINGW_BIN}/${_dll}")
            if (DEFINED SKBUILD)
                install(FILES "${_MINGW_BIN}/${_dll}" DESTINATION trifecta)
            else()
                add_custom_command(TARGET trifecta_py POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_MINGW_BIN}/${_dll}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/trifecta/${_dll}")
            endif()
        endif()
    endforeach()
endif()
