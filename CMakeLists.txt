cmake_minimum_required(VERSION 3.16)
project(TrifectaRAG LANGUAGES CXX)

# ─── Global C++ Standard ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # Disable GCC-specific extensions for portability

# ─── Compiler-Specific Optimization Flags ────────────────────────────────────
if (MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast /W4)
else()
    add_compile_options(-O2 -march=native -ffast-math -funroll-loops
                        -ftree-vectorize -Wall -Wextra)
endif()

# ─── Core Library ─────────────────────────────────────────────────────────────
add_library(trifecta_core STATIC
    src/trifecta_core.cpp
)
target_include_directories(trifecta_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ─── Phase 1 Tests ────────────────────────────────────────────────────────────
add_executable(test_phase1
    tests/test_phase1.cpp
)
target_link_libraries(test_phase1 PRIVATE trifecta_core)

enable_testing()
add_test(NAME Phase1 COMMAND test_phase1)
