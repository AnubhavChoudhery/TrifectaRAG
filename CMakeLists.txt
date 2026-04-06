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

add_library(trifecta_core STATIC
    src/trifecta_core.cpp
    src/lexical_index.cpp
    src/knowledge_graph.cpp
)
target_include_directories(trifecta_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(hnsw STATIC src/hnsw_index.cpp)
target_include_directories(hnsw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hnsw PUBLIC trifecta_core)

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
