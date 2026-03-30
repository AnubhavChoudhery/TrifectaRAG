#include "../src/hnsw_index.hpp"
#include <iostream>
#include <cassert>
#include <vector>

void test_hnsw_basic() {
    trifecta::HNSWIndex index(4, 16, 200, 100);
    std::vector<float> v1 = {1.0, 0.0, 0.0, 0.0};
    std::vector<float> v2 = {0.9, 0.1, 0.0, 0.0};
    std::vector<float> v3 = {0.0, 0.0, 1.0, 0.0};

    index.add_point(1, v1);
    index.add_point(2, v2);
    index.add_point(3, v3);

    auto results = index.search(v1, 2, 50);
    assert(results.size() >= 2);
    assert(results[0].first == 1);
    std::cout << "HNSW Phase 2 Basic Test Passed!\n";
}

int main() {
    test_hnsw_basic();
    return 0;
}