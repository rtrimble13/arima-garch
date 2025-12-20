#include "test_framework.hpp"
#include <iostream>

namespace ag_test {
    int test_count = 0;
    int test_passed = 0;
}

void report_test_results(const std::string& suite_name) {
    std::cout << "\n=== " << suite_name << " ===\n";
    std::cout << "\nResults: " << ag_test::test_passed << "/" << ag_test::test_count << " tests passed\n";
}

int get_test_result() {
    return ag_test::test_passed == ag_test::test_count ? 0 : 1;
}
