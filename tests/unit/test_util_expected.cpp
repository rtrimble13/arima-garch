#include "ag/util/Expected.hpp"
#include "test_framework.hpp"
#include <string>

// Test basic value construction
TEST(expected_value_construction) {
    ag::expected<int, std::string> e1{42};
    REQUIRE(e1.has_value());
    REQUIRE(*e1 == 42);
    REQUIRE(e1.value() == 42);
}

// Test error construction
TEST(expected_error_construction) {
    ag::expected<int, std::string> e1{ag::unexpected<std::string>("error")};
    REQUIRE(!e1.has_value());
    REQUIRE(e1.error() == "error");
}

// Test value_or
TEST(expected_value_or) {
    ag::expected<int, std::string> e1{42};
    ag::expected<int, std::string> e2{ag::unexpected<std::string>("error")};
    
    REQUIRE(e1.value_or(0) == 42);
    REQUIRE(e2.value_or(0) == 0);
}

// Test bool conversion
TEST(expected_bool_conversion) {
    ag::expected<int, std::string> e1{42};
    ag::expected<int, std::string> e2{ag::unexpected<std::string>("error")};
    
    REQUIRE(static_cast<bool>(e1) == true);
    REQUIRE(static_cast<bool>(e2) == false);
}

// Test copy construction
TEST(expected_copy_construction) {
    ag::expected<int, std::string> e1{42};
    ag::expected<int, std::string> e2{e1};
    
    REQUIRE(e2.has_value());
    REQUIRE(e2.value() == 42);
    
    ag::expected<int, std::string> e3{ag::unexpected<std::string>("error")};
    ag::expected<int, std::string> e4{e3};
    
    REQUIRE(!e4.has_value());
    REQUIRE(e4.error() == "error");
}

// Test move construction
TEST(expected_move_construction) {
    ag::expected<std::string, std::string> e1{"value"};
    ag::expected<std::string, std::string> e2{std::move(e1)};
    
    REQUIRE(e2.has_value());
    REQUIRE(e2.value() == "value");
}

// Test with custom types
TEST(expected_custom_types) {
    struct Data {
        int x;
        std::string s;
    };
    
    ag::expected<Data, std::string> e1{Data{42, "hello"}};
    REQUIRE(e1.has_value());
    REQUIRE(e1->x == 42);
    REQUIRE(e1->s == "hello");
}

// Test exception on accessing value when error
TEST(expected_value_throws_on_error) {
    ag::expected<int, std::string> e1{ag::unexpected<std::string>("error")};
    
    bool threw = false;
    try {
        int val = e1.value();
        (void)val;  // Suppress unused variable warning
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

// Test exception on accessing error when value
TEST(expected_error_throws_on_value) {
    ag::expected<int, std::string> e1{42};
    
    bool threw = false;
    try {
        std::string err = e1.error();
        (void)err;  // Suppress unused variable warning
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

int main() {
    ag_test::report_test_results("Expected Tests");
    return ag_test::get_test_result();
}
