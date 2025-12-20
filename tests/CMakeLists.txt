# Unit tests
add_executable(ag_tests
    unit/test_sanity.cpp
)

target_link_libraries(ag_tests PRIVATE arimagarch)

# Add CTest integration
add_test(NAME sanity_test COMMAND ag_tests)

# Utility tests
add_executable(test_util_expected
    unit/test_util_expected.cpp
)
target_link_libraries(test_util_expected PRIVATE arimagarch)
add_test(NAME test_util_expected COMMAND test_util_expected)

add_executable(test_util_timer
    unit/test_util_timer.cpp
)
target_link_libraries(test_util_timer PRIVATE arimagarch)
add_test(NAME test_util_timer COMMAND test_util_timer)

add_executable(test_util_logging
    unit/test_util_logging.cpp
)
target_link_libraries(test_util_logging PRIVATE arimagarch)
add_test(NAME test_util_logging COMMAND test_util_logging)

# Placeholder for tests (keeping for backward compatibility)
add_executable(test_placeholder
    test_placeholder.cpp
)

target_link_libraries(test_placeholder PRIVATE arimagarch)
