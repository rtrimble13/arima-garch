# Unit tests
add_executable(ag_tests
    unit/test_sanity.cpp
)

target_link_libraries(ag_tests PRIVATE arimagarch)

# Add CTest integration
add_test(NAME sanity_test COMMAND ag_tests)

# Placeholder for tests (keeping for backward compatibility)
add_executable(test_placeholder
    test_placeholder.cpp
)

target_link_libraries(test_placeholder PRIVATE arimagarch)
