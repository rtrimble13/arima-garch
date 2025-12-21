# Example programs demonstrating library usage

add_executable(example_basic
    example_basic.cpp
)

target_link_libraries(example_basic PRIVATE arimagarch)

add_executable(example_csv_io
    example_csv_io.cpp
)

target_link_libraries(example_csv_io PRIVATE arimagarch)

add_executable(example_acf_pacf
    example_acf_pacf.cpp
)

target_link_libraries(example_acf_pacf PRIVATE arimagarch)
