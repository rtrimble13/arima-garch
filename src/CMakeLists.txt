# Library target
add_library(arimagarch STATIC
    arima_garch.cpp
)

target_include_directories(arimagarch PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# CLI executable target
add_executable(ag
    main.cpp
)

target_link_libraries(ag PRIVATE arimagarch)
