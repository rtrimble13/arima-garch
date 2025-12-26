# Benchmarks for performance tracking

# Likelihood computation benchmark
add_executable(bench_likelihood
    bench_likelihood.cpp
)

target_link_libraries(bench_likelihood PRIVATE arimagarch)

# Optimizer benchmark
add_executable(bench_optimizer
    bench_optimizer.cpp
)

target_link_libraries(bench_optimizer PRIVATE arimagarch)
