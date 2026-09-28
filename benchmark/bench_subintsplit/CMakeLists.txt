add_executable(bench_subintsplit bench_subintsplit.cpp)
target_link_libraries(bench_subintsplit PUBLIC FastLanes::benchmarker)
target_compile_definitions(bench_subintsplit PRIVATE FLS_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
