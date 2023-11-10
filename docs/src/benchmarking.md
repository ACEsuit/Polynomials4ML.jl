# Benchmark Instructions

For general reference look BenchmarkTools [manual](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/).

A simple way to run benchmarks is to call

```julia
using BenchmarkTools
using PkgBenchmark
using Polynomials4ML

bench = benchmarkpkg(Polynomials4ML)
results = bench.benchmarkgroup

# You can search with macro "@tagged"
results[@tagged "derivative" && "Chebyshev"]
```

You can create `BenchmarkConfig` to control benchmark

```julia
t2 = BenchmarkConfig(env = Dict("JULIA_NUM_THREADS" => 2))
bench_t2 = benchmarkpkg(Polynomials4ML, t2)
```

Benchmarks can be save to a file with

```julia
export_markdown("results.md", bench)
```

Comparing current branch to another branch

```julia
# current branch to "origin/main"
j = judge(Polynomials4ML, "origin/main")
```

Benchmark scaling to different amount of threads

```julia
t4 = BenchmarkConfig(env = Dict("JULIA_NUM_THREADS" => 4))
t8 = BenchmarkConfig(env = Dict("JULIA_NUM_THREADS" => 8))

# Compare how much changing from 4-threads to 8 improves the performance
j = judge(Polynomials4ML, t8, t4)

show(j.benchmarkgroup)
```

## CI Benchmarks

Benchmarks can be run automatically on PR's by adding label "Run Benchmarks" to the PR.

## Adding more benchmarks

Take a look at `benchmark/benchmarks.jl` for an example. If your benchmark depends on an additional packages you need to add the package to `benchmark/Project.toml`.
