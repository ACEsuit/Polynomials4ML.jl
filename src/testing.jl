module Testing

using Test 

print_tf(::Test.Pass) = printstyled("+", bold=true, color=:green)
print_tf(::Test.Fail) = printstyled("-", bold=true, color=:red)
print_tf(::Tuple{Test.Error,Bool}) = printstyled("x", bold=true, color=:magenta)

println_slim(::Test.Pass) = printstyled("Test Passed\n", bold=true, color=:green)
println_slim(::Test.Fail) = printstyled("Test Failed\n", bold=true, color=:red)

end