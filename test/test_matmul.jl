using LinearAlgebra
using HyperDualNumbers: Hyper
using StrideArrays
using Polynomials4ML
using Test

using ACEbase.Testing: println_slim, print_tf

function Hyper2FloatMat(HyperMat)
    FloatMat = zeros(size(HyperMat))
    for i = 1:size(HyperMat, 1)
        for j = 1:size(HyperMat, 2)
            FloatMat[i, j] = HyperMat[i, j]
        end
    end
    return FloatMat
end


@info("Checking Hyper * Float")

for ntest = 1:30
    local AA, hAA, W
    AA = rand(Float64, (5, 4))
    W = rand(4, 3)

    hAA = Hyper.(AA)
    hAA = StrideArrays.PtrArray(hAA)

    AW = AA * W
    hAW = hAA * W

    @assert size(AW) == size(hAW)
    print_tf(@test AW ≈ Hyper2FloatMat(hAW))
end

##

@info("Checking Float * Hyper")

for ntest = 1:30
    local AA, hAA, W
    AA = rand(Float64, (4, 5))
    W = rand(3, 4)

    hAA = Hyper.(AA)
    hAA = StrideArrays.PtrArray(hAA)

    AW = W * AA
    hAW = W * hAA

    @assert size(AW) == size(hAW)
    print_tf(@test AW ≈ Hyper2FloatMat(hAW))
end

##