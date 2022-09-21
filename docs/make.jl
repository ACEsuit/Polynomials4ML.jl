using OrthogonalPolynomials
using Documenter

DocMeta.setdocmeta!(Polynomials4ML, :DocTestSetup, :(using Polynomials4ML); recursive=true)

makedocs(;
    modules=[Polynomials4ML],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/Polynomials4ML.jl/blob/{commit}{path}#{line}",
    sitename="Polynomials4ML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/Polynomials4ML.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/Polynomials4ML.jl",
    devbranch="main",
)
