using OrthogonalPolynomials
using Documenter

DocMeta.setdocmeta!(OrthogonalPolynomials, :DocTestSetup, :(using OrthogonalPolynomials); recursive=true)

makedocs(;
    modules=[OrthogonalPolynomials],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/OrthogonalPolynomials.jl/blob/{commit}{path}#{line}",
    sitename="OrthogonalPolynomials.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/OrthogonalPolynomials.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/OrthogonalPolynomials.jl",
    devbranch="main",
)
