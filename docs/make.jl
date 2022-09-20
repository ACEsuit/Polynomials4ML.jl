using OrthogonalPolynomials
using Documenter

DocMeta.setdocmeta!(OrthPolys4ML, :DocTestSetup, :(using OrthPolys4ML); recursive=true)

makedocs(;
    modules=[OrthPolys4ML],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/OrthPolys4ML.jl/blob/{commit}{path}#{line}",
    sitename="OrthPolys4ML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/OrthPolys4ML.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/OrthPolys4ML.jl",
    devbranch="main",
)
