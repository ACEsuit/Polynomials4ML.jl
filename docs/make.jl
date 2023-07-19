using Polynomials4ML
using Documenter, Literate 

DocMeta.setdocmeta!(Polynomials4ML, :DocTestSetup, :(using Polynomials4ML); recursive=true)

_tutorial_out = joinpath(@__DIR__(), "src", "_literate_tutorials")
_tutorial_src = joinpath(@__DIR__(), "..", "tutorials")

Literate.markdown(_tutorial_src * "/polyregression.jl", 
                  _tutorial_out; documenter = true)


makedocs(;
    modules=[Polynomials4ML],
    authors="Christoph Ortner <christophortner0@gmail.com> and contributors",
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
        "API" => "api.md", 
        "Experimental" => "experimental.md",
        "Tutorials" => "tutorials.md",
        "Models" => [ 
                "3D Harmonics" => "SH.md",
                "Cluster Expansion" => "ace.md", ], 
        "Docstrings" => "docstrings.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/Polynomials4ML.jl",
    devbranch="main",
)
