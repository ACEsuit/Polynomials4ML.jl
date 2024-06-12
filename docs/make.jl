using Polynomials4ML
using Documenter, Literate 

DocMeta.setdocmeta!(Polynomials4ML, :DocTestSetup, :(using Polynomials4ML); recursive=true)

_tutorial_out = joinpath(@__DIR__(), "src", "literate_tutorials")
_tutorial_src = joinpath(@__DIR__(), "..", "tutorials")
_tutorial_out_jl = joinpath(_tutorial_out, "tutorials")

try; run(`mkdir $_tutorial_out`); catch; end 
try; run(`mkdir $_tutorial_out_jl`); catch; end 

for tfile in [ "polyregression.jl", ]
    infile = _tutorial_src * "/" * tfile
    outfile_jl = _tutorial_out * "/tutorials/" * tfile 
    Literate.markdown(infile, _tutorial_out; documenter = true)
    run(`cp $infile $outfile_jl`)
end


makedocs(;
    modules=[Polynomials4ML],
    authors="Christoph Ortner <christophortner0@gmail.com> and contributors",
    # repo="https://github.com/ACEsuit/Polynomials4ML.jl/blob/{commit}{path}#{line}",    
    sitename="Polynomials4ML.jl",
    checkdocs = :none, 
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/Polynomials4ML.jl",
        edit_link="main",
        assets=String[],
        repolink="https://github.com/ACEsuit/Polynomials4ML.jl/blob/{commit}{path}#{line}", 
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md", 
        "Tutorials" => Any[
            "Tutorial Index" => "tutorials.md",
            "Linear Regression" => "literate_tutorials/polyregression.md",
        ],
        "Background" => [ 
                "SH.md",
                "ace.md", ], 
        "Docstrings" => "docstrings.md",
        "Experimental" => "experimental.md",
        "Developter Documentation" => [
            "benchmarking.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/Polynomials4ML.jl",
    devbranch="main",
)
