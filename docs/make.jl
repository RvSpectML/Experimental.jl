using Experimental
using Documenter

makedocs(;
    modules=[Experimental],
    authors="Eric Ford",
    repo="https://github.com/RvSpectML/Experimental.jl/blob/{commit}{path}#L{line}",
    sitename="Experimental.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RvSpectML.github.io/Experimental.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RvSpectML/Experimental.jl",
)
