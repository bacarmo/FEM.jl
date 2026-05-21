using FEM
using Documenter

DocMeta.setdocmeta!(FEM, :DocTestSetup, :(using FEM); recursive=true)

makedocs(;
    modules=[FEM],
    authors="Bruno Alves do Carmo",
    sitename="FEM.jl",
    format=Documenter.HTML(;
        canonical="https://bacarmo.github.io/FEM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bacarmo/FEM.jl",
    devbranch="main",
)
