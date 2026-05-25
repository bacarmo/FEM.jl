module FEM

using StaticArrays: SVector

# Exports
export Lagrange, Hermite

# Includes
include("fe_spaces/fe_basis.jl")
include("fe_spaces/dof_map.jl")
end
