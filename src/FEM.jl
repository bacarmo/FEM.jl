module FEM

using StaticArrays: SVector, SMatrix
using GaussQuadrature: legendre
using SparseArrays: sparse
using LinearAlgebra: Symmetric, lmul!

# Exports
export Lagrange, Hermite

# Includes
include("fe_spaces/fe_basis.jl")
include("fe_spaces/dof_map.jl")
include("assembly/local_matrices.jl")
include("assembly/global_matrices.jl")
include("assembly/local_vectors.jl")
include("assembly/global_vectors.jl")
end
