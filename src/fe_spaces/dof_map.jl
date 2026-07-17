# ================================================================
# Dirichlet boundary conditions
# ================================================================
"""
    AbstractDirichletBoundary

Abstract supertype for Dirichlet boundary condition configurations,
specifying which parts of the domain boundary are subject to Dirichlet BCs.
"""
abstract type AbstractDirichletBoundary end

"""
    AllSides <: AbstractDirichletBoundary

Dirichlet BCs imposed on all boundary sides (1D: left and right; 2D: all four sides).
"""
struct AllSides <: AbstractDirichletBoundary end

"""
    LeftRightTop <: AbstractDirichletBoundary

Dirichlet BCs imposed on the left, right, and top sides of a 2D rectangular domain.
"""
struct LeftRightTop <: AbstractDirichletBoundary end

# ================================================================
# DOF map
# ================================================================
"""
    DOFMap{V <: AbstractVector, I <: Integer}

Local-to-global DOF mapping with homogeneous Dirichlet BCs enforced on the FE approximation subspace.

# Fields
- `EQoLG`: Element connectivity array. `EQoLG[e][a]` gives the global free-DOF index of local DOF `a` in element `e`, or the sentinel value `m+1` if that DOF is constrained.
- `m::I`: Number of free DOFs after homogeneous Dirichlet BC enforcement on the FE approximation subspace

# Indexing Convention
- Global functions in the approximation subspace: indices `1, 2, ..., m`
- Global functions NOT in the approximation subspace: sentinel value `m+1`
"""
struct DOFMap{V <: AbstractVector, I <: Integer}
    EQoLG::V
    m::I
end

"""
    DOFMap(fe::AbstractFEBasis, bc::AbstractDirichletBoundary, nel_per_dim::NTuple)

Construct a `DOFMap` for a given finite element basis, Dirichlet BC configuration, and mesh discretization.

# Arguments
- `fe::AbstractFEBasis`: Finite element basis (e.g., `Lagrange{3, 1}()`)
- `bc::AbstractDirichletBoundary`: Dirichlet boundary condition configuration (e.g., `AllSides()`)
- `nel_per_dim::NTuple{Dim, I}`: Number of elements in each direction

# Returns
`DOFMap` containing element connectivity and number of free DOFs.

# Examples
```jldoctest
julia> using FEM: DOFMap, Lagrange, AllSides;

julia> dofmap = DOFMap(Lagrange{1,1}(), AllSides(), (4,));

julia> dofmap.EQoLG
4-element Vector{StaticArraysCore.SVector{2, Int64}}:
 [4, 1]
 [1, 2]
 [2, 3]
 [3, 4]

julia> dofmap.m
3
```
"""
function DOFMap(fe::AbstractFEBasis, bc::AbstractDirichletBoundary,
        nel_per_dim::NTuple{Dim, I}) where {Dim, I <: Integer}
    LG = build_LG(fe, nel_per_dim)
    EQ, m = build_EQ(fe, bc, nel_per_dim)

    num_local_dof = length(LG[1])
    Ne = prod(nel_per_dim)
    EQoLG = Vector{SVector{num_local_dof, I}}(undef, Ne)

    build_EQoLG!(EQoLG, LG, EQ)

    return DOFMap(EQoLG, m)
end

"""
    build_EQoLG!(EQoLG, LG, EQ)

Apply equation numbering to local-to-global map (in-place).

Transforms `LG` (before BCs) into `EQoLG` (after BCs) using `EQ` mapping.
"""
function build_EQoLG!(
        EQoLG::Vector{SVector{num_local_dof, I}},
        LG::Vector{SVector{num_local_dof, I}},
        EQ::Vector{I}) where {I <: Integer, num_local_dof}
    @assert length(EQoLG)==length(LG) "EQoLG and LG length mismatch"

    for e in eachindex(LG)
        LGe = LG[e]
        EQoLG[e] = SVector{num_local_dof, I}(ntuple(a -> EQ[LGe[a]], num_local_dof))
    end

    return nothing
end

# ================================================================
# LG
# ================================================================
"""
    build_LG(fe::AbstractFEBasis, nel_per_dim::NTuple)

Build local-to-global DOF map before Dirichlet BC enforcement.

Returns vector `LG` where `LG[e]` contains global DOF indices for element `e`.
Uses tensor product ordering: DOFs numbered left-to-right, bottom-to-top.
"""
function build_LG end

function build_LG(::Lagrange{Deg, 1}, nel_per_dim::NTuple{1, I}) where {I <: Integer, Deg}
    Nx = nel_per_dim[1]
    num_local_dof = Deg + 1
    LG = Vector{SVector{num_local_dof, I}}(undef, Nx)

    for e in 1:Nx
        start = (e - 1) * Deg
        LG[e] = SVector{num_local_dof, I}(ntuple(k -> I(start + k), num_local_dof))
    end

    return LG
end

function build_LG(::Lagrange{Deg, 2}, nel_per_dim::NTuple{2, I}) where {I <: Integer, Deg}
    Nx, Ny = nel_per_dim
    nx = Deg * Nx + 1  # Total DOFs in x-direction
    num_local_dof = (Deg + 1)^2
    Ne = Nx * Ny
    LG = Vector{SVector{num_local_dof, I}}(undef, Ne)

    # First element DOFs (left-to-right, bottom-to-top)
    first_element = SVector{num_local_dof, I}(
        (I(i + (j - 1) * nx) for i in 1:(Deg + 1), j in 1:(Deg + 1))...
    )

    # Element-to-element shifts in global DOF numbering
    horizontal_shift = I(Deg)
    vertical_shift = I(Deg * nx)

    # Loop over elements (left to right, bottom to top) 
    for j in 1:Ny
        # Global indices of the first element of the j-th layer in the y-direction
        base_row = first_element .+ vertical_shift * I(j - 1)
        for i in 1:Nx
            e = (j - 1) * Nx + i
            LG[e] = base_row .+ horizontal_shift * I(i - 1)
        end
    end

    return LG
end

function build_LG(::Hermite{3, 1}, nel_per_dim::NTuple{1, I}) where {I <: Integer}
    Nx = nel_per_dim[1]
    num_local_dof = 4
    LG = Vector{SVector{num_local_dof, I}}(undef, Nx)

    for e in 1:Nx
        start = 2*e-1
        LG[e] = SVector{num_local_dof, I}(start, start+1, start+2, start+3)
    end

    return LG
end

# ================================================================
# EQ
# ================================================================
"""
    build_EQ(fe::AbstractFEBasis, bc::AbstractDirichletBoundary, nel_per_dim::NTuple)

Build equation numbering array that enforces homogeneous Dirichlet BCs.

# Returns
- `EQ`: Array mapping global DOF index (before BCs) to free DOF index (after BCs)
- `m`: Number of free DOFs

# Indexing Convention
- Global functions in the approximation subspace: indices `1, 2, ..., m`
- Global functions NOT in the approximation subspace: sentinel value `m+1`
"""
function build_EQ end

function build_EQ(
        ::Lagrange{Deg, 1},
        ::AllSides,
        nel_per_dim::NTuple{1, I}) where {I <: Integer, Deg}
    Nx = nel_per_dim[1]
    num_dof = Deg * Nx + 1
    m = I(num_dof - 2)
    sentinel = m + one(I)

    EQ = Vector{I}(undef, num_dof)

    EQ[1] = sentinel
    for i in 2:(num_dof - 1)
        EQ[i] = I(i - 1)
    end
    EQ[num_dof] = sentinel

    return EQ, m
end

function build_EQ(
        ::Lagrange{Deg, 2},
        ::AllSides,
        nel_per_dim::NTuple{2, I}) where {I <: Integer, Deg}
    Nx, Ny = nel_per_dim
    nx = Deg * Nx + 1                   # Total DOFs in x-direction
    ny = Deg * Ny + 1                   # Total DOFs in y-direction
    num_dof = nx * ny                   # Total DOFs

    m = num_dof - 2 * ny - 2 * (nx - 2) # Free DOFs 
    EQ = fill(I(m + 1), num_dof)

    # Re-enumerate interior functions
    for j in 2:(ny - 1)
        cst1 = (j - 1) * nx
        cst2 = (j - 2) * (nx - 2) - 1
        for i in 2:(nx - 1)
            @inbounds EQ[cst1 + i] = I(cst2 + i)
        end
    end

    return EQ, m
end

function build_EQ(
        ::Lagrange{Deg, 2},
        ::LeftRightTop,
        nel_per_dim::NTuple{2, I}) where {I <: Integer, Deg}
    Nx, Ny = nel_per_dim
    nx = Deg * Nx + 1               # Total DOFs in x-direction
    ny = Deg * Ny + 1               # Total DOFs in y-direction
    num_dof = nx * ny               # Total DOFs

    m = num_dof - 2 * ny - (nx - 2) # Free DOFs 
    EQ = fill(I(m + 1), num_dof)

    # Re-enumerate interior and bottom boundary functions (excluding corners)
    for j in 1:(ny - 1)
        cst1 = (j - 1) * nx
        cst2 = (j - 1) * (nx - 2) - 1
        for i in 2:(nx - 1)
            @inbounds EQ[cst1 + i] = I(cst2 + i)
        end
    end

    return EQ, m
end

function build_EQ(
        ::Hermite{3, 1},
        ::AllSides,
        nel_per_dim::NTuple{1, I}) where {I <: Integer}
    Nx = nel_per_dim[1]
    num_dof = 2*(Nx+1)
    m = I(num_dof - 4)
    sentinel = m + one(I)

    EQ = Vector{I}(undef, num_dof)

    EQ[1] = sentinel
    EQ[2] = sentinel
    for i in 3:(num_dof - 2)
        EQ[i] = I(i - 2)
    end
    EQ[num_dof - 1] = sentinel
    EQ[num_dof] = sentinel

    return EQ, m
end