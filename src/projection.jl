"""
    projection_H01(∂ₓu, ∂ᵧu, fe, nel_per_dim, pmin, pmax, dof_map, factorized_lhs_matrix)

Compute the H₀¹ projection of a function onto a 2D FE subspace given its gradient.

Solves: find uₕ ∈ Vₕ such that (∇uₕ, ∇v) = (∇u, ∇v) for all v ∈ Vₕ.

# Arguments
- `∂ₓu`: Callable `(x, y) -> T` for ∂u/∂x
- `∂ᵧu`: Callable `(x, y) -> T` for ∂u/∂y
- `fe`: FE basis with polynomial degree `Deg` and spatial dimension `2` (e.g., `Lagrange{1, 2}()`)
- `nel_per_dim`: Number of elements along each spatial dimension
- `pmin`: Domain lower-left corner `(xmin, ymin)`
- `pmax`: Domain upper-right corner `(xmax, ymax)`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `factorized_lhs_matrix`: Pre-factorized stiffness matrix

# Returns
- `uₕ_coefs`: FE coefficient vector for `uₕ` (length `dof_map.m`)
"""
function projection_H01(
        ∂ₓu::F1,
        ∂ᵧu::F2,
        fe::AbstractFEBasis{Deg, 2},
        nel_per_dim::NTuple{2, Integer},
        pmin::NTuple{2, T},
        pmax::NTuple{2, T},
        dof_map::DOFMap,
        factorized_lhs_matrix
) where {F1, F2, Deg, T}
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    Δx, Δy = element_side_lengths

    Npg = 2 * (Deg + 1)
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1) .+ pmin[1]
    yP = (Δy / 2) .* (P .+ 1) .+ pmin[2]

    ∂φP = SMatrix{Npg, Npg}([basis_functions_derivatives(fe, P[i], P[j])
                             for i in 1:Npg, j in 1:Npg])
    W_∂φ∂ξP = SMatrix{Npg, Npg}([W[i] * W[j] * ∂φP[i, j][1] for i in 1:Npg, j in 1:Npg])
    W_∂φ∂ηP = SMatrix{Npg, Npg}([W[i] * W[j] * ∂φP[i, j][2] for i in 1:Npg, j in 1:Npg])

    uₕ_coefs = zeros(T, dof_map.m)
    vec = zeros(T, dof_map.m)

    assembly_rhs_2d!(
        vec, ∂ₓu, Δy / 2, W_∂φ∂ξP, dof_map, nel_per_dim, element_side_lengths, xP, yP)
    assembly_rhs_2d!(
        uₕ_coefs, ∂ᵧu, Δx / 2, W_∂φ∂ηP, dof_map, nel_per_dim, element_side_lengths, xP, yP)
    @. vec += uₕ_coefs

    ldiv!(uₕ_coefs, factorized_lhs_matrix, vec)

    return uₕ_coefs
end

"""
    projection_H01!(uₕ_coefs, ∂ₓu, ∂ᵧu, 
                    nel_per_dim, element_side_lengths, 
                    dof_map, factorized_lhs_matrix, 
                    xP, yP, W_∂φ∂ξP, W_∂φ∂ηP, vec)

Compute the H₀¹ projection of a function onto a 2D FE subspace given its gradient.

Solves: find uₕ ∈ Vₕ such that (∇uₕ, ∇v) = (∇u, ∇v) for all v ∈ Vₕ.

# Arguments
- `uₕ_coefs`: FE coefficient vector for `uₕ` (overwritten, length `dof_map.m`)
- `∂ₓu`: Callable `(x, y) -> T` for ∂u/∂x
- `∂ᵧu`: Callable `(x, y) -> T` for ∂u/∂y
- `nel_per_dim`: Number of elements along each spatial dimension
- `element_side_lengths`: Side lengths of the axis-aligned element along each spatial dimension
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `factorized_lhs_matrix`: Pre-factorized stiffness matrix
- `xP`: Precomputed fixed part of the physical quadrature points; `xP[i] = (Δx/2)*P[i] + xmin`
- `yP`: Precomputed fixed part of the physical quadrature points; `yP[j] = (Δy/2)*P[j] + ymin`
- `W_∂φ∂ξP`: `W_∂φ∂ξP[i,j][a] = W[i]*W[j] * (∂φₐ/∂ξ)(P[i], P[j])`
- `W_∂φ∂ηP`: `W_∂φ∂ηP[i,j][a] = W[i]*W[j] * (∂φₐ/∂η)(P[i], P[j])`
- `vec`: Work vector (overwritten, length `dof_map.m`)
"""
function projection_H01!(
        uₕ_coefs::AbstractVector{T},
        ∂ₓu::F1,
        ∂ᵧu::F2,
        nel_per_dim::NTuple{2, Integer},
        element_side_lengths::NTuple{2, T},
        dof_map::DOFMap,
        factorized_lhs_matrix::F3,
        xP::SVector{Npg, T},
        yP::SVector{Npg, T},
        W_∂φ∂ξP::SMatrix{Npg, Npg, SVector{nb, T}, Npg²},
        W_∂φ∂ηP::SMatrix{Npg, Npg, SVector{nb, T}, Npg²},
        vec::AbstractVector{T}
) where {T, F1, F2, F3, Npg, Npg², nb}
    Δx, Δy = element_side_lengths

    assembly_rhs_2d!(
        vec, ∂ₓu, Δy / 2, W_∂φ∂ξP, dof_map, nel_per_dim, element_side_lengths, xP, yP)
    assembly_rhs_2d!(
        uₕ_coefs, ∂ᵧu, Δx / 2, W_∂φ∂ηP, dof_map, nel_per_dim, element_side_lengths, xP, yP)
    @. vec += uₕ_coefs

    ldiv!(uₕ_coefs, factorized_lhs_matrix, vec)

    return nothing
end

"""
    projection_L2(u, fe, nel_per_dim, pmin, pmax, dof_map, factorized_lhs_matrix)

Compute the L² projection of a function onto a 1D FE subspace.

Solves: find uₕ ∈ Vₕ such that (uₕ, v) = (u, v) for all v ∈ Vₕ.

# Arguments
- `u`: Callable `x -> T`
- `fe`: FE basis with polynomial degree `Deg` and spatial dimension `1` (e.g., `Lagrange{3, 1}()`)
- `nel_per_dim`: Number of elements along each spatial dimension
- `pmin`: Domain left point `(xmin,)`
- `pmax`: Domain right point `(xmax,)`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `factorized_lhs_matrix`: Pre-factorized stiffness matrix

# Returns
- `uₕ_coefs`: FE coefficient vector for `uₕ` (length `dof_map.m`)
"""
function projection_L2(
        u::F1,
        fe::AbstractFEBasis{Deg, 1},
        nel_per_dim::NTuple{1, Integer},
        pmin::NTuple{1, T},
        pmax::NTuple{1, T},
        dof_map::DOFMap,
        factorized_lhs_matrix::F2
) where {F1, F2, Deg, T}
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    Δx = element_side_lengths[1]

    Npg = 2 * (Deg + 1)
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1) .+ pmin[1]
    ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])
    W_ϕP = SVector{Npg}([W[i] * ϕP[i] for i in 1:Npg])

    uₕ_coefs = zeros(T, dof_map.m)
    rhs_vec = zeros(T, dof_map.m)

    scale = Δx/2
    assembly_rhs_1d!(rhs_vec, u, scale, W_ϕP, dof_map, Δx, xP)

    ldiv!(uₕ_coefs, factorized_lhs_matrix, rhs_vec)

    return uₕ_coefs
end

"""
    projection_L2!(uₕ_coefs, u, element_side_lengths, dof_map, factorized_lhs_matrix, xP, W_ϕP, vec)

Compute the L² projection of a function onto a 1D FE subspace.

Solves: find uₕ ∈ Vₕ such that (uₕ, v) = (u, v) for all v ∈ Vₕ.

# Arguments
- `uₕ_coefs`: FE coefficient vector for `uₕ` (overwritten, length `dof_map.m`)
- `u`: Callable `x -> T`
- `element_side_lengths`: Side lengths of the axis-aligned element along each spatial dimension
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `factorized_lhs_matrix`: Pre-factorized stiffness matrix
- `xP`: Precomputed fixed part of the physical quadrature points; `xP[i] = (Δx/2)*P[i] + xmin`
- `W_ϕP`: `W_ϕP[i][a] = W[i] * ϕₐ(P[i])`
- `vec`: Work vector (overwritten, length `dof_map.m`)
"""
function projection_L2!(
        uₕ_coefs::Vector{T},
        u::F1,
        element_side_lengths::NTuple{1, T},
        dof_map::DOFMap,
        factorized_lhs_matrix::F2,
        xP::SVector{Npg, T},
        W_ϕP::SVector{Npg, SVector{nb, T}},
        vec::Vector{T}
) where {T, F1, F2, Npg, nb}
    Δx = element_side_lengths[1]

    scale = Δx/2
    assembly_rhs_1d!(vec, u, scale, W_ϕP, dof_map, Δx, xP)

    ldiv!(uₕ_coefs, factorized_lhs_matrix, vec)

    return nothing
end

"""
    projection_H02(u, ∇u, Δu, a, fe, nel_per_dim, pmin, pmax, dof_map, M, K, A)

Compute the H₀² projection of a function onto a 1D FE subspace.

Solves: find `uₕ ∈ Vₕ` such that

    a₁*(uₕ, v) + a₂*(∇uₕ, ∇v) + a₃*(Δuₕ, Δv) = a₁*(u, v) + a₂*(∇u, ∇v) + a₃*(Δu, Δv)

for all `v ∈ Vₕ`.

# Arguments
- `u`: Callable `x -> T`
- `∇u`: Callable `x -> T` for `du/dx`
- `Δu`: Callable `x -> T` for `d²u/dx²`
- `a`: Tuple `(a₁, a₂, a₃)`
- `fe`: FE basis with polynomial degree `Deg` and spatial dimension 1 (e.g., `Hermite{3, 1}()`)
- `nel_per_dim`: Number of elements along each spatial dimension.
- `pmin`: Domain lower-left corner
- `pmax`: Domain upper-right corner
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `M`: matrix `(φᵢ, φⱼ)`
- `K`: matrix `(∇φᵢ, ∇φⱼ)`
- `A`: matrix `(Δφᵢ, Δφⱼ)`

`M`, `K`, and `A` must share the same sparsity pattern (same `nzval` ordering).

# Returns
- `uₕ_coefs::Vector{T}`: FE coefficient vector for `uₕ` (length `dof_map.m`)
"""
function projection_H02(
        u::F1,
        ∇u::F2,
        Δu::F3,
        a::NTuple{3, T},
        fe::AbstractFEBasis{Deg, 1},
        nel_per_dim::NTuple{1, I},
        pmin::NTuple{1, T},
        pmax::NTuple{1, T},
        dof_map::DOFMap,
        M::Symmetric{T, SparseMatrixCSC{T, I}},
        K::Symmetric{T, SparseMatrixCSC{T, I}},
        A::Symmetric{T, SparseMatrixCSC{T, I}}
) where {T, Deg, I, F1, F2, F3}
    a₁, a₂, a₃ = a

    # Assemble lhs matrix: R = a₁*M + a₂*K + a₃*A (M, K, A share sparsity pattern)
    R = copy(M)
    @. R.data.nzval = muladd(a₁, M.data.nzval, muladd(a₂, K.data.nzval, a₃ * A.data.nzval))

    # Quadrature setup
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    Δx = element_side_lengths[1]

    Npg = 2 * (Deg + 1)
    P_raw, W_raw = legendre(T, Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1) .+ pmin[1]

    ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])
    W_ϕP = SVector{Npg}([W[i] * ϕP[i] for i in 1:Npg])

    ϕₓP = SVector{Npg}([basis_functions_derivatives(fe, P[i]) for i in 1:Npg])
    W_ϕₓP = SVector{Npg}([W[i] * ϕₓP[i] for i in 1:Npg])

    ϕₓₓP = SVector{Npg}([basis_functions_second_derivatives(fe, P[i]) for i in 1:Npg])
    W_ϕₓₓP = SVector{Npg}([W[i] * ϕₓₓP[i] for i in 1:Npg])

    # Assemble rhs vector: a₁*(u,v) + a₂*(∇u,∇v) + a₃*(Δu,Δv)
    vec1 = zeros(T, dof_map.m)
    assembly_rhs_1d!(vec1, u, a₁ * Δx / 2, W_ϕP, dof_map, Δx, xP)

    vec2 = zeros(T, dof_map.m)
    assembly_rhs_1d!(vec2, ∇u, a₂, W_ϕₓP, dof_map, Δx, xP)  # a₂ * (2/Δx) * (Δx/2)

    vec3 = zeros(T, dof_map.m)
    assembly_rhs_1d!(vec3, Δu, a₃ * 2 / Δx, W_ϕₓₓP, dof_map, Δx, xP)  # a₃ * (2/Δx)² * (Δx/2)

    @. vec1 += vec2 + vec3

    # Solve linear system
    return R \ vec1
end