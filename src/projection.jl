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