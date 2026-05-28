"""
    assembly_rhs_1d!(F, f, scale, W_basisP, dof_map, Δx, xP)

Assemble global RHS vector for 1D FEM by integrating f(x) against basis functions.

# Arguments
- `F`: Global RHS vector (modified in-place, length `dof_map.m`)
- `f`: Callable `x -> T`
- `scale`: Scaling factor (typically `Δx/2` for F[i] = ∫Ω f(x) ϕᵢ(x) dx)
- `W_basisP`: Precomputed weighted basis evaluations at quadrature points
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `Δx`: Uniform element size
- `xP`: Precomputed fixed part of physical quadrature points; `xP = (Δx/2)*(P+1) + xmin`
"""
function assembly_rhs_1d!(
        F::AbstractVector{T},
        f::Fun,
        scale::T,
        W_basisP::SVector{Npg, SVector{nb, T}},
        dof_map::DOFMap,
        Δx::T,
        xP::SVector{Npg, T}
) where {Fun, T, Npg, nb}
    fill!(F, zero(T))

    EQoLG = dof_map.EQoLG
    m = dof_map.m

    for e in eachindex(EQoLG)
        eq = EQoLG[e]
        xeP = @. muladd(e - 1, Δx, xP)

        for k in 1:Npg
            fx = f(xeP[k])
            Wₖ_basisPₖ = W_basisP[k]

            for a in 1:nb
                ia = eq[a]
                ia > m && continue
                F[ia] = muladd(Wₖ_basisPₖ[a], fx, F[ia])
            end
        end
    end

    lmul!(scale, F)
    return nothing
end

"""
    assembly_rhs_2d!(F, f, scale, W_basisP, dof_map, nel_per_dim, element_side_lengths, xP, yP)

Assemble global RHS vector for 2D FEM by integrating f(x,y) against basis functions.

# Arguments
- `F`: Global RHS vector (modified in-place, length `dof_map.m`)
- `f`: Callable `x -> T`
- `scale`: Scaling factor (typically `Δx*Δy/4` for F[i] = ∫Ω f(x,y) φᵢ(x,y) dx dy)
- `W_basisP`: Precomputed weighted basis evaluations at quadrature points
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `nel_per_dim`: Number of elements in each direction
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension.
- `xP`: Precomputed fixed part of physical quadrature points; `xP = (Δx/2)*(P+1) + xmin`
- `yP`: Precomputed fixed part of physical quadrature points; `yP = (Δy/2)*(P+1) + ymin`
"""
function assembly_rhs_2d!(
        F::AbstractVector{T},
        f::Fun,
        scale::T,
        W_basisP::SMatrix{Npg, Npg, SVector{nb, T}},
        dof_map::DOFMap,
        nel_per_dim::NTuple{2, I},
        element_side_lengths::NTuple{2, T},
        xP::SVector{Npg, T},
        yP::SVector{Npg, T}
) where {Fun, T, I, Npg, nb}
    fill!(F, zero(T))

    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Nx, Ny = nel_per_dim
    Δx, Δy = element_side_lengths

    for ey in 1:Ny
        yeP = @. muladd(ey - 1, Δy, yP)

        for ex in 1:Nx
            xeP = @. muladd(ex - 1, Δx, xP)
            e = ex + (ey - 1) * Nx
            eq = EQoLG[e]

            Fe = zero(SVector{nb, T})
            for j in 1:Npg, i in 1:Npg
                Fe = muladd(f(xeP[i], yeP[j]), W_basisP[i, j], Fe)
            end

            for a in 1:nb
                ia = eq[a]
                ia <= m && (F[ia] += Fe[a])
            end
        end
    end

    lmul!(scale, F)
    return nothing
end

"""
    assembly_nonlinearity_F!(F, scale, f, d, dof_map, element_side_lengths, φP, W_φP)

Fᵢ = scale * ∬ φᵢ(x,y) * f(Uₕ(x,y)) dx dy over Ω, with Uₕ(x,y) = Σ d[j] φⱼ(x,y).

# Arguments
- `F`: Global vector (zeroed and filled in-place, length `dof_map.m`)
- `scale`: Scaling factor
- `f`: Nonlinearity function, callable `s -> T`
- `d`: FE coefficient vector for Uₕ, length `dof_map.m`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension.
- `φP`: Basis functions at each quadrature point; `φP[i,j][a] = φₐ(Pᵢ,Pⱼ)`
- `W_φP`: `W_φP[i,j][a] = Wᵢ⋅Wⱼ⋅φₐ(Pᵢ,Pⱼ)`
"""
function assembly_nonlinearity_F!(
        F::AbstractVector{T},
        scale::T,
        f::Fun,
        d::AbstractVector{T},
        dof_map::DOFMap,
        element_side_lengths::NTuple{2, T},
        φP::SMatrix{Npg, Npg, SVector{nb, T}},
        W_φP::SMatrix{Npg, Npg, SVector{nb, T}}
) where {Fun, T, Npg, nb}
    fill!(F, zero(T))

    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Δx, Δy = element_side_lengths

    for e in eachindex(EQoLG)
        global_indices = EQoLG[e]

        for j in 1:Npg, i in 1:Npg
            uh_at_xy = zero(T)
            φPᵢⱼ = φP[i, j]
            for a in 1:nb
                ia = global_indices[a]
                ia > m && continue
                uh_at_xy = muladd(d[ia], φPᵢⱼ[a], uh_at_xy)
            end

            fuh = f(uh_at_xy)
            W_φPᵢⱼ = W_φP[i, j]
            for a in 1:nb
                ia = global_indices[a]
                ia > m && continue
                F[ia] = muladd(W_φPᵢⱼ[a], fuh, F[ia])
            end
        end
    end

    scale_jacobian = scale * (Δx * Δy / 4)
    lmul!(scale_jacobian, F)

    return nothing
end

"""
    assembly_nonlinearity_G!(G, scale, g, v, dof_map, Δx, xP, ϕP, W_ϕP)

Gᵢ = scale * ∫ ϕᵢ(x) * g(x, Vₕ(x)) dx over Ω, with Vₕ(x) = Σ v[j] ϕⱼ(x).

# Arguments
- `G`: Global vector (zeroed and filled in-place, length `dof_map.m`)
- `scale`: Scaling factor
- `g`: Callable (x, s) -> T
- `v`: FE coefficient vector for Vₕ, length `dof_map.m`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `ϕP`: Basis functions at each quadrature point; `ϕP[j][a] = ϕₐ(Pⱼ)`
- `W_ϕP`: `W_ϕP[j][a] = W[j] * ϕₐ(Pⱼ)`
"""
function assembly_nonlinearity_G!(
        G::AbstractVector{T},
        scale::T,
        g::Fun,
        v::AbstractVector{T},
        dof_map::DOFMap,
        Δx::T,
        xP::SVector{Npg, T},
        ϕP::SVector{Npg, SVector{nb, T}},
        W_ϕP::SVector{Npg, SVector{nb, T}}
) where {Fun, T, Npg, nb}
    fill!(G, zero(T))

    EQoLG = dof_map.EQoLG
    m = dof_map.m

    for e in eachindex(EQoLG)
        xeP = @. muladd(e - 1, Δx, xP)
        eq = EQoLG[e]

        for j in 1:Npg
            Vₕx = zero(T)
            ϕPⱼ = ϕP[j]
            for a in 1:nb
                ia = eq[a]
                ia > m && continue
                Vₕx = muladd(v[ia], ϕPⱼ[a], Vₕx)
            end

            gxVₕx = g(xeP[j], Vₕx)
            W_ϕPⱼ = W_ϕP[j]
            for a in 1:nb
                ia = eq[a]
                ia > m && continue
                G[ia] = muladd(W_ϕPⱼ[a], gxVₕx, G[ia])
            end
        end
    end

    scale_jacobian = scale * (Δx / 2)
    lmul!(scale_jacobian, G)

    return nothing
end

"""
    assembly_∫basis(fe, element_side_lengths, dof_map)
 
Assemble the global integral vector
```math
b_i = \\int_Ω φᵢ(x, y) dΩ, \\quad i = 1, \\ldots, m,
```
where ``\\{\\varphi_i\\}`` are the global basis functions associated with the free DOFs of `dof_map`.

Returns a `Vector{T}` of length `m = dof_map.m`.
 
# Arguments
- `fe::AbstractFEBasis{Deg, 2}`: FE basis with polynomial degree `Deg` and spatial dimension `2` (e.g., `Lagrange{1, 2}()`)
- `element_side_lengths::NTuple{2, T}`: Side lengths of the (axis-aligned) element along each spatial dimension.
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
"""
function assembly_∫basis(
        fe::AbstractFEBasis{Deg, 2},
        element_side_lengths::NTuple{2, T},
        dof_map::DOFMap
) where {T, Deg}
    b_local = assembly_local_∫basis(fe, element_side_lengths)
    EQoLG = dof_map.EQoLG
    m = dof_map.m
    b = zeros(T, m)

    for e in eachindex(EQoLG)
        eq = EQoLG[e]
        for a in eachindex(b_local)
            ia = eq[a]
            ia <= m && (b[ia] += b_local[a])
        end
    end

    return b
end