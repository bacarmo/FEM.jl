"""
    error_L2(u, uₕ_coefs, dof_map, nel_per_dim, element_side_lengths, W, xP, yP, φP)

Compute the L² error ‖u - uₕ‖_L²(Ω) between the exact solution and its FE approximation.

# Arguments
- `u::F`: Exact solution function with signature `u(x, y) → T`
- `uₕ_coefs`: FE coefficient vector for `uₕ`, length `dof_map.m`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `nel_per_dim`: Number of elements in each direction
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension
- `W`: Gauss quadrature weights
- `xP`: Precomputed fixed part of physical quadrature points; `xP = (Δx/2)*(P+1) + xmin`
- `yP`: Precomputed fixed part of physical quadrature points; `yP = (Δy/2)*(P+1) + ymin`
- `φP`: Basis functions at each quadrature point; `φP[i,j][a] = φₐ(Pᵢ,Pⱼ)`
"""
function error_L2(
    u::Fun,
    uₕ_coefs::AbstractVector{T},
    dof_map::DOFMap,
    nel_per_dim::NTuple{2,Integer},
    element_side_lengths::NTuple{2,T},
    W::SVector{Npg,T},
    xP::SVector{Npg,T},
    yP::SVector{Npg,T},
    φP::SMatrix{Npg,Npg,SVector{nb,T},Npg²}
) where {Fun,T,Npg,Npg²,nb}
    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Δx, Δy = element_side_lengths
    Nx, Ny = nel_per_dim

    result = zero(T)
    for ey in 1:Ny
        yeP = @. muladd(ey - 1, Δy, yP)

        for ex in 1:Nx
            xeP = @. muladd(ex - 1, Δx, xP)
            e = ex + (ey - 1) * Nx
            eq = EQoLG[e]

            for j in 1:Npg, i in 1:Npg

                φPᵢⱼ = φP[i, j]
                uh_at_xy = zero(T)
                for a in 1:nb
                    ia = eq[a]
                    ia > m && continue
                    uh_at_xy = muladd(uₕ_coefs[ia], φPᵢⱼ[a], uh_at_xy)
                end

                err = u(xeP[i], yeP[j]) - uh_at_xy
                result += W[i] * W[j] * err * err
            end
        end
    end

    return sqrt(result * Δx * Δy / 4)
end

"""
    error_L2(u, uₕ_coefs, dof_map, Δx, W, xP, ϕP)

Compute the L² error ‖u - uₕ‖_L²(Ω) between the exact solution and its FE approximation.

# Arguments
- `u::F`: Exact solution function with signature `u(x) → T`
- `uₕ_coefs`: FE coefficient vector for `uₕ`, length `dof_map.m`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `Δx`: Uniform element size
- `W`: Gauss quadrature weights
- `xP`: Precomputed fixed part of physical quadrature points; `xP = (Δx/2)*(P+1) + xmin`
- `ϕP`: Basis functions at each quadrature point; `ϕP[i][a] = ϕₐ(Pᵢ)`
"""
function error_L2(
    u::Fun,
    uₕ_coefs::AbstractVector{T},
    dof_map::DOFMap,
    Δx::T,
    W::SVector{Npg,T},
    xP::SVector{Npg,T},
    ϕP::SVector{Npg,SVector{nb,T}}
) where {Fun,T,Npg,nb}
    EQoLG = dof_map.EQoLG
    m = dof_map.m

    result = zero(T)
    for e in eachindex(EQoLG)
        xeP = @. muladd(e - 1, Δx, xP)
        eq = EQoLG[e]

        for j in 1:Npg
            ϕPⱼ = ϕP[j]
            uh_at_x = zero(T)
            for a in 1:nb
                ia = eq[a]
                ia > m && continue
                uh_at_x = muladd(uₕ_coefs[ia], ϕPⱼ[a], uh_at_x)
            end

            err = u(xeP[j]) - uh_at_x
            result += W[j] * err * err
        end
    end

    return sqrt(result * Δx / 2)
end

"""
    norm_H01²(uₕ_coefs, dof_map, Δx, W, ∇ϕP)

Compute the squared H¹₀ norm ‖∇uₕ‖² = ∫_Ω |∇uₕ|² dx of the FE function `uₕ`

# Arguments
- `uₕ_coefs`: FE coefficient vector for `uₕ`, length `dof_map.m`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `Δx`: Uniform element size
- `W`: Gauss quadrature weights
- `∇ϕP`: Basis functions derivatives at each quadrature point; `∇ϕP[i][a] = ∇ϕₐ(Pᵢ)`
"""
function norm_H01²(
    uₕ_coefs::AbstractVector{T},
    dof_map::DOFMap,
    Δx::T,
    W::SVector{Npg,T},
    ∇ϕP::SVector{Npg,SVector{nb,T}}
) where {T,Npg,nb}
    EQoLG = dof_map.EQoLG
    m = dof_map.m

    result = zero(T)
    for e in eachindex(EQoLG)
        eq = EQoLG[e]

        for j in 1:Npg
            ∇ϕPⱼ = ∇ϕP[j]
            ∇uₕ_at_xPⱼ = zero(T)
            for a in 1:nb
                ia = eq[a]
                ia > m && continue
                ∇uₕ_at_xPⱼ = muladd(uₕ_coefs[ia], ∇ϕPⱼ[a], ∇uₕ_at_xPⱼ)
            end

            result = muladd(W[j], ∇uₕ_at_xPⱼ*∇uₕ_at_xPⱼ, result)  # (Δx/2)·(2/Δx)² scaling folded in below
        end
    end

    return result * 2 / Δx
end

"""
    norm_L2(uₕ_coefs, dof_map, element_side_lengths, W, φP)

Compute the L² norm ‖uₕ‖_L²(Ω) of a finite element function on a uniform, axis-aligned mesh in `Dim ∈ {1,2}` spatial dimensions.

# Arguments
- `uₕ_coefs`: FE coefficient vector for `uₕ`, length `dof_map.m`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension
- `W`: Tensor-product Gauss quadrature weights; in 2D, `W[i,j] = wᵢwⱼ`
- `φP`: Basis functions at each quadrature point; in 1D, `φP[i][a] = φₐ(Pᵢ)`; in 2D, `φP[i,j][a] = φₐ(Pᵢ,Pⱼ)`

# Notes
- 1D case: `W::SVector{Npg,T}`, `φP::SVector{Npg,SVector{nb,T}}`
- 2D case: `W::SMatrix{Npg,Npg,T,Npg²}`, `φP::SMatrix{Npg,Npg,SVector{nb,T},Npg²}`
"""
function norm_L2(
    uₕ_coefs::AbstractVector{T},
    dof_map::DOFMap,
    element_side_lengths::NTuple{Dim,T},
    W::StaticArray{S,T,Dim},
    φP::StaticArray{S,SVector{nb,T},Dim}
) where {T,Dim,nb,S<:Tuple}
    EQoLG = dof_map.EQoLG
    m = dof_map.m
    detJ = prod(element_side_lengths ./ 2)

    result = zero(T)
    for eq in EQoLG
        cᵉ = local_coefs(uₕ_coefs, eq, m, Val(nb))
        for (Wₖ, φPₖ) in zip(W, φP)
            uh_at_xPₖ = dot(cᵉ, φPₖ)
            result = muladd(Wₖ, uh_at_xPₖ^2, result)
        end
    end
    return sqrt(detJ * result)
end

"""
    norm_H01(uₕ_coefs, dof_map, element_side_lengths, W, ∇ϕP)

Compute the H¹₀ norm ‖∇uₕ‖ = sqrt(∫_Ω |∇uₕ|²dΩ) of the FE function `uₕ`

# Arguments
- `uₕ_coefs`: FE coefficient vector for `uₕ`, length `dof_map.m`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension
- `W`: Tensor-product Gauss quadrature weights; in 2D, `W[i,j] = wᵢwⱼ`
- `∇φP`: Basis functions derivatives at each quadrature point; in 1D, `∇φP[i][a] = ∂φₐ_∂ξ(Pᵢ)`; in 2D, `∇φP[i,j][1][a] = ∂φₐ_∂ξ(Pᵢ,Pⱼ)`, `∇φP[i,j][2][a] = ∂φₐ_∂η(Pᵢ,Pⱼ)`
"""
function norm_H01 end

function norm_H01(
    uₕ_coefs::AbstractVector{T},
    dof_map::DOFMap,
    element_side_lengths::NTuple{1,T},
    W::SVector{Npg,T},
    ∇φP::SVector{Npg,SVector{nb,T}}
) where {T,Npg,nb}
    EQoLG = dof_map.EQoLG
    m = dof_map.m
    scale = 2 / element_side_lengths[1] # (2/Δx)² * (Δx/2)

    result = zero(T)
    for eq in EQoLG
        cᵉ = local_coefs(uₕ_coefs, eq, m, Val(nb))
        for (Wₖ, ∇φPₖ) in zip(W, ∇φP)
            ∂₁uh_at_xPₖ = dot(cᵉ, ∇φPₖ)
            result = muladd(Wₖ, ∂₁uh_at_xPₖ^2, result)
        end
    end
    return sqrt(scale * result)
end

function norm_H01(
    uₕ_coefs::AbstractVector{T},
    dof_map::DOFMap,
    element_side_lengths::NTuple{2,T},
    W::SMatrix{Npg,Npg,T,Npg²},
    ∇φP::SMatrix{Npg,Npg,NTuple{2,SVector{nb,T}},Npg²}
) where {T,Npg,Npg²,nb}
    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Δx, Δy = element_side_lengths
    scale₁ = Δy/Δx # (2/Δx)² * (Δx*Δy/4)
    scale₂ = Δx/Δy # (2/Δy)² * (Δx*Δy/4)
    W1 = scale₁ .* W
    W2 = scale₂ .* W

    result = zero(T)
    for eq in EQoLG
        cᵉ = local_coefs(uₕ_coefs, eq, m, Val(nb))
        for (Wₖ1, Wₖ2, ∇φPₖ) in zip(W1, W2, ∇φP)
            ∂₁uh_at_xPₖ = dot(cᵉ, ∇φPₖ[1])
            ∂₂uh_at_xPₖ = dot(cᵉ, ∇φPₖ[2])
            result = muladd(Wₖ1, ∂₁uh_at_xPₖ^2, result)
            result = muladd(Wₖ2, ∂₂uh_at_xPₖ^2, result)
        end
    end
    return sqrt(result)
end