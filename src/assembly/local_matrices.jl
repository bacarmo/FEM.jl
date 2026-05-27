"""
    assembly_local_matrix_ϕxϕ(fe::AbstractFEBasis{Deg, Dim}, element_side_lengths::NTuple{Dim, T})

Assemble the local mass matrix ∫_{Ωₑ} ϕₐᵉ ϕᵦᵉ dx for an arbitrary element Ωₑ.

# Arguments
- `fe::AbstractFEBasis{Deg, Dim}`: FE basis with polynomial degree `Deg` and spatial dimension `Dim` (e.g., `Lagrange{3, 1}()`)
- `element_side_lengths::NTuple{Dim, T}`: Side lengths of the (axis-aligned) element along each spatial dimension. The type `T` controls the floating-point precision.

# Returns
- `SMatrix{nb, nb, T, nb*nb}`: Local mass matrix where `nb` is the number of local DOF
"""
function assembly_local_matrix_ϕxϕ end

function assembly_local_matrix_ϕxϕ(
        fe::AbstractFEBasis{Deg, 1},
        element_side_lengths::NTuple{1, T}) where {Deg, T}
    Δx = element_side_lengths[1]
    Npg = Deg + 1
    P, W = legendre(T, Npg)
    ϕP = ntuple(i -> basis_functions(fe, P[i]), Npg)

    nb = length(ϕP[1])
    M = zeros(T, nb, nb)

    jac = Δx / 2
    W_jac = SVector{Npg, T}(jac * W)

    for i in eachindex(ϕP)
        ϕPᵢ = ϕP[i]
        for b in 1:nb, a in 1:nb
            M[a, b] += W_jac[i] * ϕPᵢ[a] * ϕPᵢ[b]
        end
    end

    return SMatrix{nb, nb, T, nb * nb}(M)
end

function assembly_local_matrix_ϕxϕ(
        fe::AbstractFEBasis{Deg, 2},
        element_side_lengths::NTuple{2, T}) where {Deg, T}
    Δx, Δy = element_side_lengths
    Npg = Deg + 1
    P, W = legendre(T, Npg)

    φP₁₁ = basis_functions(fe, P[1], P[1])
    nb = length(φP₁₁)
    M = zeros(T, nb, nb)
    jac = Δx * Δy / 4

    for i in 1:Npg, j in 1:Npg
        φ = basis_functions(fe, P[i], P[j])
        w_jac = W[i] * W[j] * jac
        for b in 1:nb, a in 1:nb
            M[a, b] += w_jac * φ[a] * φ[b]
        end
    end

    return SMatrix{nb, nb, T, nb * nb}(M)
end

"""
    assembly_local_matrix_∇ϕx∇ϕ(fe::AbstractFEBasis{Deg, Dim}, element_side_lengths::NTuple{Dim, T})

Assemble the local stiffness matrix ∫_{Ωₑ} ∇ϕₐᵉ · ∇ϕᵦᵉ dx for an arbitrary element Ωₑ.

# Arguments
- `fe::AbstractFEBasis{Deg, Dim}`: FE basis with polynomial degree `Deg` and spatial dimension `Dim` (e.g., `Lagrange{3, 1}()`)
- `element_side_lengths::NTuple{Dim, T}`: Side lengths of the (axis-aligned) element along each spatial dimension. The type `T` controls the floating-point precision.

# Returns
- `SMatrix{nb, nb, T, nb*nb}`: Local stiffness matrix where `nb` is the number of local DOF
"""
function assembly_local_matrix_∇ϕx∇ϕ end

function assembly_local_matrix_∇ϕx∇ϕ(
        fe::AbstractFEBasis{Deg, 1},
        element_side_lengths::NTuple{1, T}) where {Deg, T}
    Δx = element_side_lengths[1]
    Npg = Deg + 1
    P, W = legendre(T, Npg)
    dϕP = ntuple(i -> basis_functions_derivatives(fe, P[i]), Npg)

    nb = length(dϕP[1])
    K = zeros(T, nb, nb)
    scale = 2 / Δx  # (2 / Δx)^2 * (Δx / 2)
    w_scale = SVector{Npg, T}(scale * W)

    for i in 1:Npg
        dϕPᵢ = dϕP[i]
        for b in 1:nb, a in 1:nb
            K[a, b] += w_scale[i] * dϕPᵢ[a] * dϕPᵢ[b]
        end
    end

    return SMatrix{nb, nb, T, nb * nb}(K)
end

function assembly_local_matrix_∇ϕx∇ϕ(
        fe::AbstractFEBasis{Deg, 2},
        element_side_lengths::NTuple{2, T}) where {Deg, T}
    Δx, Δy = element_side_lengths
    Npg = Deg + 1
    P, W = legendre(T, Npg)
    ∂φ∂ξ, ∂φ∂η = basis_functions_derivatives(fe, P[1], P[1])

    nb = length(∂φ∂ξ)
    K = zeros(T, nb, nb)
    scale_x = Δy / Δx   # (2 / Δx)^2 * (Δx * Δy / 4))
    scale_y = Δx / Δy   # (2 / Δy)^2 * (Δx * Δy / 4))

    for i in 1:Npg, j in 1:Npg
        ∂φ∂ξ, ∂φ∂η = basis_functions_derivatives(fe, P[i], P[j])
        w = W[i] * W[j]
        for b in 1:nb, a in 1:nb
            K[a, b] += w * (∂φ∂ξ[a] * ∂φ∂ξ[b] * scale_x +
                            ∂φ∂η[a] * ∂φ∂η[b] * scale_y)
        end
    end

    return SMatrix{nb, nb, T, nb * nb}(K)
end

"""
    assembly_local_matrix_ϕxc∇ϕ(fe::AbstractFEBasis{Deg, 2}, element_side_lengths::NTuple{2, T}, c::NTuple{2, T})

Assemble the local matrix ∫_{Ωₑ} ϕₐᵉ (c·∇)ϕᵦᵉ dx for an arbitrary element Ωₑ.

# Arguments
- `fe::AbstractFEBasis{Deg, 2}`: FE basis with polynomial degree `Deg` and spatial dimension `2` (e.g., `Lagrange{3, 2}()`)
- `element_side_lengths::NTuple{2, T}`: Side lengths of the (axis-aligned) element along each spatial dimension. The type `T` controls the floating-point precision.
- `c::NTuple{2,T}`: Constant vector

# Returns
- `SMatrix{nb, nb, T, nb*nb}`: Local matrix where `nb` is the number of local DOF
"""
function assembly_local_matrix_ϕxc∇ϕ(
        fe::AbstractFEBasis{Deg, 2},
        element_side_lengths::NTuple{2, T},
        c::NTuple{2, T}) where {Deg, T}
    Δx, Δy = element_side_lengths
    Npg = Deg + 1
    P, W = legendre(T, Npg)

    φP₁₁ = basis_functions(fe, P[1], P[1])
    nb = length(φP₁₁)
    K = zeros(T, nb, nb)
    scale_x = c[1] * Δy / 2   # c₁ * (2 / Δx) * (Δx * Δy / 4))
    scale_y = c[2] * Δx / 2   # c₂ * (2 / Δy) * (Δx * Δy / 4))

    for i in 1:Npg, j in 1:Npg
        φ = basis_functions(fe, P[i], P[j])
        ∂φ∂ξ, ∂φ∂η = basis_functions_derivatives(fe, P[i], P[j])
        w = W[i] * W[j]
        for b in 1:nb, a in 1:nb
            K[a, b] += w * φ[a] * (∂φ∂ξ[b] * scale_x + ∂φ∂η[b] * scale_y)
        end
    end

    return SMatrix{nb, nb, T, nb * nb}(K)
end

"""
    assembly_local_matrix_DG!(DG, ∂ₛg, v, m, eq, xeP, ϕP, W_ϕPϕP)

DGₐᵦ = ∫ ϕₐ(ξ) * ϕᵦ(ξ) * ∂ₛg(x(ξ), Vₕ(xᵉ(ξ))) dx over Ω = (-1,1), with Vₕ(xᵉ(ξ)) = Σ v[eq[j]] ϕⱼ(ξ).

# Arguments
- `DG`: Local matrix (nb × nb), zeroed and filled in-place (upper triangle only)
- `∂ₛg`: Callable `(x, s) -> T`, the derivative of `g` with respect to `s`
- `v`: FE coefficient vector for Vₕ, length `m`
- `m`: Number of free DOFs
- `eq`: Local-to-global DOF map for the element (`EQoLG[e]`), length `nb`
- `xeP`: Physical quadrature points on the element, length `Npg`; `xᵉ(P) = (Δx/2)*(P + 1) + x_start + (e-1)*Δx`
- `ϕP`: Basis functions at each quadrature point; `ϕP[j][a] = ϕₐ(Pⱼ)`
- `W_ϕPϕP`: `W_ϕPϕP[j][a,b] = W[j] * ϕₐ(Pⱼ) * ϕᵦ(Pⱼ)`

# Notes
- Scaling factor and Jacobian are NOT applied here
"""
function assembly_local_matrix_DG!(
        DG::AbstractMatrix{T},
        ∂ₛg::Fun,
        v::AbstractVector{T},
        m::I,
        eq::SVector{nb, I},
        xeP::SVector{Npg, T},
        ϕP::SVector{Npg, SVector{nb, T}},
        W_ϕPϕP::SVector{Npg, <:SMatrix{nb, nb, T}}
) where {Fun, T, I, Npg, nb}
    fill!(DG, zero(T))

    for j in 1:Npg
        ϕPⱼ = ϕP[j]

        # Evaluate Vₕ at quadrature point j
        Vₕx = zero(T)
        for a in 1:nb
            ia = eq[a]
            ia > m && continue
            Vₕx = muladd(v[ia], ϕPⱼ[a], Vₕx)
        end

        # Evaluate ∂ₛg at current point
        g_val = ∂ₛg(xeP[j], Vₕx)

        # Accumulate contributions to local matrix
        W_ϕϕ = W_ϕPϕP[j]
        @inbounds for b in 1:nb, a in 1:b  # Upper triangle: a ≤ b
            DG[a, b] = muladd(W_ϕϕ[a, b], g_val, DG[a, b])
        end
    end

    return nothing
end

"""
    assembly_local_matrix_DF!(DF, f, d, m, eq, φP, W_φPφP)

DFₐᵦ = ∬ φₐ(ξ,η) φᵦ(ξ,η) f(Uₕ(x(ξ,η),y(ξ,η))) dξ dη over reference element Ω = (-1,1)², with Uₕ(x(ξ,η),y(ξ,η)) = Σ d[eq[j]] φⱼ(ξ,η).

# Arguments
- `DF`: Local matrix (nb × nb), zeroed and filled in-place (upper triangle only)
- `f`: Callable `x -> T`
- `d`: FE coefficient vector for Uₕ, length `m`
- `m`: Number of free DOFs
- `eq`: Local-to-global DOF map for the element (`EQoLG[e]`), length `nb`
- `φP`: Basis functions at each quadrature point; `φP[i,j][a] = φₐ(Pᵢ,Pⱼ)`
- `W_φPφP`: `W_φPφP[i,j][a,b] = Wᵢ⋅Wⱼ⋅φₐ(Pᵢ,Pⱼ)⋅φᵦ(Pᵢ,Pⱼ)`

# Notes
- Scaling factor and Jacobian are NOT applied here
"""
function assembly_local_matrix_DF!(
        DF::AbstractMatrix{T},
        f::Fun,
        d::AbstractVector{T},
        m::I,
        eq::SVector{nb, I},
        φP::SMatrix{Npg, Npg, SVector{nb, T}},
        W_φPφP::SMatrix{Npg, Npg, <:SMatrix{nb, nb, T}}
) where {Fun, T, I, Npg, nb}
    fill!(DF, zero(T))

    for j in 1:Npg, i in 1:Npg
        # Compute Uₕ(xᵉ(ξ,η),yᵉ(ξ,η)) at quadrature point (Pᵢ,Pⱼ)
        u_val = zero(T)
        φPᵢⱼ = φP[i, j]
        for a in 1:nb
            ia = eq[a]
            ia > m && continue
            u_val = muladd(d[ia], φPᵢⱼ[a], u_val)
        end

        # Evaluate f at current point
        f_val = f(u_val)

        # Accumulate contributions to local matrix
        W_φφ = W_φPφP[i, j]
        @inbounds for b in 1:nb, a in 1:b # Upper triangle: a ≤ b
            DF[a, b] = muladd(W_φφ[a, b], f_val, DF[a, b])
        end
    end

    return nothing
end
