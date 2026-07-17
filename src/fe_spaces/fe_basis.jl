"""
    AbstractFEBasis{Deg, Dim}

Abstract supertype for finite element basis functions parametrised by polynomial degree `Deg` and spatial dimension `Dim`.

Concrete subtypes: [`Lagrange{Deg, Dim}`](@ref), [`Hermite{Deg, Dim}`](@ref).
"""
abstract type AbstractFEBasis{Deg, Dim} end

"""
    Lagrange{Deg, Dim} <: AbstractFEBasis{Deg, Dim}

Lagrange basis of degree `Deg` in spatial dimension `Dim`.
"""
struct Lagrange{Deg, Dim} <: AbstractFEBasis{Deg, Dim} end

"""
    Hermite{Deg, Dim} <: AbstractFEBasis{Deg, Dim}

Hermite basis of degree `Deg` in spatial dimension `Dim`.
"""
struct Hermite{Deg, Dim} <: AbstractFEBasis{Deg, Dim} end

# ============================================================================
# LAGRANGE 1D - Reference element [-1, 1]
# ============================================================================

"""
    basis_functions(::Lagrange{Deg, 1}, ξ)

Lagrange basis functions on the reference element [-1, 1].

| Deg | Nodes Layout    | Returned SVector |
|:----|:----------------|:-----------------|
| `1` | `1-----------2` | ``(\\frac{1-ξ}{2}, \\frac{1+ξ}{2})`` |
| `2` | `1-----2-----3` | ``(\\frac{ξ(ξ-1)}{2}, (1-ξ)(1+ξ), \\frac{ξ(ξ+1)}{2})`` |
| `3` | `1---2---3---4` | ``(\\frac{(3ξ+1)(3ξ-1)(1-ξ)}{16}, \\frac{(27ξ-9)(ξ+1)(ξ-1)}{16}, \\frac{(27ξ+9)(ξ+1)(1-ξ)}{16}, \\frac{(3ξ+1)(3ξ-1)(ξ+1)}{16})`` |
"""
@inline function basis_functions(::Lagrange{1, 1}, ξ::T) where {T <: Real}
    cst = T(0.5)
    SVector{2, T}(cst * (1 - ξ), cst * (1 + ξ))
end

@inline function basis_functions(::Lagrange{2, 1}, ξ::T) where {T <: Real}
    cst = T(0.5)
    SVector{3, T}(cst * ξ * (ξ - 1), (1 - ξ) * (1 + ξ), cst * ξ * (ξ + 1))
end

@inline function basis_functions(::Lagrange{3, 1}, ξ::T) where {T <: Real}
    cst = T(0.0625)
    SVector{4, T}(
        cst * (3 * ξ + 1) * (3 * ξ - 1) * (1 - ξ),
        cst * (27 * ξ - 9) * (ξ + 1) * (ξ - 1),
        cst * (27 * ξ + 9) * (ξ + 1) * (1 - ξ),
        cst * (3 * ξ + 1) * (3 * ξ - 1) * (ξ + 1)
    )
end

"""
    basis_functions_derivatives(::Lagrange{Deg, 1}, ξ)

Lagrange basis functions derivatives on the reference element [-1, 1].

| Deg | Nodes Layout    | Returned SVector |
|:----|:----------------|:-----------------|
| `1` | `1-----------2` | ``(-\\frac{1}{2}, \\frac{1}{2})`` |
| `2` | `1-----2-----3` | ``(ξ-\\frac{1}{2}, -2ξ, ξ+\\frac{1}{2})`` |
| `3` | `1---2---3---4` | ``(\\frac{ξ(18-27ξ)+1}{16}, \\frac{-27-ξ(18-81ξ)}{16}, \\frac{27-ξ(18+81ξ)}{16}, \\frac{ξ(18+27ξ)-1}{16})`` |
"""
@inline function basis_functions_derivatives(::Lagrange{1, 1}, ξ::T) where {T <: Real}
    cst = T(0.5)
    SVector{2, T}(-cst, cst)
end

@inline function basis_functions_derivatives(::Lagrange{2, 1}, ξ::T) where {T <: Real}
    cst = T(0.5)
    SVector{3, T}(ξ - cst, -2 * ξ, ξ + cst)
end

@inline function basis_functions_derivatives(::Lagrange{3, 1}, ξ::T) where {T <: Real}
    cst = T(0.0625)
    SVector{4, T}(
        cst * (ξ * (18 - 27 * ξ) + 1),
        cst * (-27 - ξ * (18 - 81 * ξ)),
        cst * (27 - ξ * (18 + 81 * ξ)),
        cst * (ξ * (18 + 27 * ξ) - 1)
    )
end

# ============================================================================
# LAGRANGE 2D - Reference element [-1,1] × [-1,1]
# ============================================================================

"""
    basis_functions(::Lagrange{Deg, 2}, ξ, η)

Tensor-product Lagrange basis functions on the reference element [-1,1] × [-1,1].

# Arguments
- `ξ::T`: First coordinate in the reference element
- `η::T`: Second coordinate in the reference element

# Returns
- `SVector{(Deg+1)²,T}`: Values of all basis functions

# Node Layout (Deg = 1)
```
3 --- 4
|     |
1 --- 2
```

# Node Layout (Deg = 2)
```
7 --- 8 --- 9
|     |     |
4 --- 5 --- 6
|     |     |
1 --- 2 --- 3
```
"""
@inline function basis_functions(::Lagrange{Deg, 2}, ξ::T, η::T) where {Deg, T <: Real}
    φξ = basis_functions(Lagrange{Deg, 1}(), ξ)
    φη = basis_functions(Lagrange{Deg, 1}(), η)

    N = Deg + 1
    num_local_dof = N * N

    return SVector{num_local_dof, T}(ntuple(k -> begin
            i = mod(k - 1, N) + 1
            j = div(k - 1, N) + 1
            φξ[i] * φη[j]
        end, num_local_dof))
end

"""
    basis_functions_derivatives(::Lagrange{Deg, 2}, ξ, η)

Derivatives of tensor-product Lagrange basis functions.

# Arguments
- `ξ::T`: First coordinate in the reference element
- `η::T`: Second coordinate in the reference element

# Returns
- Tuple `(∂ϕ/∂ξ, ∂ϕ/∂η)` where:
  - `∂ϕ/∂ξ::SVector{(Deg+1)²,T}`: Derivatives ∂ϕᵢ/∂ξ
  - `∂ϕ/∂η::SVector{(Deg+1)²,T}`: Derivatives ∂ϕᵢ/∂η
"""
@inline function basis_functions_derivatives(
        ::Lagrange{Deg, 2}, ξ::T, η::T) where {Deg, T <: Real}
    ϕξ = basis_functions(Lagrange{Deg, 1}(), ξ)
    ϕη = basis_functions(Lagrange{Deg, 1}(), η)
    dϕξ = basis_functions_derivatives(Lagrange{Deg, 1}(), ξ)
    dϕη = basis_functions_derivatives(Lagrange{Deg, 1}(), η)

    N = Deg + 1
    num_local_dof = N * N

    ∂ϕ_∂ξ = SVector{num_local_dof, T}(ntuple(k -> begin
            i = mod(k - 1, N) + 1
            j = div(k - 1, N) + 1
            dϕξ[i] * ϕη[j]
        end, num_local_dof))

    ∂ϕ_∂η = SVector{num_local_dof, T}(ntuple(k -> begin
            i = mod(k - 1, N) + 1
            j = div(k - 1, N) + 1
            ϕξ[i] * dϕη[j]
        end, num_local_dof))

    return ∂ϕ_∂ξ, ∂ϕ_∂η
end

# ============================================================================
# HERMITE 1D - Reference element [-1, 1]
# ============================================================================

"""
    basis_functions(::Hermite{3, 1}, ξ)

Cubic Hermite basis functions on the reference element [-1, 1].

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{4,T}`: Values [H₁(ξ), H₁'(ξ), H₂(ξ), H₂'(ξ)]

# Node Layout
```
1:2 ---- 3:4
```
Each node has 2 DOFs: (u, du/dξ)
"""
@inline function basis_functions(::Hermite{3, 1}, ξ::T) where {T <: Real}
    cst = T(0.25)
    SVector(
        (2 + ξ) * (1 - ξ)^2 * cst,
        (ξ + 1) * (1 - ξ)^2 * cst,
        (2 - ξ) * (1 + ξ)^2 * cst,
        (ξ - 1) * (1 + ξ)^2 * cst
    )
end

"""
    basis_functions_derivatives(::Hermite{3, 1}, ξ)

Derivatives of cubic Hermite basis functions with respect to ξ.

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{4,T}`: Derivatives [dH₁/dξ, dH₁'/dξ, dH₂/dξ, dH₂'/dξ]
"""
@inline function basis_functions_derivatives(::Hermite{3, 1}, ξ::T) where {T <: Real}
    cst = T(0.25)
    SVector(
        3 * (-1 + ξ) * (1 + ξ) * cst,
        (ξ - 1) * (1 + 3 * ξ) * cst,
        3 * (1 - ξ) * (1 + ξ) * cst,
        (ξ + 1) * (-1 + 3 * ξ) * cst
    )
end

"""
    basis_functions_second_derivatives(::Hermite{3, 1}, ξ)

Second derivatives of cubic Hermite basis functions with respect to ξ.

# Arguments
- `ξ::T`: Coordinate in the reference element [-1, 1]

# Returns
- `SVector{4,T}`: Second derivatives [d²H₁/dξ², d²H₁'/dξ², d²H₂/dξ², d²H₂'/dξ²]
"""
@inline function basis_functions_second_derivatives(::Hermite{3, 1}, ξ::T) where {T <: Real}
    c1 = T(1.5)
    c2 = T(0.5)
    SVector(
        c1 * ξ,
        c1 * ξ - c2,
        -c1 * ξ,
        c1 * ξ + c2
    )
end

# ============================================================================
# HERMITE 2D - Reference element [-1,1] × [-1,1]
# ============================================================================

"""
    basis_functions(::Hermite{3, 2}, ξ, η)

Bicubic Hermite basis functions on the reference element [-1,1] × [-1,1].

# Arguments
- `ξ::T`: First coordinate in the reference element
- `η::T`: Second coordinate in the reference element

# Returns
- `SVector{16,T}`: Values of all 16 basis functions

# Node Layout
```
 9:12 ---- 13:16
  |          |
  |          |
 1:4  ----  5:8
```
Each corner node has 4 DOFs: (u, ∂u/∂ξ, ∂u/∂η, ∂²u/∂ξ∂η)
"""
@inline function basis_functions(::Hermite{3, 2}, ξ::T, η::T) where {T <: Real}
    ϕξ = basis_functions(Hermite{3, 1}(), ξ) # [H₁(ξ), H₁'(ξ), H₂(ξ), H₂'(ξ)]
    ϕη = basis_functions(Hermite{3, 1}(), η) # [H₁(η), H₁'(η), H₂(η), H₂'(η)]

    ϕ = SVector{16}(
        # NODE 1: (ξ,η) = (-1,-1)
        ϕξ[1] * ϕη[1],  # u:        H₁(ξ)·H₁(η)    → value = 1 at (-1,-1)
        ϕξ[2] * ϕη[1],  # ∂u/∂ξ:    H₁'(ξ)·H₁(η)   → ∂/∂ξ = 1 at (-1,-1)
        ϕξ[1] * ϕη[2],  # ∂u/∂η:    H₁(ξ)·H₁'(η)   → ∂/∂η = 1 at (-1,-1)
        ϕξ[2] * ϕη[2],  # ∂²u/∂ξ∂η: H₁'(ξ)·H₁'(η)  → ∂²/∂ξ∂η = 1 at (-1,-1)

        # NODE 2: (ξ,η) = (+1,-1)
        ϕξ[3] * ϕη[1],  # u:        H₂(ξ)·H₁(η)    → value = 1 at (+1,-1)
        ϕξ[4] * ϕη[1],  # ∂u/∂ξ:    H₂'(ξ)·H₁(η)   → ∂/∂ξ = 1 at (+1,-1)
        ϕξ[3] * ϕη[2],  # ∂u/∂η:    H₂(ξ)·H₁'(η)   → ∂/∂η = 1 at (+1,-1)
        ϕξ[4] * ϕη[2],  # ∂²u/∂ξ∂η: H₂'(ξ)·H₁'(η)  → ∂²/∂ξ∂η = 1 at (+1,-1)

        # NODE 3: (ξ,η) = (-1,+1)
        ϕξ[1] * ϕη[3],  # u:        H₁(ξ)·H₂(η)    → value = 1 at (-1,+1)
        ϕξ[2] * ϕη[3],  # ∂u/∂ξ:    H₁'(ξ)·H₂(η)   → ∂/∂ξ = 1 at (-1,+1)
        ϕξ[1] * ϕη[4],  # ∂u/∂η:    H₁(ξ)·H₂'(η)   → ∂/∂η = 1 at (-1,+1)
        ϕξ[2] * ϕη[4],  # ∂²u/∂ξ∂η: H₁'(ξ)·H₂'(η)  → ∂²/∂ξ∂η = 1 at (-1,+1)

        # NODE 4: (ξ,η) = (+1,+1)
        ϕξ[3] * ϕη[3],  # u:        H₂(ξ)·H₂(η)    → value = 1 at (+1,+1)
        ϕξ[4] * ϕη[3],  # ∂u/∂ξ:    H₂'(ξ)·H₂(η)   → ∂/∂ξ = 1 at (+1,+1)
        ϕξ[3] * ϕη[4],  # ∂u/∂η:    H₂(ξ)·H₂'(η)   → ∂/∂η = 1 at (+1,+1)
        ϕξ[4] * ϕη[4]   # ∂²u/∂ξ∂η: H₂'(ξ)·H₂'(η)  → ∂²/∂ξ∂η = 1 at (+1,+1)
    )

    return ϕ
end

"""
    basis_functions_derivatives(::Hermite{3, 2}, ξ, η)

Derivatives of Bicubic Hermite basis functions.

# Arguments
- `ξ::T`: First coordinate in the reference element
- `η::T`: Second coordinate in the reference element

# Returns
- Tuple `(∂ϕ/∂ξ, ∂ϕ/∂η)` where:
  - `∂ϕ/∂ξ::SVector{16,T}`: Derivatives ∂ϕᵢ/∂ξ
  - `∂ϕ/∂η::SVector{16,T}`: Derivatives ∂ϕᵢ/∂η
"""
@inline function basis_functions_derivatives(::Hermite{3, 2}, ξ::T, η::T) where {T <: Real}
    ϕξ = basis_functions(Hermite{3, 1}(), ξ)
    ϕη = basis_functions(Hermite{3, 1}(), η)

    dϕξ = basis_functions_derivatives(Hermite{3, 1}(), ξ)
    dϕη = basis_functions_derivatives(Hermite{3, 1}(), η)

    ∂ϕ_∂ξ = SVector{16}(
        # NODE 1: (ξ,η) = (-1,-1)
        dϕξ[1] * ϕη[1],
        dϕξ[2] * ϕη[1],
        dϕξ[1] * ϕη[2],
        dϕξ[2] * ϕη[2],

        # NODE 2: (ξ,η) = (+1,-1)
        dϕξ[3] * ϕη[1],
        dϕξ[4] * ϕη[1],
        dϕξ[3] * ϕη[2],
        dϕξ[4] * ϕη[2],

        # NODE 3: (ξ,η) = (-1,+1)
        dϕξ[1] * ϕη[3],
        dϕξ[2] * ϕη[3],
        dϕξ[1] * ϕη[4],
        dϕξ[2] * ϕη[4],

        # NODE 4: (ξ,η) = (+1,+1)
        dϕξ[3] * ϕη[3],
        dϕξ[4] * ϕη[3],
        dϕξ[3] * ϕη[4],
        dϕξ[4] * ϕη[4]
    )

    ∂ϕ_∂η = SVector{16}(
        # NODE 1: (ξ,η) = (-1,-1)
        ϕξ[1] * dϕη[1],
        ϕξ[2] * dϕη[1],
        ϕξ[1] * dϕη[2],
        ϕξ[2] * dϕη[2],

        # NODE 2: (ξ,η) = (+1,-1)
        ϕξ[3] * dϕη[1],
        ϕξ[4] * dϕη[1],
        ϕξ[3] * dϕη[2],
        ϕξ[4] * dϕη[2],

        # NODE 3: (ξ,η) = (-1,+1)
        ϕξ[1] * dϕη[3],
        ϕξ[2] * dϕη[3],
        ϕξ[1] * dϕη[4],
        ϕξ[2] * dϕη[4],

        # NODE 4: (ξ,η) = (+1,+1)
        ϕξ[3] * dϕη[3],
        ϕξ[4] * dϕη[3],
        ϕξ[3] * dϕη[4],
        ϕξ[4] * dϕη[4]
    )

    return ∂ϕ_∂ξ, ∂ϕ_∂η
end