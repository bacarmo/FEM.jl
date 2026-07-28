"""
    assembly_global_matrix(local_matrix, dof_map) -> SparseMatrixCSC

Assemble global FEM matrix from element-local matrix.

# Arguments
- `local_matrix`: Element matrix (nb × nb)
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)

# Returns
- `SparseMatrixCSC{T,I}`: Assembled global matrix (m × m), excluding DOFs > m
"""
function assembly_global_matrix(
        local_matrix::SMatrix{nb, nb, T},
        dof_map::DOFMap{<:AbstractVector, I}
) where {T, I, nb}
    m = dof_map.m
    EQoLG = dof_map.EQoLG
    Ne = length(EQoLG)

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    capacity = Ne * nb * nb
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)

    idx = 0
    @inbounds for e in 1:Ne
        global_indices = EQoLG[e]
        for b in 1:nb
            jb = global_indices[b]
            jb > m && continue
            for a in 1:nb
                ia = global_indices[a]
                ia > m && continue

                idx += 1
                I_rows[idx] = ia
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b]
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    return sparse(I_rows, J_cols, V_vals, m, m)
end

"""
    assembly_global_matrix(local_matrix, dof_map) -> Symmetric{T, SparseMatrixCSC{T,I}}

Assemble global symmetric FEM matrix from element-local symmetric matrix.
Only stores upper triangle, reducing memory and assembly time.

# Arguments
- `local_matrix`: Symmetric element matrix (nb × nb)
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)

# Returns
- `Symmetric{T, SparseMatrixCSC{T,I}}`: Assembled symmetric global matrix (m × m)

# Assumptions
Assumes `a ≤ b ⇒ ia ≤ jb` for all local indices, where `ia = EQoLG[e][a]` and `jb = EQoLG[e][b]`. 
This condition holds for cartesian meshes with left-to-right, bottom-to-top numbering.
"""
function assembly_global_matrix(
        local_matrix::Symmetric{T, <:SMatrix{nb, nb, T}},
        dof_map::DOFMap{<:AbstractVector, I}
) where {T, I, nb}
    m = dof_map.m
    EQoLG = dof_map.EQoLG
    Ne = length(EQoLG)

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    entries_per_element = (nb * (nb + 1)) ÷ 2
    capacity = Ne * entries_per_element
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)

    idx = 0
    @inbounds for e in 1:Ne
        global_indices = EQoLG[e]
        for b in 1:nb
            jb = global_indices[b]
            jb > m && continue
            for a in 1:b  # Upper triangle: a ≤ b
                ia = global_indices[a]
                ia > m && continue

                idx += 1
                I_rows[idx] = ia  # Assuming a ≤ b ⇒ ia ≤ jb
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b]
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    K_upper = sparse(I_rows, J_cols, V_vals, m, m)
    return Symmetric(K_upper, :U)
end

"""
    assembly_global_matrix(local_matrix, dof_map_i, dof_map_j) -> SparseMatrixCSC

Assemble a global sparse FEM matrix from a single element-local matrix using two independent DOF maps. 
The assembly rule is:

    A[i, j] += Aᵉ[a, b],  i = dof_map_i.EQoLG[e][a],  j = dof_map_j.EQoLG[e][b]

Entries mapped to DOF indices exceeding `dof_map_i.m` or `dof_map_j.m` are discarded,
effectively enforcing Dirichlet boundary conditions.

# Arguments
- `local_matrix`: Element-local matrix of size nb × nb, shared across all elements.
- `dof_map_i`: Left/row DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `dof_map_j`: Right/column DOF mapping (`EQoLG` connectivity, `m` free DOFs)

# Returns
- `SparseMatrixCSC{T,I}`: Global assembled matrix of size `dof_map_i.m × dof_map_j.m`

# Throws
- `AssertionError` if `dof_map_i` and `dof_map_j` have different numbers of elements.
"""
function assembly_global_matrix(
        local_matrix::SMatrix{nb, nb, T},
        dof_map_i::DOFMap{<:AbstractVector, I},
        dof_map_j::DOFMap{<:AbstractVector, I}
) where {T, I, nb}
    mᵢ = dof_map_i.m
    mⱼ = dof_map_j.m
    Neᵢ = length(dof_map_i.EQoLG)
    Neⱼ = length(dof_map_j.EQoLG)
    @assert Neᵢ==Neⱼ "DOF maps must have the same number of elements (got $Neᵢ vs $Neⱼ)"
    Ne = Neᵢ

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    capacity = Ne * nb * nb
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)

    idx = 0
    @inbounds for e in 1:Ne
        global_indices_i = dof_map_i.EQoLG[e]
        global_indices_j = dof_map_j.EQoLG[e]
        for b in 1:nb
            jb = global_indices_j[b]
            jb > mⱼ && continue
            for a in 1:nb
                ia = global_indices_i[a]
                ia > mᵢ && continue

                idx += 1
                I_rows[idx] = ia
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b]
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    return sparse(I_rows, J_cols, V_vals, mᵢ, mⱼ)
end

"""
    assembly_global_matrix_DG(scale, ∂ₛg, v, dof_map, Δx, xP, ϕP, W_ϕPϕP)

DGᵢⱼ = scale * ∫ ϕᵢ(x) * ϕⱼ(x) * ∂ₛg(x, Vₕ(x)) dx over Ω ⊂ ℜ, with Vₕ(x) = Σ v[k] ϕₖ(x).

# Arguments
- `scale`: Scaling factor
- `∂ₛg`: Callable `(x, s) -> T`, the derivative of `g` with respect to `s`
- `v`: FE coefficient vector for Vₕ, length `dof_map.m`
- `dof_map`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `Δx`: Uniform element size
- `xP`: Precomputed fixed part of physical quadrature points; `xP = (Δx/2)*(P+1) + xmin`
- `ϕP`: Basis functions at each quadrature point; `ϕP[j][a] = ϕₐ(Pⱼ)`
- `W_ϕPϕP`: `W_ϕPϕP[j][a,b] = W[j] * ϕₐ(Pⱼ) * ϕᵦ(Pⱼ)`

# Returns
- `Symmetric{T, SparseMatrixCSC{T,I}}`: Assembled symmetric global matrix (m × m)
"""
function assembly_global_matrix_DG(
        scale::T,
        ∂ₛg::Fun,
        v::AbstractVector{T},
        dof_map::DOFMap{<:AbstractVector, I},
        Δx::T,
        xP::SVector{Npg, T},
        ϕP::SVector{Npg, SVector{nb, T}},
        W_ϕPϕP::SVector{Npg, <:SMatrix{nb, nb, T}}
) where {Fun, T, I, Npg, nb}
    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Ne = length(EQoLG)

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    entries_per_element = (nb * (nb + 1)) ÷ 2
    capacity = Ne * entries_per_element
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)
    local_matrix = zeros(T, nb, nb)

    scale_jacobian = scale * (Δx / 2)

    idx = 0
    for e in eachindex(EQoLG)
        eq = EQoLG[e]
        xeP = @. muladd(e - 1, Δx, xP)
        assembly_local_matrix_DG!(local_matrix, ∂ₛg, v, m, eq, xeP, ϕP, W_ϕPϕP)

        for b in 1:nb
            jb = eq[b]
            jb > m && continue

            for a in 1:b # Upper triangle: a ≤ b
                ia = eq[a]
                ia > m && continue

                idx += 1
                I_rows[idx] = ia  # Assuming a ≤ b ⇒ ia ≤ jb
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b] * scale_jacobian
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    DG_upper = sparse(I_rows, J_cols, V_vals, m, m)

    return Symmetric(DG_upper, :U)
end

"""
    assembly_global_matrix_DF(scale, f, d, dof_map, element_side_lengths, φP, W_φPφP)

DFᵢⱼ = scale * ∬ φᵢ(x,y) * φⱼ(x,y) * f(Uₕ(x,y)) dx dy over Ω, with Uₕ(x,y) = Σ d[k] φₖ(x,y).

# Arguments
- `scale`: Scaling factor
- `f`: Callable `s -> T`
- `d`: FE coefficient vector for Uₕ, length `dof_map.m`
- `dof_map::DOFMap`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension
- `φP`: Basis functions at each quadrature point; `φP[i,j][a] = φₐ(Pᵢ,Pⱼ)`
- `W_φPφP`: `W_φPφP[i,j][a,b] = Wᵢ⋅Wⱼ⋅φₐ(Pᵢ,Pⱼ)⋅φᵦ(Pᵢ,Pⱼ)`

# Returns
- `Symmetric{T, SparseMatrixCSC{T,I}}`: Assembled symmetric global matrix (m×m)
"""
function assembly_global_matrix_DF(
        scale::T,
        f::Fun,
        d::AbstractVector{T},
        dof_map::DOFMap{<:AbstractVector, I},
        element_side_lengths::NTuple{2, T},
        φP::SMatrix{Npg, Npg, SVector{nb, T}},
        W_φPφP::SMatrix{Npg, Npg, <:SMatrix{nb, nb, T}}
) where {Fun, T, I, Npg, nb}
    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Ne = length(EQoLG)
    Δx, Δy = element_side_lengths

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    entries_per_element = (nb * (nb + 1)) ÷ 2
    capacity = Ne * entries_per_element
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)
    local_matrix = zeros(T, nb, nb)

    scale_jacobian = scale * (Δx * Δy / 4)

    idx = 0
    for e in 1:Ne
        eq = EQoLG[e]
        assembly_local_matrix_DF!(local_matrix, f, d, m, eq, φP, W_φPφP)

        for b in 1:nb
            jb = eq[b]
            jb > m && continue

            for a in 1:b # Upper triangle: a ≤ b
                ia = eq[a]
                ia > m && continue

                idx += 1
                I_rows[idx] = ia  # Assuming a ≤ b ⇒ ia ≤ jb
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b] * scale_jacobian
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    DF_upper = sparse(I_rows, J_cols, V_vals, m, m)

    return Symmetric(DF_upper, :U)
end

"""
    assembly_global_matrix_DF(scale, f, d, dof_map, element_side_lengths, ϕP, W_ϕPϕP)

DFᵢⱼ = scale * ∫ ϕᵢ * ϕⱼ * f(Uₕ) dΩ, with Uₕ = Σ d[k] ϕₖ.

# Arguments
- `scale`: Scaling factor
- `f`: Callable `s -> T`
- `d`: FE coefficient vector for Uₕ, length `dof_map.m`
- `dof_map::DOFMap`: DOF mapping (`EQoLG` connectivity, `m` free DOFs)
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension
- `ϕP`: Basis functions at each quadrature point; `ϕP[i][a] = ϕₐ(Pᵢ)`
- `W_ϕPϕP`: `W_ϕPϕP[i][a,b] = Wᵢ⋅ϕₐ(Pᵢ)⋅ϕᵦ(Pᵢ)`

# Returns
- `Symmetric{T, SparseMatrixCSC{T,I}}`: Assembled symmetric global matrix (m×m)
"""
function assembly_global_matrix_DF(
        scale::T,
        f::Fun,
        d::AbstractVector{T},
        dof_map::DOFMap{<:AbstractVector, I},
        element_side_lengths::NTuple{1, T},
        ϕP::SVector{Npg, SVector{nb, T}},
        W_ϕPϕP::SVector{Npg, <:SMatrix{nb, nb, T}}
) where {Fun, T, I, Npg, nb}
    EQoLG = dof_map.EQoLG
    m = dof_map.m
    Ne = length(EQoLG)
    Δx = element_side_lengths[1]

    # Pre-allocate triplet arrays (row indices, column indices, values) for sparse assembly
    entries_per_element = (nb * (nb + 1)) ÷ 2
    capacity = Ne * entries_per_element
    I_rows = Vector{I}(undef, capacity)
    J_cols = Vector{I}(undef, capacity)
    V_vals = Vector{T}(undef, capacity)
    local_matrix = zeros(T, nb, nb)

    scale_jacobian = scale * (Δx / 2)

    idx = 0
    for e in 1:Ne
        eq = EQoLG[e]
        assembly_local_matrix_DF!(local_matrix, f, d, m, eq, ϕP, W_ϕPϕP)

        for b in 1:nb
            jb = eq[b]
            jb > m && continue

            for a in 1:b # Upper triangle: a ≤ b
                ia = eq[a]
                ia > m && continue

                idx += 1
                I_rows[idx] = ia  # Assuming a ≤ b ⇒ ia ≤ jb
                J_cols[idx] = jb
                V_vals[idx] = local_matrix[a, b] * scale_jacobian
            end
        end
    end

    # Trim to actual number of entries and assemble sparse matrix
    resize!(I_rows, idx)
    resize!(J_cols, idx)
    resize!(V_vals, idx)

    DF_upper = sparse(I_rows, J_cols, V_vals, m, m)

    return Symmetric(DF_upper, :U)
end