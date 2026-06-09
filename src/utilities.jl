# ==============================================================================
# build_upper_to_full_maps
# ==============================================================================
"""
    build_upper_to_full_maps11(M, M₁₁upper) -> (Vector{I}, Vector{I})

Precompute positional index maps from the upper-triangular nonzeros of `M₁₁upper.data` into `M.nzval`, 
enabling efficient in-place assembly via [`scatter_symmetric!`](@ref).

The suffix `11` denotes the `(1,1)` block of the matrix M.
`M₁₁upper` wraps the upper triangle of the symmetric submatrix `M[1:m, 1:m]` (m ≤ size(M, 1)); 
`M` must carry the full symmetric sparsity pattern of that block in both triangles.

# Arguments
- `M::SparseMatrixCSC{T,I}`: global matrix whose sparsity pattern covers both triangles of `M[1:m, 1:m]`.
- `M₁₁upper::Symmetric{T,SparseMatrixCSC{T,I}}`: upper-triangular storage of the `(1,1)` submatrix.

# Returns
- `direct::Vector{I}`: `direct[k]` is the index in `M.nzval` of entry `M[i,j]`, where `(i,j)` is the `k`-th nonzero of `M₁₁upper.data`.
- `mirror::Vector{I}`: `mirror[k]` is the index in `M.nzval` of the transposed entry `M[j,i]`.  Equals `direct[k]` on the diagonal.
"""
function build_upper_to_full_maps11(
        M::SparseMatrixCSC{T, I},
        M₁₁upper::Symmetric{T, SparseMatrixCSC{T, I}}
) where {T <: AbstractFloat, I <: Integer}
    U = M₁₁upper.data
    nnz_U = nnz(U)
    direct = Vector{I}(undef, nnz_U)
    mirror = Vector{I}(undef, nnz_U)
    for j in 1:size(U, 2)            # Iterate over columns of U
        for kU in nzrange(U, j)      # Iterate over nonzeros of U[:,j]
            i = U.rowval[kU]
            for kM in nzrange(M, j)  # Iterate over nonzeros of M[:,j] and find position of M[i,j]
                if M.rowval[kM] == i
                    direct[kU] = kM
                    break
                end
            end
            for kᵀM in nzrange(M, i) # Iterate over nonzeros of M[:,i] and find position of M[j,i]
                if M.rowval[kᵀM] == j
                    mirror[kU] = kᵀM
                    break
                end
            end
        end
    end
    return direct, mirror
end

# ==============================================================================
# scatter_symmetric!
# ==============================================================================
"""
    scatter_symmetric!(M, M_upper, direct, mirror) -> nothing

Scatter the nonzeros of `M_upper.data` into both triangles of `M.nzval` in-place.
For each `k`, assigns `M.nzval[direct[k]] = M.nzval[mirror[k]] = M_upper.data.nzval[k]`.

# Arguments
- `M::SparseMatrixCSC{T,I}`: target matrix updated in-place.
- `M_upper::Symmetric{T,SparseMatrixCSC{T,I}}`: source upper-triangular storage.
- `direct::Vector{I}`: index map for entry `M[i,j]`.
- `mirror::Vector{I}`: index map for the transposed entry `M[j,i]`.
"""
function scatter_symmetric!(
        M::SparseMatrixCSC{T, I},
        M_upper::Symmetric{T, SparseMatrixCSC{T, I}},
        direct::Vector{I},
        mirror::Vector{I}
) where {T <: AbstractFloat, I <: Integer}
    nzval_U = M_upper.data.nzval
    nzval_M = M.nzval
    @inbounds for k in eachindex(nzval_U)
        nzval_M[direct[k]] = nzval_U[k]
        nzval_M[mirror[k]] = nzval_U[k]
    end
    return nothing
end