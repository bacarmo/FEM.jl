"""
    assembly_local_∫basis(fe, element_side_lengths)
 
Compute the local integral vector
```math
b_a^e = \\int_{Ωᵉ} \\varphi_a^e(x, y) \\, dΩ, \\quad a = 1, \\ldots, Nb2,
```
where ``\\{\\varphi_a^e\\}`` are the local basis functions of `fe` on a rectangular element `Ωᵉ`.
 
On a uniform Cartesian mesh all elements are congruent, so this vector is computed once and reused.
Returns an `SVector{nb, T}`, where `nb` is the number of local DOFs.
 
# Arguments
- `fe: FE basis with polynomial degree `Deg` and spatial dimension `2` (e.g., `Lagrange{1, 2}()`)
- `element_side_lengths`: Side lengths of the (axis-aligned) element along each spatial dimension.
"""
function assembly_local_∫basis end

function assembly_local_∫basis(
        ::Lagrange{1, 2}, element_side_lengths::NTuple{2, T}) where {T}
    Δx, Δy = element_side_lengths
    c = Δx * Δy / 4
    return c * SVector{4, T}(1, 1, 1, 1)
end

function assembly_local_∫basis(
        ::Lagrange{2, 2}, element_side_lengths::NTuple{2, T}) where {T}
    Δx, Δy = element_side_lengths
    c = Δx * Δy / 36 # == (Δx * Δy / 4) / 9
    return c * SVector{9, T}(1, 4, 1, 4, 16, 4, 1, 4, 1)
end

function assembly_local_∫basis(
        ::Lagrange{3, 2}, element_side_lengths::NTuple{2, T}) where {T}
    Δx, Δy = element_side_lengths
    c = Δx * Δy / 64 # == (Δx * Δy / 4) / 16
    return c * SVector{16, T}(
        1, 3, 3, 1,
        3, 9, 9, 3,
        3, 9, 9, 3,
        1, 3, 3, 1
    )
end