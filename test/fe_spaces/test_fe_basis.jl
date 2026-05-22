# =============================================================================
# 1D Lagrange, Degree 1
# =============================================================================

@testitem "1D Lagrange, Degree 1: functions" begin
    using FEM: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    @test basis_functions(Lagrange{1, 1}(), -1.0) ≈ [1.0, 0.0]
    @test basis_functions(Lagrange{1, 1}(), 1.0) ≈ [0.0, 1.0]

    # Partition of unity: Σϕᵢ = 1
    @test sum(basis_functions(Lagrange{1, 1}(), -0.5)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 1}(), 0.0)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 1}(), 0.7)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{1, 1}(), 0.0)
    vec32 = basis_functions(Lagrange{1, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "1D Lagrange, Degree 1: derivatives" begin
    using FEM: Lagrange, basis_functions_derivatives

    # Linear basis: derivatives are constant
    @test basis_functions_derivatives(Lagrange{1, 1}(), -1.0) ≈ [-0.5, 0.5]
    @test basis_functions_derivatives(Lagrange{1, 1}(), 1.0) ≈ [-0.5, 0.5]

    # Test type
    vec64 = basis_functions_derivatives(Lagrange{1, 1}(), 0.0)
    vec32 = basis_functions_derivatives(Lagrange{1, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 1D Lagrange, Degree 2
# =============================================================================

@testitem "1D Lagrange, Degree 2: functions" begin
    using FEM: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    @test basis_functions(Lagrange{2, 1}(), -1.0) ≈ [1.0, 0.0, 0.0]
    @test basis_functions(Lagrange{2, 1}(), 0.0) ≈ [0.0, 1.0, 0.0]
    @test basis_functions(Lagrange{2, 1}(), 1.0) ≈ [0.0, 0.0, 1.0]

    # Partition of unity
    @test sum(basis_functions(Lagrange{2, 1}(), -0.7)) ≈ 1.0
    @test sum(basis_functions(Lagrange{2, 1}(), 0.3)) ≈ 1.0
    @test sum(basis_functions(Lagrange{2, 1}(), 0.9)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{2, 1}(), 0.0)
    vec32 = basis_functions(Lagrange{2, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "1D Lagrange, Degree 2: derivatives" begin
    using FEM: Lagrange, basis_functions_derivatives

    # Node values
    @test basis_functions_derivatives(Lagrange{2, 1}(), -1.0) ≈ [-1.5, 2.0, -0.5]
    @test basis_functions_derivatives(Lagrange{2, 1}(), 0.0) ≈ [-0.5, 0.0, 0.5]
    @test basis_functions_derivatives(Lagrange{2, 1}(), 1.0) ≈ [0.5, -2.0, 1.5]

    # Test type
    vec64 = basis_functions_derivatives(Lagrange{2, 1}(), 0.0)
    vec32 = basis_functions_derivatives(Lagrange{2, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 1D Lagrange, Degree 3
# =============================================================================

@testitem "1D Lagrange, Degree 3: functions" begin
    using FEM: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    @test basis_functions(Lagrange{3, 1}(), -1.0) ≈ [1.0, 0.0, 0.0, 0.0]
    @test basis_functions(Lagrange{3, 1}(), -1.0 / 3.0) ≈ [0.0, 1.0, 0.0, 0.0]
    @test basis_functions(Lagrange{3, 1}(), 1.0 / 3.0) ≈ [0.0, 0.0, 1.0, 0.0]
    @test basis_functions(Lagrange{3, 1}(), 1.0) ≈ [0.0, 0.0, 0.0, 1.0]

    # Partition of unity
    @test sum(basis_functions(Lagrange{3, 1}(), -0.8)) ≈ 1.0
    @test sum(basis_functions(Lagrange{3, 1}(), 0.0)) ≈ 1.0
    @test sum(basis_functions(Lagrange{3, 1}(), 0.6)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{3, 1}(), 0.0)
    vec32 = basis_functions(Lagrange{3, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "1D Lagrange, Degree 3: derivatives" begin
    using FEM: Lagrange, basis_functions_derivatives

    # Node values
    @test basis_functions_derivatives(Lagrange{3, 1}(), -1.0) ≈
          [-11 / 4, 9 / 2, -9 / 4, 1 / 2]
    @test basis_functions_derivatives(Lagrange{3, 1}(), -1.0 / 3.0) ≈
          [-1 / 2, -3 / 4, 3 / 2, -1 / 4]
    @test basis_functions_derivatives(Lagrange{3, 1}(), 1.0 / 3.0) ≈
          [1 / 4, -3 / 2, 3 / 4, 1 / 2]
    @test basis_functions_derivatives(Lagrange{3, 1}(), 1.0) ≈
          [-1 / 2, 9 / 4, -9 / 2, 11 / 4]

    # Test type
    vec64 = basis_functions_derivatives(Lagrange{3, 1}(), 0.0)
    vec32 = basis_functions_derivatives(Lagrange{3, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 2D Lagrange, Degree 1
# =============================================================================

@testitem "2D Lagrange, Degree 1: functions" begin
    using FEM: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    # 1:(−1,−1), 2:(+1,−1), 3:(−1,+1), 4:(+1,+1)
    @test basis_functions(Lagrange{1, 2}(), -1.0, -1.0) ≈ [1.0, 0.0, 0.0, 0.0]
    @test basis_functions(Lagrange{1, 2}(), 1.0, -1.0) ≈ [0.0, 1.0, 0.0, 0.0]
    @test basis_functions(Lagrange{1, 2}(), -1.0, 1.0) ≈ [0.0, 0.0, 1.0, 0.0]
    @test basis_functions(Lagrange{1, 2}(), 1.0, 1.0) ≈ [0.0, 0.0, 0.0, 1.0]

    # Partition of unity
    @test sum(basis_functions(Lagrange{1, 2}(), -0.3, -0.7)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 2}(), 0.0, 0.0)) ≈ 1.0
    @test sum(basis_functions(Lagrange{1, 2}(), 0.5, 0.8)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{1, 2}(), 0.0, 0.0)
    vec32 = basis_functions(Lagrange{1, 2}(), Float32(0.0), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "2D Lagrange, Degree 1: derivatives" begin
    using FEM: Lagrange, basis_functions_derivatives

    # Derivatives at center (0,0)
    ∂ϕ_∂ξ, ∂ϕ_∂η = basis_functions_derivatives(Lagrange{1, 2}(), 0.0, 0.0)
    @test ∂ϕ_∂ξ ≈ [-0.25, 0.25, -0.25, 0.25]
    @test ∂ϕ_∂η ≈ [-0.25, -0.25, 0.25, 0.25]

    # Derivatives at node 1: (−1,−1)
    ∂ϕ_∂ξ, ∂ϕ_∂η = basis_functions_derivatives(Lagrange{1, 2}(), -1.0, -1.0)
    @test ∂ϕ_∂ξ ≈ [-0.5, 0.5, 0.0, 0.0]
    @test ∂ϕ_∂η ≈ [-0.5, 0.0, 0.5, 0.0]

    # Derivatives at node 2: (+1,−1)
    ∂ϕ_∂ξ, ∂ϕ_∂η = basis_functions_derivatives(Lagrange{1, 2}(), +1.0, -1.0)
    @test ∂ϕ_∂ξ ≈ [-0.5, 0.5, 0.0, 0.0]
    @test ∂ϕ_∂η ≈ [0.0, -0.5, 0.0, 0.5]

    # Derivatives at node 3: (−1,+1)
    ∂ϕ_∂ξ, ∂ϕ_∂η = basis_functions_derivatives(Lagrange{1, 2}(), -1.0, +1.0)
    @test ∂ϕ_∂ξ ≈ [0.0, 0.0, -0.5, 0.5]
    @test ∂ϕ_∂η ≈ [-0.5, 0.0, 0.5, 0.0]

    # Derivatives at node 4: (+1,+1)
    ∂ϕ_∂ξ, ∂ϕ_∂η = basis_functions_derivatives(Lagrange{1, 2}(), +1.0, +1.0)
    @test ∂ϕ_∂ξ ≈ [0.0, 0.0, -0.5, 0.5]
    @test ∂ϕ_∂η ≈ [0.0, -0.5, 0.0, 0.5]

    # Test type
    vec64_ξ, vec64_η = basis_functions_derivatives(Lagrange{1, 2}(), 0.0, 0.0)
    vec32_ξ, vec32_η = basis_functions_derivatives(Lagrange{1, 2}(), 0.0f0, 0.0f0)
    @test eltype(vec64_ξ) == Float64
    @test eltype(vec64_η) == Float64
    @test eltype(vec32_ξ) == Float32
    @test eltype(vec32_η) == Float32
end

# =============================================================================
# 2D Lagrange, Degree 2
# =============================================================================

@testitem "2D Lagrange, Degree 2: functions" begin
    using FEM: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    #    7 --- 8 --- 9
    #    |     |     |
    #    4 --- 5 --- 6
    #    |     |     |
    #    1 --- 2 --- 3
    nodes = [
        (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),  # layer 1
        (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),     # layer 2
        (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0)      # layer 3
    ]
    for (i, (ξ, η)) in enumerate(nodes)
        result = basis_functions(Lagrange{2, 2}(), ξ, η)
        expected = zeros(9)
        expected[i] = 1.0
        @test result ≈ expected
    end

    # Partition of unity
    @test sum(basis_functions(Lagrange{2, 2}(), -0.5, -0.3)) ≈ 1.0
    @test sum(basis_functions(Lagrange{2, 2}(), 0.7, 0.4)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{2, 2}(), 0.0, 0.0)
    vec32 = basis_functions(Lagrange{2, 2}(), Float32(0.0), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 2D Lagrange, Degree 3
# =============================================================================
@testitem "2D Lagrange, Degree 3: functions" begin
    using FEM: Lagrange, basis_functions

    # Node values: ϕᵢ(node_j) = δᵢⱼ
    #   13 -- 14 -- 15 -- 16
    #   |     |     |     |
    #   9  -- 10 -- 11 -- 12
    #   |     |     |     |
    #   5  -- 6  -- 7  -- 8
    #   |     |     |     |
    #   1  -- 2  -- 3  -- 4
    coords = [-1.0, -1 / 3, 1 / 3, 1.0]
    nodes = [(coords[i], coords[j])
             for j in 1:4 for i in 1:4]
    for (i, (ξ, η)) in enumerate(nodes)
        result = basis_functions(Lagrange{3, 2}(), ξ, η)
        expected = zeros(16)
        expected[i] = 1.0
        @test result ≈ expected
    end

    # Partition of unity
    @test sum(basis_functions(Lagrange{3, 2}(), -0.5, -0.3)) ≈ 1.0
    @test sum(basis_functions(Lagrange{3, 2}(), 0.7, 0.4)) ≈ 1.0

    # Test type
    vec64 = basis_functions(Lagrange{3, 2}(), 0.0, 0.0)
    vec32 = basis_functions(Lagrange{3, 2}(), Float32(0.0), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

# =============================================================================
# 1D Hermite, Degree 3
# =============================================================================
@testitem "1D Hermite, Degree 3: functions" begin
    using FEM: Hermite, basis_functions

    # Node values: [H₁(ξ), H₁'(ξ), H₂(ξ), H₂'(ξ)]
    @test basis_functions(Hermite{3, 1}(), -1.0) ≈ [1.0, 0.0, 0.0, 0.0]
    @test basis_functions(Hermite{3, 1}(), 1.0) ≈ [0.0, 0.0, 1.0, 0.0]

    # Test type
    vec64 = basis_functions(Hermite{3, 1}(), 0.0)
    vec32 = basis_functions(Hermite{3, 1}(), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "1D Hermite, Degree 3: derivatives" begin
    using FEM: Hermite, basis_functions_derivatives

    # Node values: [dH₁/dξ, dH₁'/dξ, dH₂/dξ, dH₂'/dξ]
    @test basis_functions_derivatives(Hermite{3, 1}(), -1.0) ≈ [0.0, 1.0, 0.0, 0.0]
    @test basis_functions_derivatives(Hermite{3, 1}(), 1.0) ≈ [0.0, 0.0, 0.0, 1.0]

    # Test type
    vec64_ξ, vec64_η = basis_functions_derivatives(Hermite{3, 1}(), 0.0)
    vec32_ξ, vec32_η = basis_functions_derivatives(Hermite{3, 1}(), Float32(0.0))
    @test eltype(vec64_ξ) == Float64
    @test eltype(vec64_η) == Float64
    @test eltype(vec32_ξ) == Float32
    @test eltype(vec32_η) == Float32
end

# =============================================================================
# 2D Hermite, Degree 3
# =============================================================================
@testitem "2D Hermite, Degree 3: functions" begin
    using FEM: Hermite, basis_functions

    # Each node has 4 DOFs: (u, ∂u/∂ξ, ∂u/∂η, ∂²u/∂ξ∂η)
    # NODE 1: indices 1:4   at (ξ,η) = (-1,-1)
    # NODE 2: indices 5:8   at (ξ,η) = (+1,-1)
    # NODE 3: indices 9:12  at (ξ,η) = (-1,+1)
    # NODE 4: indices 13:16 at (ξ,η) = (+1,+1)

    # Test NODE 1 at (-1,-1): only first DOF (u) should be 1
    ϕ_node1 = basis_functions(Hermite{3, 2}(), -1.0, -1.0)
    @test ϕ_node1[1] ≈ 1.0
    @test ϕ_node1[2] ≈ 0.0
    @test ϕ_node1[3] ≈ 0.0
    @test ϕ_node1[4] ≈ 0.0
    @test all(ϕ_node1[5:16] .≈ 0.0)

    # Test NODE 2 at (+1,-1): only fifth DOF (u at node 2) should be 1
    ϕ_node2 = basis_functions(Hermite{3, 2}(), 1.0, -1.0)
    @test all(ϕ_node2[1:4] .≈ 0.0)
    @test ϕ_node2[5] ≈ 1.0
    @test ϕ_node2[6] ≈ 0.0
    @test ϕ_node2[7] ≈ 0.0
    @test ϕ_node2[8] ≈ 0.0
    @test all(ϕ_node2[9:16] .≈ 0.0)

    # Test NODE 3 at (-1,+1): only ninth DOF (u at node 3) should be 1
    ϕ_node3 = basis_functions(Hermite{3, 2}(), -1.0, 1.0)
    @test all(ϕ_node3[1:8] .≈ 0.0)
    @test ϕ_node3[9] ≈ 1.0
    @test ϕ_node3[10] ≈ 0.0
    @test ϕ_node3[11] ≈ 0.0
    @test ϕ_node3[12] ≈ 0.0
    @test all(ϕ_node3[13:16] .≈ 0.0)

    # Test NODE 4 at (+1,+1): only thirteenth DOF (u at node 4) should be 1
    ϕ_node4 = basis_functions(Hermite{3, 2}(), 1.0, 1.0)
    @test all(ϕ_node4[1:12] .≈ 0.0)
    @test ϕ_node4[13] ≈ 1.0
    @test ϕ_node4[14] ≈ 0.0
    @test ϕ_node4[15] ≈ 0.0
    @test ϕ_node4[16] ≈ 0.0

    # Test type preservation
    vec64 = basis_functions(Hermite{3, 2}(), 0.0, 0.0)
    vec32 = basis_functions(Hermite{3, 2}(), Float32(0.0), Float32(0.0))
    @test eltype(vec64) == Float64
    @test eltype(vec32) == Float32
end

@testitem "2D Hermite, Degree 3: derivatives" begin
    using FEM: Hermite, basis_functions_derivatives

    # Each node has 4 DOFs: (u, ∂u/∂ξ, ∂u/∂η, ∂²u/∂ξ∂η)
    # Test that derivatives satisfy Kronecker delta property

    # Test NODE 1 at (-1,-1): ∂/∂ξ of second DOF should be 1
    dϕ_dξ, dϕ_dη = basis_functions_derivatives(Hermite{3, 2}(), -1.0, -1.0)
    @test dϕ_dξ[1] ≈ 0.0
    @test dϕ_dξ[2] ≈ 1.0
    @test dϕ_dξ[3] ≈ 0.0
    @test dϕ_dξ[4] ≈ 0.0
    @test all(dϕ_dξ[5:16] .≈ 0.0)

    @test dϕ_dη[1] ≈ 0.0
    @test dϕ_dη[2] ≈ 0.0
    @test dϕ_dη[3] ≈ 1.0
    @test dϕ_dη[4] ≈ 0.0
    @test all(dϕ_dη[5:16] .≈ 0.0)

    # Test NODE 2 at (+1,-1): ∂/∂ξ of sixth DOF should be 1
    dϕ_dξ, dϕ_dη = basis_functions_derivatives(Hermite{3, 2}(), 1.0, -1.0)
    @test all(dϕ_dξ[1:5] .≈ 0.0)
    @test dϕ_dξ[6] ≈ 1.0
    @test dϕ_dξ[7] ≈ 0.0
    @test dϕ_dξ[8] ≈ 0.0
    @test all(dϕ_dξ[9:16] .≈ 0.0)

    @test all(dϕ_dη[1:6] .≈ 0.0)
    @test dϕ_dη[7] ≈ 1.0
    @test dϕ_dη[8] ≈ 0.0
    @test all(dϕ_dη[9:16] .≈ 0.0)

    # Test NODE 3 at (-1,+1): ∂/∂ξ of tenth DOF should be 1
    dϕ_dξ, dϕ_dη = basis_functions_derivatives(Hermite{3, 2}(), -1.0, 1.0)
    @test all(dϕ_dξ[1:9] .≈ 0.0)
    @test dϕ_dξ[10] ≈ 1.0
    @test dϕ_dξ[11] ≈ 0.0
    @test dϕ_dξ[12] ≈ 0.0
    @test all(dϕ_dξ[13:16] .≈ 0.0)

    @test all(dϕ_dη[1:10] .≈ 0.0)
    @test dϕ_dη[11] ≈ 1.0
    @test dϕ_dη[12] ≈ 0.0
    @test all(dϕ_dη[13:16] .≈ 0.0)

    # Test NODE 4 at (+1,+1): ∂/∂ξ of fourteenth DOF should be 1
    dϕ_dξ, dϕ_dη = basis_functions_derivatives(Hermite{3, 2}(), 1.0, 1.0)
    @test all(dϕ_dξ[1:13] .≈ 0.0)
    @test dϕ_dξ[14] ≈ 1.0
    @test dϕ_dξ[15] ≈ 0.0
    @test dϕ_dξ[16] ≈ 0.0

    @test all(dϕ_dη[1:14] .≈ 0.0)
    @test dϕ_dη[15] ≈ 1.0
    @test dϕ_dη[16] ≈ 0.0

    # Test type preservation
    dϕ_dξ_64, dϕ_dη_64 = basis_functions_derivatives(Hermite{3, 2}(), 0.0, 0.0)
    dϕ_dξ_32, dϕ_dη_32 = basis_functions_derivatives(Hermite{3, 2}(), 0.0f0, 0.0f0)
    @test eltype(dϕ_dξ_64) == Float64
    @test eltype(dϕ_dη_64) == Float64
    @test eltype(dϕ_dξ_32) == Float32
    @test eltype(dϕ_dη_32) == Float32
end