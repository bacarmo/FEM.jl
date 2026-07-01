@testitem "projection_H01: AllSides(), u=(x²-x)(y²-y), Lagrange{Deg, 2}()" begin
    using FEM: projection_H01, Lagrange, AllSides, DOFMap, assembly_local_matrix_∇ϕx∇ϕ,
               assembly_global_matrix
    using LinearAlgebra: Symmetric, cholesky

    # Setup
    bc = AllSides()
    nel_per_dim = (4, 4)
    pmin = (0.0, 0.0)
    pmax = (1.0, 1.0)
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    Δx, Δy = element_side_lengths

    u(x, y) = x * (x - 1) * y * (y - 1)
    ∂ₓu(x, y) = (2 * x - 1) * y * (y - 1)
    ∂ᵧu(x, y) = x * (x - 1) * (2 * y - 1)

    @testset "Deg = $Deg" for Deg in (2, 3)
        fe = Lagrange{Deg, 2}()
        dof_map = DOFMap(fe, bc, nel_per_dim)

        Ke = Symmetric(assembly_local_matrix_∇ϕx∇ϕ(fe, element_side_lengths))
        K = assembly_global_matrix(Ke, dof_map)
        factorized_lhs_matrix = cholesky(K)

        # Compute
        uₕ_coefs = projection_H01(
            ∂ₓu, ∂ᵧu, fe, nel_per_dim, pmin, pmax, dof_map, factorized_lhs_matrix)

        # Expected solution
        xs = (Δx / Deg):(Δx / Deg):(1 - Δx / Deg)
        ys = (Δy / Deg):(Δy / Deg):(1 - Δy / Deg)
        uₕ_coefs_expected = [u(x, y) for y in ys for x in xs]

        # Test 
        @test uₕ_coefs ≈ uₕ_coefs_expected
    end
end

@testitem "projection_H01!: AllSides(), u=(x²-x)(y²-y), Lagrange{Deg, 2}()" begin
    using FEM: projection_H01!, Lagrange, AllSides, DOFMap, assembly_local_matrix_∇ϕx∇ϕ,
               assembly_global_matrix, basis_functions_derivatives
    using LinearAlgebra: Symmetric, cholesky
    using StaticArrays: SVector, SMatrix
    using GaussQuadrature: legendre

    # Setup
    bc = AllSides()
    nel_per_dim = (4, 4)
    pmin = (0.0, 0.0)
    pmax = (1.0, 1.0)
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    Δx, Δy = element_side_lengths

    u(x, y) = x * (x - 1) * y * (y - 1)
    ∂ₓu(x, y) = (2 * x - 1) * y * (y - 1)
    ∂ᵧu(x, y) = x * (x - 1) * (2 * y - 1)

    @testset "Deg = $Deg" for Deg in (2, 3)
        fe = Lagrange{Deg, 2}()
        dof_map = DOFMap(fe, bc, nel_per_dim)

        Ke = Symmetric(assembly_local_matrix_∇ϕx∇ϕ(fe, element_side_lengths))
        K = assembly_global_matrix(Ke, dof_map)
        factorized_lhs_matrix = cholesky(K)

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

        # Compute
        uₕ_coefs = zeros(dof_map.m)
        vec = zeros(dof_map.m)
        projection_H01!(
            uₕ_coefs, ∂ₓu, ∂ᵧu,
            nel_per_dim, element_side_lengths,
            dof_map, factorized_lhs_matrix,
            xP, yP, W_∂φ∂ξP, W_∂φ∂ηP, vec)

        # Expected solution
        xs = (Δx / Deg):(Δx / Deg):(1 - Δx / Deg)
        ys = (Δy / Deg):(Δy / Deg):(1 - Δy / Deg)
        uₕ_coefs_expected = [u(x, y) for y in ys for x in xs]

        # Test 
        @test uₕ_coefs ≈ uₕ_coefs_expected
    end
end

@testitem "projection_L2: AllSides(), u=(x²-x), Lagrange{Deg, 1}()" begin
    using FEM: projection_L2, Lagrange, AllSides, DOFMap, assembly_local_matrix_ϕxϕ,
               assembly_global_matrix
    using LinearAlgebra: Symmetric, cholesky

    # Setup
    bc = AllSides()
    nel_per_dim = (4,)
    pmin = (0.0,)
    pmax = (1.0,)
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    Δx = element_side_lengths[1]

    u(x) = x * (x - 1)

    @testset "Deg = $Deg" for Deg in (2, 3)
        fe = Lagrange{Deg, 1}()
        dof_map = DOFMap(fe, bc, nel_per_dim)

        Me = Symmetric(assembly_local_matrix_ϕxϕ(fe, element_side_lengths))
        M = assembly_global_matrix(Me, dof_map)
        factorized_lhs_matrix = cholesky(M)

        # Compute
        uₕ_coefs = projection_L2(
            u, fe, nel_per_dim, pmin, pmax, dof_map, factorized_lhs_matrix)

        # Expected solution
        xs = (Δx / Deg):(Δx / Deg):(1 - Δx / Deg)
        uₕ_coefs_expected = [u(x) for x in xs]

        # Test 
        @test uₕ_coefs ≈ uₕ_coefs_expected
    end
end

@testitem "projection_L2!: AllSides(), u=(x²-x), Lagrange{Deg, 1}()" begin
    using FEM: projection_L2!, Lagrange, AllSides, DOFMap, assembly_local_matrix_ϕxϕ,
               assembly_global_matrix, basis_functions
    using LinearAlgebra: Symmetric, cholesky
    using StaticArrays: SVector
    using GaussQuadrature: legendre

    # Setup
    bc = AllSides()
    nel_per_dim = (4,)
    pmin = (0.0,)
    pmax = (1.0,)
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    Δx = element_side_lengths[1]

    u(x) = x * (x - 1)

    @testset "Deg = $Deg" for Deg in (2, 3)
        fe = Lagrange{Deg, 1}()
        dof_map = DOFMap(fe, bc, nel_per_dim)

        Me = Symmetric(assembly_local_matrix_ϕxϕ(fe, element_side_lengths))
        M = assembly_global_matrix(Me, dof_map)
        factorized_lhs_matrix = cholesky(M)

        Npg = 2 * (Deg + 1)
        P_raw, W_raw = legendre(Npg)
        P = SVector{Npg}(P_raw)
        W = SVector{Npg}(W_raw)

        xP = (Δx / 2) .* (P .+ 1) .+ pmin[1]
        ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])
        W_ϕP = SVector{Npg}([W[i] * ϕP[i] for i in 1:Npg])

        # Compute
        uₕ_coefs = zeros(dof_map.m)
        vec = zeros(dof_map.m)
        projection_L2!(
            uₕ_coefs, u, element_side_lengths, dof_map, factorized_lhs_matrix, xP, W_ϕP, vec)

        # Expected solution
        xs = (Δx / Deg):(Δx / Deg):(1 - Δx / Deg)
        uₕ_coefs_expected = [u(x) for x in xs]

        # Test 
        @test uₕ_coefs ≈ uₕ_coefs_expected
    end
end