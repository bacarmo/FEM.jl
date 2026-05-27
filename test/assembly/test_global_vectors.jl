@testitem "assembly_rhs_1d!: Lagrange{1,1}(), AllSides(), ∫f(x)*ϕᵢ(x) dx" begin
    using FEM: assembly_rhs_1d!, Lagrange, AllSides, DOFMap, basis_functions
    using StaticArrays: SVector
    using GaussQuadrature: legendre

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    Δx = 1 / nel_per_dim[1]
    dof_map = DOFMap(fe, bc, nel_per_dim)
    m = dof_map.m

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P, W = SVector{Npg}(P_raw), SVector{Npg}(W_raw)
    xP = (Δx / 2) .* (P .+ 1)

    ϕP = SVector{Npg}(basis_functions(fe, P[i]) for i in 1:Npg)
    W_ϕP = SVector{Npg}(W[i] * ϕP[i] for i in 1:Npg)

    # Tests 
    F = zeros(m)
    scale = Δx / 2
    @testset "f(x) = 1.0" begin
        assembly_rhs_1d!(F, x -> 1.0, scale, W_ϕP, dof_map, Δx, xP)
        @test F ≈ fill(1 / 4, m)
    end

    @testset "f(x) = x" begin
        assembly_rhs_1d!(F, x -> x, scale, W_ϕP, dof_map, Δx, xP)
        F_expected = [1 / 16, 1 / 8, 3 / 16]
        @test F ≈ F_expected
    end

    @testset "f(x) = sin(x)" begin
        alloc = @allocated assembly_rhs_1d!(F, sin, scale, W_ϕP, dof_map, Δx, xP)
        F_expected = [
            8 * sin(1 / 4) - 4 * sin(1 / 2),
            -4 * sin(1 / 4) + 8 * sin(1 / 2) - 4 * sin(3 / 4),
            -4 * sin(1 / 2) + 8 * sin(3 / 4) - 4 * sin(1)
        ]
        @test F ≈ F_expected
        @test alloc == 0
    end
end

@testitem "assembly_rhs_1d!: Lagrange{1,1}(), AllSides(), ∫f(x)*dϕᵢ(x) dx" begin
    using FEM: assembly_rhs_1d!, Lagrange, AllSides, DOFMap, basis_functions_derivatives
    using StaticArrays: SVector
    using GaussQuadrature: legendre

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    Δx = 1 / nel_per_dim[1]
    dof_map = DOFMap(fe, bc, nel_per_dim)
    m = dof_map.m

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P, W = SVector{Npg}(P_raw), SVector{Npg}(W_raw)
    xP = (Δx / 2) .* (P .+ 1)

    dϕP = SVector{Npg}(basis_functions_derivatives(fe, P[i]) for i in 1:Npg)
    W_dϕP = SVector{Npg}(W[i] * dϕP[i] for i in 1:Npg)

    # Tests 
    F = zeros(m)
    scale = 1.0  # == (Δx/2) * (2/Δx)
    @testset "f(x) = 1.0" begin
        assembly_rhs_1d!(F, x -> 1.0, scale, W_dϕP, dof_map, Δx, xP)
        F_expected = fill(0.0, m)
        @test F≈F_expected atol=1e-15
    end

    @testset "f(x) = x" begin
        assembly_rhs_1d!(F, x -> x, scale, W_dϕP, dof_map, Δx, xP)
        F_expected = fill(-1 / 4, m)
        @test F ≈ F_expected
    end

    @testset "f(x) = sin(x)" begin
        alloc = @allocated assembly_rhs_1d!(F, sin, scale, W_dϕP, dof_map, Δx, xP)
        F_expected = [
            8 * sin(1 / 8)^2 - 4 * cos(1 / 4) + 4 * cos(1 / 2),
            4 * cos(1 / 4) - 8 * cos(1 / 2) + 4 * cos(3 / 4),
            4 * cos(1 / 2) - 8 * cos(3 / 4) + 4 * cos(1)
        ]
        @test F ≈ F_expected
        @test alloc == 0
    end
end

@testitem "assembly_rhs_2d!: Lagrange{1,2}(), AllSides(), ∫f(x,y)*φᵢ(x,y) dx dy" begin
    using FEM: assembly_rhs_2d!, Lagrange, AllSides, DOFMap, basis_functions
    using StaticArrays: SVector, SMatrix
    using GaussQuadrature: legendre

    # Setup
    fe = Lagrange{1, 2}()
    bc = AllSides()
    nel_per_dim = (4, 3)
    element_side_lengths = 1 ./ nel_per_dim
    Δx, Δy = element_side_lengths
    dof_map = DOFMap(fe, bc, nel_per_dim)
    m = dof_map.m

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P, W = SVector{Npg}(P_raw), SVector{Npg}(W_raw)
    xP = (Δx / 2) .* (P .+ 1)
    yP = (Δy / 2) .* (P .+ 1)

    φP = SMatrix{Npg, Npg}([basis_functions(fe, P[i], P[j]) for i in 1:Npg, j in 1:Npg])
    W_φP = SMatrix{Npg, Npg}([W[i] * W[j] * φP[i, j] for i in 1:Npg, j in 1:Npg])

    # Tests
    F = zeros(m)
    scale = Δx * Δy / 4

    @testset "f(x,y) = 1.0" begin
        f(x, y) = 1.0
        alloc = @allocated assembly_rhs_2d!(
            F, f, scale, W_φP, dof_map, nel_per_dim, element_side_lengths, xP, yP)
        F_expected = fill(Δx * Δy, m)
        @test F ≈ F_expected
        @test alloc == 0
    end

    @testset "f(x,y) = x * (x - 1) * y * (y - 1)" begin
        f(x, y) = x * (x - 1) * y * (y - 1)
        alloc = @allocated assembly_rhs_2d!(
            F, f, scale, W_φP, dof_map, nel_per_dim, element_side_lengths, xP, yP)
        cst1 = 13 / 27648 + 7 / 9216 + 169 / 248_832 + 91 / 82_944
        cst2 = 2 * 23 / 27_648 + 2 * 299 / 248_832
        @test F[1] ≈ F[3] ≈ F[4] ≈ F[6] ≈ cst1
        @test F[2] ≈ F[5] ≈ cst2
        @test alloc == 0
    end
end

@testitem "assembly_nonlinearity_F!: Lagrange{1,2}(), LeftRightTop(), f(u) = u, d = ones(m)" begin
    using FEM: assembly_nonlinearity_F!, Lagrange, LeftRightTop, DOFMap,
               assembly_global_matrix, assembly_local_matrix_ϕxϕ, basis_functions
    using GaussQuadrature: legendre
    using StaticArrays: SMatrix

    # Setup
    fe = Lagrange{1, 2}()
    bc = LeftRightTop()
    nel_per_dim = (4, 3)
    element_side_lengths = 1 ./ nel_per_dim
    Δx, Δy = element_side_lengths
    dof_map = DOFMap(fe, bc, nel_per_dim)

    f(u) = u
    scale = 1.0
    d = ones(dof_map.m)
    F = zeros(dof_map.m)

    Npg = 2
    P, W = legendre(Npg)
    φP = SMatrix{Npg, Npg}([basis_functions(fe, P[i], P[j]) for i in 1:Npg, j in 1:Npg])
    W_φP = SMatrix{Npg, Npg}([W[i] * W[j] * φP[i, j] for i in 1:Npg, j in 1:Npg])

    # Compute
    alloc = @allocated assembly_nonlinearity_F!(
        F, scale, f, d, dof_map, element_side_lengths, φP, W_φP)

    # Expected solution
    # If f(u) = u and d = ones, F = M·d where M is mass matrix
    Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)
    F_expected = assembly_global_matrix(Me, dof_map) * d

    # Test
    @test F ≈ F_expected
    @test alloc == 0
end

@testitem "assembly_nonlinearity_G!: Lagrange{1,1}(), AllSides(), g(x,v) = v, v = ones(m)" begin
    using FEM: assembly_nonlinearity_G!, Lagrange, AllSides, DOFMap, assembly_global_matrix,
               assembly_local_matrix_ϕxϕ, basis_functions
    using StaticArrays: SVector
    using GaussQuadrature: legendre

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    element_side_lengths = 1 ./ nel_per_dim
    Δx = element_side_lengths[1]
    dof_map = DOFMap(fe, bc, nel_per_dim)

    g(x, v) = v
    scale = 1.0
    v = ones(dof_map.m)
    G = zeros(dof_map.m)

    Npg = 2
    P_raw, W_raw = legendre(Npg)
    P, W = SVector{Npg}(P_raw), SVector{Npg}(W_raw)
    xP = (Δx / 2) .* (P .+ 1)
    ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])
    W_ϕP = SVector{Npg}([W[i] * ϕP[i] for i in 1:Npg])

    # Compute
    alloc = @allocated assembly_nonlinearity_G!(G, scale, g, v, dof_map, Δx, xP, ϕP, W_ϕP)

    # Expected solution
    # If g(x, v) = v and v = ones, G = M·v where M is mass matrix
    Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)
    G_expected = assembly_global_matrix(Me, dof_map) * v

    # Test
    @test G ≈ G_expected
    @test alloc == 0
end