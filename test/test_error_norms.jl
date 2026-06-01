@testitem "error_L2: AllSides(), Lagrange{1, 2}(), 2×2 grid, u=(x²-x)(y²-y)" begin
    using FEM: error_L2, Lagrange, AllSides, DOFMap, basis_functions
    using GaussQuadrature: legendre
    using StaticArrays: SVector, SMatrix

    # Setup
    fe = Lagrange{1, 2}()
    bc = AllSides()
    nel_per_dim = (2, 2)
    pmin = (0.0, 0.0)
    pmax = (1.0, 1.0)
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    dof_map = DOFMap(fe, bc, nel_per_dim)

    Δx, Δy = element_side_lengths

    u(x, y) = x * (x - 1) * y * (y - 1)
    uₕ_coefs = [u(Δx, Δy)]  # single interior DOF at (0.5, 0.5)

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1) .+ pmin[1]
    yP = (Δy / 2) .* (P .+ 1) .+ pmin[2]

    φP = SMatrix{Npg, Npg}([basis_functions(fe, P[i], P[j])
                            for i in 1:Npg, j in 1:Npg])

    # Compute
    L2_err = error_L2(
        u, uₕ_coefs, dof_map, nel_per_dim, element_side_lengths, W, xP, yP, φP)

    # Expected solution
    L2_err_expected = sqrt(4 * 29 / 614_400)

    # Test
    @test uₕ_coefs[1] ≈ 1 / 16
    @test L2_err ≈ L2_err_expected
    @test (@allocated error_L2(
        u, uₕ_coefs, dof_map, nel_per_dim, element_side_lengths, W, xP, yP, φP)) == 0
end

@testitem "error_L2: AllSides(), Lagrange{1, 2}(), 4×3 grid, u=(x²-x)(y²-y)" begin
    using FEM: error_L2, Lagrange, AllSides, DOFMap, basis_functions
    using GaussQuadrature: legendre
    using StaticArrays: SVector, SMatrix

    # Setup
    fe = Lagrange{1, 2}()
    bc = AllSides()
    nel_per_dim = (4, 3)
    pmin = (0.0, 0.0)
    pmax = (1.0, 1.0)
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    dof_map = DOFMap(fe, bc, nel_per_dim)

    Nx, Ny = nel_per_dim
    Δx, Δy = element_side_lengths

    u(x, y) = x * (x - 1) * y * (y - 1)
    uₕ_coefs = [u(i * Δx, j * Δy) for j in 1:(Ny - 1) for i in 1:(Nx - 1)]

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1) .+ pmin[1]
    yP = (Δy / 2) .* (P .+ 1) .+ pmin[2]

    φP = SMatrix{Npg, Npg}([basis_functions(fe, P[i], P[j])
                            for i in 1:Npg, j in 1:Npg])

    # Compute
    L2_err = error_L2(
        u, uₕ_coefs, dof_map, nel_per_dim, element_side_lengths, W, xP, yP, φP)

    # Expected solution
    L2_err_expected = sqrt(
        4 * 77 / 74_649_600 +
        4 * 631 / 223_948_800 +
        2 * 11 / 6_220_800 +
        2 * 7 / 1_749_600
    )

    # Test
    @test all(uₕ_coefs[[1, 3, 4, 6]] .≈ 1 / 24)
    @test all(uₕ_coefs[[2, 5]] .≈ 1 / 18)
    @test L2_err≈L2_err_expected rtol=1e-12
    @test (@allocated error_L2(
        u, uₕ_coefs, dof_map, nel_per_dim, element_side_lengths, W, xP, yP, φP)) == 0
end

@testitem "error_L2: AllSides(), Lagrange{1, 2}(), 4×3 grid, u=0, ‖φᵢ‖" begin
    using FEM: error_L2, Lagrange, AllSides, DOFMap, basis_functions
    using GaussQuadrature: legendre
    using StaticArrays: SVector, SMatrix

    # Setup
    fe = Lagrange{1, 2}()
    bc = AllSides()
    nel_per_dim = (4, 3)
    pmin = (0.0, 0.0)
    pmax = (1.0, 1.0)
    element_side_lengths = (pmax .- pmin) ./ nel_per_dim
    dof_map = DOFMap(fe, bc, nel_per_dim)

    Nx, Ny = nel_per_dim
    Δx, Δy = element_side_lengths

    u(x, y) = 0.0

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1) .+ pmin[1]
    yP = (Δy / 2) .* (P .+ 1) .+ pmin[2]

    φP = SMatrix{Npg, Npg}([basis_functions(fe, P[i], P[j])
                            for i in 1:Npg, j in 1:Npg])

    # Expected solution
    # Each interior basis function φᵢ has ‖φᵢ‖_L2 = sqrt(4 * Δx * Δy / 9)
    L2_err_expected = sqrt(4 * Δx * Δy / 9)

    # Test
    for i in 1:(dof_map.m)
        uₕ_coefs = zeros(dof_map.m)
        uₕ_coefs[i] = 1.0
        L2_err = error_L2(
            u, uₕ_coefs, dof_map, nel_per_dim, element_side_lengths, W, xP, yP, φP)
        @test L2_err ≈ L2_err_expected
    end
end

@testitem "error_L2: AllSides(), Lagrange{1, 1}(), 4 elements, u=x²-x" begin
    using FEM: error_L2, Lagrange, AllSides, DOFMap, basis_functions
    using GaussQuadrature: legendre
    using StaticArrays: SVector

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    Δx = 1 / nel_per_dim[1]
    dof_map = DOFMap(fe, bc, nel_per_dim)

    u(x) = x * (x - 1)
    uₕ_coefs = [u(i * Δx) for i in 1:(dof_map.m)]  # x = 0.25, 0.50, 0.75

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1)
    ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])

    # Compute
    L2_err = error_L2(u, uₕ_coefs, dof_map, Δx, W, xP, ϕP)

    # Expected solution
    L2_err_expected = sqrt(1 / 7680)

    # Test
    @test L2_err ≈ L2_err_expected
    @test (@allocated error_L2(u, uₕ_coefs, dof_map, Δx, W, xP, ϕP)) == 0
end
