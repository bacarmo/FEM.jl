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

@testitem "norm_H01²: AllSides(), Lagrange{1, 1}(), uₕ=ϕᵢ" begin
    using FEM: norm_H01², Lagrange, AllSides, DOFMap, basis_functions_derivatives
    using GaussQuadrature: legendre
    using StaticArrays: SVector

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    Δx = 1 / nel_per_dim[1]
    dof_map = DOFMap(fe, bc, nel_per_dim)
    m = dof_map.m

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)

    ∇ϕP = SVector{Npg}([basis_functions_derivatives(fe, P[i]) for i in 1:Npg])

    for i in 1:m
        uₕ_coefs = zeros(m)
        uₕ_coefs[i] = 1.0

        # Compute
        result = norm_H01²(uₕ_coefs, dof_map, Δx, W, ∇ϕP)

        # Expected solution
        result_expected = 2/Δx

        # Test
        @test result ≈ result_expected
    end

    # Test allocations
    uₕ_coefs = zeros(m)
    uₕ_coefs[m] = 1.0
    @test (@allocated norm_H01²(uₕ_coefs, dof_map, Δx, W, ∇ϕP)) == 0
end

@testitem "norm_H01²: AllSides(), Lagrange{2, 1}(), uₕ=ϕᵢ" begin
    using FEM: norm_H01², Lagrange, AllSides, DOFMap, basis_functions_derivatives
    using GaussQuadrature: legendre
    using StaticArrays: SVector

    fe = Lagrange{2, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    Δx = 1 / nel_per_dim[1]
    dof_map = DOFMap(fe, bc, nel_per_dim)
    m = dof_map.m
    EQoLG = dof_map.EQoLG

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)
    ∇ϕP = SVector{Npg}([basis_functions_derivatives(fe, P[i]) for i in 1:Npg])

    # Analytic ‖∇ϕₐ‖²_{L²(-1,1)} for the reference quadratic-Lagrange basis
    #   ϕ₁ = ξ(ξ-1)/2, ϕ₂ = 1-ξ², ϕ₃ = ξ(ξ+1)/2
    #  dϕ₁ = ξ-1/2,   dϕ₂ = -2ξ, dϕ₃ = ξ+1/2
    vertex = 7 / 6 # a = 1 or a = 3
    bubble = 8 / 3 # a = 2 (element-interior, not shared)

    for i in 1:m
        uₕ_coefs = zeros(m)
        uₕ_coefs[i] = 1.0

        result = norm_H01²(uₕ_coefs, dof_map, Δx, W, ∇ϕP)
        result_expected = isodd(i) ? bubble * 2 / Δx : 2 * vertex * 2 / Δx

        @test result ≈ result_expected
    end
end

@testitem "norm_H01²: AllSides(), Hermite{3, 1}(), uₕ=ϕᵢ" begin
    using FEM: norm_H01², Hermite, AllSides, DOFMap, basis_functions_derivatives
    using GaussQuadrature: legendre
    using StaticArrays: SVector

    fe = Hermite{3, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    Δx = 1 / nel_per_dim[1]
    dof_map = DOFMap(fe, bc, nel_per_dim)
    m = dof_map.m
    EQoLG = dof_map.EQoLG

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P = SVector{Npg}(P_raw)
    W = SVector{Npg}(W_raw)
    ∇ϕP = SVector{Npg}([basis_functions_derivatives(fe, P[i]) for i in 1:Npg])

    # Analytic ‖∇ϕₐ‖²_{L²(-1,1)} for the reference cubic-Hermite basis
    #   ϕ₁ = (2+ξ)(1-ξ)²/4,   ϕ₂ = (ξ+1)(1-ξ)²/4,  ϕ₃ = (2-ξ)(1+ξ)²/4,  ϕ₄ = (ξ-1)(1+ξ)²/4
    #  dϕ₁ = 3(-1+ξ)(1+ξ)/4, dϕ₂ = (ξ-1)(1+3ξ)/4, dϕ₃ = 3(1-ξ)(1+ξ)/4, dϕ₄ = (ξ+1)(-1+3ξ)/4
    value = 3 / 5  # a = 1 or a = 3
    deriv = 4 / 15 # a = 2 or a = 4

    for i in 1:m
        uₕ_coefs = zeros(m)
        uₕ_coefs[i] = 1.0

        result = norm_H01²(uₕ_coefs, dof_map, Δx, W, ∇ϕP)
        result_expected = isodd(i) ? 2 * value * 2 / Δx : 2 * deriv * 2 / Δx

        @test result ≈ result_expected
    end
end