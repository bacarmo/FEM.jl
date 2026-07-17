@testitem "assembly_local_matrix_ϕxϕ: Lagrange{1, 1}" begin
    using FEM: assembly_local_matrix_ϕxϕ, Lagrange
    using StaticArrays: SMatrix

    @testset "Type $T" for T in (Float64, Float32)
        # Setup
        fe = Lagrange{1, 1}()
        Δx = T(0.5)
        element_side_lengths = (Δx,)

        # Compute
        Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)

        # Expected solution
        Me_expected = (Δx / 6) * SMatrix{2, 2}([2 1;
                                                1 2])

        # Test
        @test Me ≈ Me_expected
        @test Me isa SMatrix{2, 2, T, 4}
    end
end

@testitem "assembly_local_matrix_ϕxϕ: Lagrange{1, 2}" begin
    using FEM: assembly_local_matrix_ϕxϕ, Lagrange
    using StaticArrays: SMatrix

    @testset "Type $T" for T in (Float64, Float32)
        # Setup 
        fe = Lagrange{1, 2}()
        Δx, Δy = T(0.5), T(0.5)
        element_side_lengths = (Δx, Δy)

        # Compute
        Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)

        # Expected solution
        Me_expected = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                                      2 4 1 2;
                                                      2 1 4 2;
                                                      1 2 2 4])
        # Test
        @test Me ≈ Me_expected
        @test Me isa SMatrix{4, 4, T, 16}
    end
end

@testitem "assembly_local_matrix_∇ϕx∇ϕ: Lagrange{1, 2}" begin
    using FEM: assembly_local_matrix_∇ϕx∇ϕ, Lagrange
    using StaticArrays: SMatrix

    @testset "Type $T" for T in (Float64, Float32)
        # Setup 
        fe = Lagrange{1, 2}()
        Δx, Δy = T(0.5), T(0.5)
        element_side_lengths = (Δx, Δy)

        # Compute
        Ke = assembly_local_matrix_∇ϕx∇ϕ(fe, element_side_lengths)

        # Expected solution
        Ke_expected = ((Δy / Δx) / 6) *
                      SMatrix{4, 4}([2 -2 1 -1;
                                     -2 2 -1 1;
                                     1 -1 2 -2;
                                     -1 1 -2 2]) +
                      ((Δx / Δy) / 6) *
                      SMatrix{4, 4}([2 1 -2 -1;
                                     1 2 -1 -2;
                                     -2 -1 2 1;
                                     -1 -2 1 2])

        # Test
        @test Ke ≈ Ke_expected
        @test Ke isa SMatrix{4, 4, T, 16}
    end
end

@testitem "assembly_local_matrix_ϕxc∇ϕ: Lagrange{1, 2}(), c = (1, 1)" begin
    using FEM: assembly_local_matrix_ϕxc∇ϕ, Lagrange
    using StaticArrays: SMatrix

    @testset "Type $T" for T in (Float64, Float32)
        # Setup 
        fe = Lagrange{1, 2}()
        Δx, Δy = T(1 / 4), T(1 / 3)
        element_side_lengths = (Δx, Δy)
        c = (one(T), one(T))

        # Compute
        K = assembly_local_matrix_ϕxc∇ϕ(fe, element_side_lengths, c)

        # Expected solution
        # K = c₁ * (Δy/2) * K_x + c₂ * (Δx/2) * K_y
        # K_x[a,b] = ∫∫ φₐ ∂φᵦ/∂ξ dξ dη over [-1,1]²
        # K_y[a,b] = ∫∫ φₐ ∂φᵦ/∂η dξ dη over [-1,1]²
        #! format: off
        K_x = (1/6) * SMatrix{4,4}(
            [-2  2 -1  1;
            -2  2 -1  1;
            -1  1 -2  2;
            -1  1 -2  2]
        )
        K_y = (1/6) * SMatrix{4,4}(
            [-2 -1  2  1;
            -1 -2  1  2;
            -2 -1  2  1;
            -1 -2  1  2]
        )
        #! format: on
        K_expected = (c[1] * Δy / 2) * K_x + (c[2] * Δx / 2) * K_y

        # Test
        @test K ≈ K_expected
        @test K isa SMatrix{4, 4, T, 16}
    end
end

@testitem "assembly_local_matrix_ΔϕxΔϕ: Hermite{3, 1}" begin
    using FEM: assembly_local_matrix_ΔϕxΔϕ, Hermite
    using StaticArrays: SMatrix

    @testset "Type $T" for T in (Float64, Float32)
        # Setup
        fe = Hermite{3, 1}()
        Δx = T(0.5)
        element_side_lengths = (Δx,)

        # Compute
        Ae = assembly_local_matrix_ΔϕxΔϕ(fe, element_side_lengths)

        # Expected solution
        Ae_expected = (4/Δx^3) * SMatrix{4, 4}([3 3 -3 3;
                                                3 4 -3 2;
                                                -3 -3 3 -3;
                                                3 2 -3 4])

        # Test
        @test Ae ≈ Ae_expected
        @test Ae isa SMatrix{4, 4, T, 16}
    end
end

@testitem "assembly_local_matrix_DG!: Lagrange{1, 1}(), AllSides(), ∂ₛg(x,v) = 1" begin
    using FEM: assembly_local_matrix_DG!, assembly_local_matrix_ϕxϕ, Lagrange, DOFMap,
               AllSides, basis_functions
    using StaticArrays: SVector, SMatrix
    using GaussQuadrature: legendre

    @testset "Type $T" for T in (Float64, Float32)
        # Setup
        fe = Lagrange{1, 1}()
        bc = AllSides()
        nel_per_dim = (2,)
        element_side_lengths = one(T) ./ nel_per_dim
        Δx = element_side_lengths[1]
        ∂ₛg(x, s) = one(T)

        dof_map = DOFMap(fe, bc, nel_per_dim)
        eq = dof_map.EQoLG[1]
        m = dof_map.m
        v = ones(T, m)

        Npg = 2
        P_raw, W_raw = legendre(T, Npg)
        P, W = SVector{Npg, T}(P_raw), SVector{Npg, T}(W_raw)

        xeP = (Δx / 2) .* (P .+ one(T)) # Physical quadrature points on the element [0,Δx]×[0,Δx]
        ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])
        nb = length(ϕP[1])
        W_ϕPϕP = SVector{Npg}([SMatrix{nb, nb, T}(W[j] * ϕP[j][a] * ϕP[j][b]
                               for a in 1:nb, b in 1:nb)
                               for j in 1:Npg])

        # Compute
        DG = zeros(T, nb, nb)
        assembly_local_matrix_DG!(DG, ∂ₛg, v, m, eq, xeP, ϕP, W_ϕPϕP)

        # Expected solution
        Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)
        DG_expected = Me * (2 / Δx)

        # Test correctness
        @testset "Entry ($a,$b)" for b in axes(DG, 2), a in 1:b # Upper triangle: a ≤ b
            @test DG[a, b] ≈ DG_expected[a, b]
        end
        @test eltype(DG) == T

        # Test allocation-free operation
        alloc = @allocated assembly_local_matrix_DG!(DG, ∂ₛg, v, m, eq, xeP, ϕP, W_ϕPϕP)
        @test alloc == 0
    end
end

@testitem "assembly_local_matrix_DG!: Lagrange{1, 1}(), AllSides(), ∂ₛg(x,v) = x²+v²" begin
    using FEM: assembly_local_matrix_DG!, Lagrange, DOFMap, AllSides, basis_functions
    using StaticArrays: SVector, SMatrix
    using GaussQuadrature: legendre

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    element_side_lengths = 1 ./ nel_per_dim
    Δx = element_side_lengths[1]
    ∂ₛg(x, s) = x^2 + s^2

    dof_map = DOFMap(fe, bc, nel_per_dim)
    v = ones(dof_map.m)

    Npg = 4
    P_raw, W_raw = legendre(Npg)
    P, W = SVector{Npg}(P_raw), SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1) # Physical quadrature points on the element [0,Δx]×[0,Δx]
    ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])
    nb = length(ϕP[1])
    W_ϕPϕP = SVector{Npg}([SMatrix{nb, nb, Float64}(W[j] * ϕP[j][a] * ϕP[j][b]
                           for a in 1:nb, b in 1:nb)
                           for j in 1:Npg])

    # Expected solution for elements e = 1, 2, 3, 4
    DG_expected = (
        [17/240 17/160; 17/160 17/40],    # e=1: [0.00, 0.25], Vₕ=(1+ξ)/2
        [11/15 61/160; 61/160 191/240],   # e=2: [0.25, 0.50], Vₕ=1
        [211/240 223/480; 223/480 59/60], # e=3: [0.50, 0.75], Vₕ=1
        [101/120 57/160; 57/160 157/240]  # e=4: [0.75, 1.00], Vₕ=(1-ξ)/2
    )

    # Compute and test
    DG = zeros(Float64, nb, nb)
    @testset "Element $e" for e in 1:4
        xeP = xP .+ (e - 1) * Δx
        assembly_local_matrix_DG!(DG, ∂ₛg, v, dof_map.m, dof_map.EQoLG[e], xeP, ϕP, W_ϕPϕP)
        @testset "Entry ($a,$b)" for b in axes(DG, 2), a in 1:b # Upper triangle: a ≤ b
            @test DG[a, b] ≈ DG_expected[e][a, b]
        end
    end
end

@testitem "assembly_local_matrix_DF!: Lagrange{1, 2}(), LeftRightTop(), f(s) = 1.0" begin
    using FEM: assembly_local_matrix_DF!, assembly_local_matrix_ϕxϕ, Lagrange, DOFMap,
               LeftRightTop, basis_functions
    using StaticArrays: SVector, SMatrix
    using GaussQuadrature: legendre

    # Setup
    fe = Lagrange{1, 2}()
    bc = LeftRightTop()
    nel_per_dim = (4, 3)
    element_side_lengths = 1 ./ nel_per_dim
    Δx, Δy = element_side_lengths
    f(s) = 1.0

    dof_map = DOFMap(fe, bc, nel_per_dim)
    d = ones(dof_map.m)

    Npg = 2
    P_raw, W_raw = legendre(Npg)
    P, W = SVector{Npg}(P_raw), SVector{Npg}(W_raw)
    φP = SMatrix{Npg, Npg}([basis_functions(fe, P[i], P[j]) for i in 1:Npg, j in 1:Npg])
    nb = length(φP[1, 1])
    W_φPφP = SMatrix{Npg, Npg}([SMatrix{nb, nb, Float64}(
                                    W[i] * W[j] * φP[i, j][a] * φP[i, j][b]
                                for a in 1:nb, b in 1:nb)
                                for i in 1:Npg, j in 1:Npg])

    # Compute
    DF = zeros(nb, nb)
    assembly_local_matrix_DF!(DF, f, d, dof_map.m, dof_map.EQoLG[1], φP, W_φPφP)

    # Expected solution
    Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)
    DF_expected = Me * (4 / (Δx * Δy))

    # Test
    @testset "Entry ($a,$b)" for b in axes(DF, 2), a in 1:b # Upper triangle: a ≤ b
        @test DF[a, b] ≈ DF_expected[a, b]
    end

    # Test allocation-free operation
    ## Warning: Direct field access (dof_map.m, dof_map.EQoLG[1]) is causing allocations. Why?
    m = dof_map.m
    eq = dof_map.EQoLG[1]

    alloc = @allocated assembly_local_matrix_DF!(DF, f, d, m, eq, φP, W_φPφP)
    @test alloc == 0
end