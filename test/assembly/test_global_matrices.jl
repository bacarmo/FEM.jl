@testitem "assembly_global_matrix: Lagranget{1, 1}(), AllSides(), Me" begin
    using FEM: assembly_global_matrix, Lagrange, DOFMap, AllSides
    using SparseArrays: sparse
    using StaticArrays: SMatrix

    # Setup
    # 1D mesh with 4 linear elements; AllSides() removes boundary DOFs:
    # Basis layout:  [x  1  2  3  x]  (x = removed, 3 free interior DOFs)
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    dof_map = DOFMap(fe, bc, nel_per_dim)

    Δx = 1 / nel_per_dim[1]
    Me = (Δx / 6) * SMatrix{2, 2}(2, 1, 1, 2)

    # Compute
    M = assembly_global_matrix(Me, dof_map)

    # Analytical solution
    #! format: off
    M_expected = (Δx / 6) * sparse([
    #  1  2  3
       4  1  0;  # DOF 1
       1  4  1;  # DOF 2
       0  1  4   # DOF 3
    ])
    #! format: on

    # Test
    @test M ≈ M_expected
end

@testitem "assembly_global_matrix: Lagrange{1, 1}(), AllSides(), Me Symmetric" begin
    using FEM: assembly_global_matrix, Lagrange, DOFMap, AllSides
    using SparseArrays: SparseMatrixCSC
    using StaticArrays: SMatrix
    using LinearAlgebra: Symmetric

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (2^3,)
    dof_map = DOFMap(fe, bc, nel_per_dim)

    Δx = 1 / nel_per_dim[1]
    Me = (Δx / 6) * SMatrix{2, 2}(2, 1, 1, 2)
    Me_sym = Symmetric(Me)

    # Compute
    M = assembly_global_matrix(Me, dof_map)
    M_sym = assembly_global_matrix(Me_sym, dof_map)

    # Test
    @test M isa SparseMatrixCSC{Float64, Int64}
    @test M_sym isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
    @test M ≈ M_sym
end

@testitem "assembly_global_matrix: Lagrange{1, 2}(), LeftRightTop(), Me" begin
    using FEM: assembly_global_matrix, Lagrange, DOFMap, LeftRightTop
    using StaticArrays: SMatrix
    using SparseArrays: sparse

    # Setup
    # 2D mesh with 4×3 bilinear elements; LeftRightTop() removes boundary DOFs:
    # Basis layout (5×4 grid, 9 free DOFs):
    # Row 4 (top):    [x  x  x  x  x]  - Top BC (removed)
    # Row 3:          [x  7  8  9  x]  - Left/Right BC (removed)
    # Row 2:          [x  4  5  6  x]  - Left/Right BC (removed)
    # Row 1 (bottom): [x  1  2  3  x]  - Left/Right BC (removed), bottom free
    fe = Lagrange{1, 2}()
    bc = LeftRightTop()
    nel_per_dim = (4, 3)
    element_side_lengths = (16, 27) ./ nel_per_dim
    dof_map = DOFMap(fe, bc, nel_per_dim)

    Δx, Δy = element_side_lengths
    Me = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                         2 4 1 2;
                                         2 1 4 2;
                                         1 2 2 4])

    # Compute
    M = assembly_global_matrix(Me, dof_map)

    # Analytical solution
    #! format: off
    M_expected = (Δx * Δy / 36) * sparse([
    #  1   2   3   4   5   6   7   8   9
       8   2   0   4   1   0   0   0   0;  # DOF 1
       2   8   2   1   4   1   0   0   0;  # DOF 2
       0   2   8   0   1   4   0   0   0;  # DOF 3
       4   1   0  16   4   0   4   1   0;  # DOF 4
       1   4   1   4  16   4   1   4   1;  # DOF 5
       0   1   4   0   4  16   0   1   4;  # DOF 6
       0   0   0   4   1   0  16   4   0;  # DOF 7
       0   0   0   1   4   1   4  16   4;  # DOF 8
       0   0   0   0   1   4   0   4  16   # DOF 9
    ])
    #! format: on

    # Test
    @test M ≈ M_expected
end

@testitem "assembly_global_matrix: Lagrange{1, 2}(), LeftRightTop(), Me Symmetric" begin
    using FEM: assembly_global_matrix, Lagrange, DOFMap, LeftRightTop
    using SparseArrays: SparseMatrixCSC
    using StaticArrays: SMatrix
    using LinearAlgebra: Symmetric

    # Setup
    fe = Lagrange{1, 2}()
    bc = LeftRightTop()
    nel_per_dim = (4, 3)
    element_side_lengths = (16, 27) ./ nel_per_dim
    dof_map = DOFMap(fe, bc, nel_per_dim)

    Δx, Δy = element_side_lengths
    Me = (Δx * Δy / 36) * SMatrix{4, 4}([4 2 2 1;
                                         2 4 1 2;
                                         2 1 4 2;
                                         1 2 2 4])
    Me_sym = Symmetric(Me)

    # Compute
    M = assembly_global_matrix(Me, dof_map)
    M_sym = assembly_global_matrix(Me_sym, dof_map)

    # Test
    @test M isa SparseMatrixCSC{Float64, Int64}
    @test M_sym isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
    @test M ≈ M_sym
end

@testitem "assembly_global_matrix_DG: Lagrange{1, 1}(), AllSides(), ∂ₛg(x,v) = 1.0" begin
    using FEM: assembly_global_matrix_DG, assembly_local_matrix_ϕxϕ, assembly_global_matrix,
               Lagrange, DOFMap, AllSides, basis_functions
    using SparseArrays: SparseMatrixCSC
    using LinearAlgebra: Symmetric
    using GaussQuadrature: legendre
    using StaticArrays: SVector, SMatrix

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (2,)
    element_side_lengths = 1 ./ nel_per_dim
    Δx = element_side_lengths[1]
    ∂ₛg(x, s) = 1.0

    dof_map = DOFMap(fe, bc, nel_per_dim)
    v = ones(dof_map.m)

    Npg = 2
    P_raw, W_raw = legendre(Npg)
    P, W = SVector{Npg}(P_raw), SVector{Npg}(W_raw)

    xP = (Δx / 2) .* (P .+ 1) # Physical quadrature points on the element [0,Δx]×[0,Δx]
    ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])
    nb = length(ϕP[1])
    W_ϕPϕP = SVector{Npg}([SMatrix{nb, nb, Float64}(W[j] * ϕP[j][a] * ϕP[j][b]
                           for a in 1:nb, b in 1:nb)
                           for j in 1:Npg])

    # Compute
    DG_global = assembly_global_matrix_DG(1.0, ∂ₛg, v, dof_map, Δx, xP, ϕP, W_ϕPϕP)

    # Expected solution
    Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)
    DG_global_expected = assembly_global_matrix(Symmetric(Me), dof_map)

    # Test
    @test DG_global isa Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
    @test DG_global ≈ DG_global_expected
    @test size(DG_global) == (dof_map.m, dof_map.m)
end

@testitem "assembly_global_matrix_DG: Lagrange{1, 1}(), AllSides(), ∂ₛg(x,v) = x²+v²" begin
    using FEM: assembly_global_matrix_DG, assembly_local_matrix_ϕxϕ, assembly_global_matrix,
               Lagrange, DOFMap, AllSides, basis_functions
    using GaussQuadrature: legendre
    using StaticArrays: SVector, SMatrix

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

    # Compute
    DG_global = assembly_global_matrix_DG(1.0, ∂ₛg, v, dof_map, Δx, xP, ϕP, W_ϕPϕP)

    # Expected solution
    #! format: off
    DG_global_expected = (1/8) * [
        17/40 + 11/15      61/160              0.0;
        61/160             191/240 + 211/240   223/480;
        0.0                223/480             59/60 + 101/120
    ]
    #! format: on

    # Test
    @test DG_global ≈ DG_global_expected
    @test size(DG_global) == (dof_map.m, dof_map.m)
end

@testitem "assembly_global_matrix_DF: Lagrange{1, 2}(), LeftRightTop(), f(s) = 1.0" begin
    using FEM: assembly_global_matrix_DF, assembly_local_matrix_ϕxϕ,
               assembly_global_matrix, Lagrange, DOFMap, LeftRightTop, basis_functions
    using StaticArrays: SVector, SMatrix
    using GaussQuadrature: legendre
    using LinearAlgebra: Symmetric

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
    DF_global = assembly_global_matrix_DF(
        1.0, f, d, dof_map, element_side_lengths, φP, W_φPφP)

    # Expected solution
    Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)
    DF_global_expected = assembly_global_matrix(Symmetric(Me), dof_map)

    # Test
    @test DF_global ≈ DF_global_expected
    @test size(DF_global) == (dof_map.m, dof_map.m)
end

@testitem "assembly_global_matrix_DF: Lagrange{1, 1}(), AllSides(), f(s) = 1.0" begin
    using FEM: assembly_global_matrix_DF, assembly_local_matrix_ϕxϕ,
               assembly_global_matrix, Lagrange, DOFMap, AllSides, basis_functions
    using StaticArrays: SVector, SMatrix
    using GaussQuadrature: legendre
    using LinearAlgebra: Symmetric

    # Setup
    fe = Lagrange{1, 1}()
    bc = AllSides()
    nel_per_dim = (4,)
    element_side_lengths = 1 ./ nel_per_dim
    f(s) = 1.0

    dof_map = DOFMap(fe, bc, nel_per_dim)
    d = ones(dof_map.m)

    Npg = 2
    P_raw, W_raw = legendre(Npg)
    P, W = SVector{Npg}(P_raw), SVector{Npg}(W_raw)

    ϕP = SVector{Npg}([basis_functions(fe, P[i]) for i in 1:Npg])
    nb = length(ϕP[1])
    W_ϕPϕP = SVector{Npg}([SMatrix{nb, nb, Float64}(W[j] * ϕP[j][a] * ϕP[j][b]
                           for a in 1:nb, b in 1:nb)
                           for j in 1:Npg])

    # Compute
    DF_global = assembly_global_matrix_DF(
        1.0, f, d, dof_map, element_side_lengths, ϕP, W_ϕPϕP)

    # Expected solution
    Me = assembly_local_matrix_ϕxϕ(fe, element_side_lengths)
    DF_global_expected = assembly_global_matrix(Symmetric(Me), dof_map)

    # Test
    @test DF_global ≈ DF_global_expected
    @test size(DF_global) == (dof_map.m, dof_map.m)
end