# =============================================================================
# build_upper_to_full_maps11
# =============================================================================
@testitem "build_upper_to_full_maps11: size(M)=size(U)=3x3" begin
    using SparseArrays: sparse, nnz, nzrange
    using LinearAlgebra: Symmetric
    using FEM: build_upper_to_full_maps11

    M = sparse([1.0 2.0 3.0;
                2.0 4.0 5.0;
                3.0 5.0 6.0])
    U = sparse([1.0 2.0 3.0;
                0.0 4.0 5.0;
                0.0 0.0 6.0])
    M₁₁upper = Symmetric(U, :U)

    direct, mirror = build_upper_to_full_maps11(M, M₁₁upper)

    @test length(direct) == nnz(U)
    @test length(mirror) == nnz(U)
    @test eltype(direct) == eltype(M.rowval)
    @test eltype(mirror) == eltype(M.rowval)

    # For each nonzero kU in U at (i, j):
    #   direct[kU] must address M[i, j] in M.nzval
    #   mirror[kU] must address M[j, i] in M.nzval
    for j in 1:size(U, 2)
        for kU in nzrange(U, j)
            i = U.rowval[kU]
            @test M.rowval[direct[kU]] == i
            @test M.rowval[mirror[kU]] == j
        end
    end
end

@testitem "build_upper_to_full_maps11: size(M)=4x4, size(U)=3x3" begin
    using SparseArrays: sparse, nnz, nzrange
    using LinearAlgebra: Symmetric
    using FEM: build_upper_to_full_maps11

    # M is 4×4 but M₁₁upper covers only the top-left 3×3 block (m < size(M,1))
    M = sparse([1.0 2.0 3.0 0.0;
                2.0 4.0 5.0 0.0;
                3.0 5.0 6.0 0.0;
                0.0 0.0 0.0 7.0])
    U = sparse([1.0 2.0 3.0;
                0.0 4.0 5.0;
                0.0 0.0 6.0])
    M₁₁upper = Symmetric(U, :U)

    direct, mirror = build_upper_to_full_maps11(M, M₁₁upper)

    @test length(direct) == nnz(U)
    @test length(mirror) == nnz(U)
    for j in 1:size(U, 2)
        for kU in nzrange(U, j)
            i = U.rowval[kU]
            @test M.rowval[direct[kU]] == i
            @test M.rowval[mirror[kU]] == j
        end
    end
end

@testitem "build_upper_to_full_maps22: size(M)=5x5, size(U)=3x3" begin
    using SparseArrays: sparse, nnz, nzrange
    using LinearAlgebra: Symmetric
    using FEM: build_upper_to_full_maps22

    # M is 5×5 and M₂₂upper the bottom-right 3×3 block
    M = sparse([1.1 1.2 3.1 3.3 3.5;
                1.2 1.3 3.2 3.4 3.6;
                2.1 2.4 4.1 4.2 4.3;
                2.2 2.5 4.2 4.4 4.5;
                2.3 2.6 4.3 4.5 4.6])
    U = sparse([4.1 4.2 4.3;
                0.0 4.4 4.5;
                0.0 0.0 4.6])
    M₂₂upper = Symmetric(U, :U)

    direct, mirror = build_upper_to_full_maps22(M, M₂₂upper)

    @test length(direct) == nnz(U)
    @test length(mirror) == nnz(U)
    m₁ = 2
    for j in 1:size(U, 2)
        for kU in nzrange(U, j)
            i = U.rowval[kU]
            @test M.rowval[direct[kU]] == i + m₁
            @test M.rowval[mirror[kU]] == j + m₁
        end
    end
end

# =============================================================================
# scatter_symmetric!
# =============================================================================
@testitem "scatter_symmetric!:  size(M)=4x4, size(U)=3x3" begin
    using SparseArrays: sparse, nnz, nzrange
    using LinearAlgebra: Symmetric
    using FEM: build_upper_to_full_maps11, scatter_symmetric!

    M = sparse([1.0 2.0 3.0 0.0;
                2.0 4.0 5.0 0.0;
                3.0 5.0 6.0 0.0;
                0.0 0.0 0.0 7.0])
    U = sparse([10.0 20.0 30.0;
                0.0 40.0 50.0;
                0.0 0.0 60.0])
    M₁₁upper = Symmetric(U, :U)
    direct, mirror = build_upper_to_full_maps11(M, M₁₁upper)

    scatter_symmetric!(M, M₁₁upper, direct, mirror)

    M_expected = [10.0 20.0 30.0 0.0;
                  20.0 40.0 50.0 0.0;
                  30.0 50.0 60.0 0.0;
                  0.0 0.0 0.0 7.0]
    @test Matrix(M) ≈ M_expected
end

@testitem "scatter_symmetric!:  size(M)=5x5, size(U)=3x3" begin
    using SparseArrays: sparse, nnz, nzrange
    using LinearAlgebra: Symmetric
    using FEM: build_upper_to_full_maps22, scatter_symmetric!

    M = sparse([1.1 1.2 3.1 3.3 3.5;
                1.2 1.3 3.2 3.4 3.6;
                2.1 2.4 4.1 4.2 4.3;
                2.2 2.5 4.2 4.4 4.5;
                2.3 2.6 4.3 4.5 4.6])
    U = sparse([0.1 0.2 0.3;
                0.0 0.4 0.5;
                0.0 0.0 0.6])
    M₂₂upper = Symmetric(U, :U)
    direct, mirror = build_upper_to_full_maps22(M, M₂₂upper)

    scatter_symmetric!(M, M₂₂upper, direct, mirror)

    M_expected = sparse([1.1 1.2 3.1 3.3 3.5;
                         1.2 1.3 3.2 3.4 3.6;
                         2.1 2.4 0.1 0.2 0.3;
                         2.2 2.5 0.2 0.4 0.5;
                         2.3 2.6 0.3 0.5 0.6])
    @test Matrix(M) ≈ M_expected
end