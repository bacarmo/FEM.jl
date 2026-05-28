@testitem "assembly_local_∫basis: Lagrange{Deg, 2}" begin
    using FEM: assembly_local_∫basis, basis_functions, Lagrange

    # Setup
    T = Float64
    nel_per_dim = (4, 3)
    element_side_lengths = one(T) ./ nel_per_dim
    Δx, Δy = element_side_lengths
    jacobian = Δx * Δy / 4

    # Tests
    @testset "Deg = 1" begin
        result = assembly_local_∫basis(Lagrange{1, 2}(), element_side_lengths)
        expect_result = fill(jacobian, 4)
        @test result ≈ expect_result
    end

    @testset "Deg = 2" begin
        result = assembly_local_∫basis(Lagrange{2, 2}(), element_side_lengths)
        expect_result = (jacobian / 9) .* [1, 4, 1, 4, 16, 4, 1, 4, 1]
        @test result ≈ expect_result
    end

    @testset "Deg = 3" begin
        result = assembly_local_∫basis(Lagrange{3, 2}(), element_side_lengths)
        expect_result = (jacobian / 16) .* [
            1, 3, 3, 1,
            3, 9, 9, 3,
            3, 9, 9, 3,
            1, 3, 3, 1]
        @test result ≈ expect_result
    end
end