
Nvalues = [31, 34]
pvalues = [9, 10]
qvalues = [5, 7]
Tvalues = [Float64, BigFloat]

function test_prolates(N, p, q, T)
    Ac = BlockDFT.CenteredBlock(N, p, q)
    Pleft = BlockDFT.DiscreteProlateMatrix(N, q, p)
    Pright = BlockDFT.DiscreteProlateMatrix(N, p, q)
    Jleft = BlockDFT.jacobi_prolate(N, q, p)
    Jright = BlockDFT.jacobi_prolate(N, p, q)

    # the discrete prolate matrices correspond to the normal equations
    @test norm(Ac'*Ac/p-Pright) < 1e-10
    @test norm(Ac*Ac'/q-Pleft) < 1e-10
    # verify commutation relations
    @test norm(Pleft*Jleft-Jleft*Pleft) < 1e-10
    @test norm(Pright*Jright-Jright*Pright) < 1e-10

    # verify diagonalization
    Vleft = BlockDFT.pdpss(Pleft)
    Dleft = Vleft'*Pleft*Vleft
    Vright = BlockDFT.pdpss(Pright)
    Dright = Vright'*Pright*Vright
    @test norm(Dleft-Diagonal(diag(Dleft))) < 1e-10

    A = DFTBlock(N, p, q)
    Adense = collect(A)
    c1 = cond(A)
    c2 = cond(Adense)
    @test abs(c1-c2)/c2 < 1e-4

end

@testset "Prolate vectors" begin
    for T in Tvalues
        for N in Nvalues
            for p in pvalues
                for q in qvalues
                    test_prolates(N, p, q, T)
                    test_prolates(N, q, p, T)
                end
            end
        end
    end
end
