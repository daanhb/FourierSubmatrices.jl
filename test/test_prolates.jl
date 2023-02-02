
Nvalues = [31, 34]
pvalues = [9, 10]
qvalues = [5, 7]
Tvalues = [Float64, BigFloat]

function test_prolates(N, p, q, T)
    Ac = BlockDFT.CenteredBlock{T}(N, p, q)
    Pleft = BlockDFT.DiscreteProlateMatrix{T}(N, q, p)
    Pright = BlockDFT.DiscreteProlateMatrix{T}(N, p, q)
    Jleft = BlockDFT.jacobi_prolate(N, q, p, T)
    Jright = BlockDFT.jacobi_prolate(N, p, q, T)

    # the discrete prolate matrices correspond to the normal equations
    @test norm(Ac'*Ac/p-Pright) < sqrt(eps(T))
    @test norm(Ac*Ac'/q-Pleft) < sqrt(eps(T))
    # verify commutation relations
    @test norm(Pleft*Jleft-Jleft*Pleft) < sqrt(eps(T))
    @test norm(Pright*Jright-Jright*Pright) < sqrt(eps(T))

    # verify diagonalization
    Vleft = pdpss(Pleft)
    Dleft = Vleft'*Pleft*Vleft
    Vright = pdpss(Pright)
    Dright = Vright'*Pright*Vright
    @test norm(Dleft-Diagonal(diag(Dleft))) < sqrt(eps(T))

    A1 = DFTBlock{T}(N, p, q)
    Dp, Dq, c = BlockDFT.blockshift_top_to_center(N, p, q, T)
    @test norm(Ac - Dp*A1*Dq/c) < sqrt(eps(T))

    x = rand(T, q)
    @test norm(collect(Ac)*x-Ac*x) < sqrt(eps(T))
    @test norm(collect(A1)*x-A1*x) < sqrt(eps(T))

    A = DFTBlock{T}(N, 2:2+p-1, 3:3+q-1)
    Dp, Dq, c = BlockDFT.blockshift_sub_to_top(N, A.Ip, A.Iq, T)
    @test norm(A - 1/c*Dp*A1*Dq) < sqrt(eps(T))
    @test norm(collect(A)*x-A*x) < sqrt(eps(T))

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
