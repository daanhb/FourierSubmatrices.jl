
Nvalues = [31, 34]
pvalues = [9, 10]
qvalues = [5, 7]
Tvalues = [Float64, BigFloat]

function test_prolates(N, p, q, T)
    A = DFTBlockPlan(N, p, q)
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
                    # test_prolates(N, q, p, T)
                end
            end
        end
    end
end
