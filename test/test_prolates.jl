
"Is the given svd factorization accurate?"
function correct_svd(A, u, s, v, tol)
    result = abs(cond(u)-1) < tol
    result = result && abs(cond(v)-1) < tol
    result = result && all(t->t>=0, s)
    D = zeros(eltype(A),size(A))
    r = length(s)
    D[1:r,1:r] = Diagonal(s)
    result = result && (norm(A - u*D*v') < tol)
end

function test_prolates(N, p, q, T)
    Ac = FourierSubmatrices.CenteredBlock{T}(N, p, q)
    Pleft = FourierSubmatrices.DiscreteProlateMatrix{T}(N, q, p)
    Pright = FourierSubmatrices.DiscreteProlateMatrix{T}(N, p, q)
    Jleft = FourierSubmatrices.pdpss_tridiag_matrix(N, q, p, T)
    Jright = FourierSubmatrices.pdpss_tridiag_matrix(N, p, q, T)

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
    Dp, Dq, c = FourierSubmatrices.blockshift_top_to_center(N, p, q, T)
    @test norm(Ac - Dp*A1*Dq/c) < sqrt(eps(T))

    x = rand(T, q)
    @test norm(collect(Ac)*x-Ac*x) < sqrt(eps(T))
    @test norm(collect(A1)*x-A1*x) < sqrt(eps(T))

    A = DFTBlock{T}(N, 2:2+p-1, 3:3+q-1)
    Dp, Dq, c = FourierSubmatrices.blockshift_top_to_sub(N, A.Ip, A.Iq, T)
    @test norm(A - 1/c*Dp*A1*Dq) < sqrt(eps(T))
    @test norm(collect(A)*x-A*x) < sqrt(eps(T))

    Dp, Dq, c = FourierSubmatrices.blockshift_sub_to_center(N, A.Ip, A.Iq, T)
    @test norm(Ac - 1/c*Dp*A*Dq) < sqrt(eps(T))

    u,s,v = svd(Ac)
    @test correct_svd(Ac, u, s, v, sqrt(eps(T)))
    u2,s2,v2 = svd(A)
    @test correct_svd(A, u2, s2, v2, sqrt(eps(T)))

    Adense = collect(A)
    c1 = cond(A)
    c2 = cond(Adense)
    @test abs(c1-c2)/c2 < sqrt(eps(T))
end

# we test all combinations of parity of N, p and q
Nvalues = [31, 34]
pvalues = [9, 10]
qvalues = [5, 7]
Tvalues = [Float64, BigFloat]

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
