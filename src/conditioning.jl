
LinearAlgebra.cond(A::CenteredBlock) = dftcond_centered(A)

function dftcond_centered(A::CenteredBlock{T}) where T
    N = dftlength(A)
    p,q = size(A)
    if q > p
        return dftcond_centered(CenteredBlock{T}(N, q, p))
    end
    if min(p,q) > 10
        dftcond_centered_sparse(A)
    else
        u,s,v = svd(A)
        s[1]/s[end]
    end
end

function dftcond_centered_sparse(A::CenteredBlock{T}) where T
    N = dftlength(A)
    p,q = size(A)
    @assert p >= q
    Pleft = DiscreteProlateMatrix{T}(N, q, p)
    Pright = DiscreteProlateMatrix{T}(N, p, q)
    U1 = pdpss(Pleft, 1:1)
    U2 = pdpss(Pleft, q:q)
    V1 = pdpss(Pright, 1:1)
    V2 = pdpss(Pright, q:q)
    s1 = compute_singular_value(A, U1[:,1], V1[:,1])
    s2 = compute_singular_value(A, U2[:,1], V2[:,1])
    s1/s2
end

LinearAlgebra.cond(A::DFTBlock) = cond(centered(A))


"""
Start from an approximate eigenvalue `λ` and eigenvector `v` of `A` and
refine iteratively to produce a higher accuracy estimation.
"""
function refine_eigenvalue(A, lambda, v, K=5)
    T = eltype(A)
    if min(size(A)...) == 1
        A[1], [one(T)]
    else
        r = norm(A*v-lambda*v)
        k = 0
        while k < K && r > sqrt(eps(T))
            k += 1
            u = (A - lambda*I) \ v
            v1 = u/norm(u)
            lambda1 = (v1'*A*v1)[1]
            if !isnan(lambda1)
                lambda = lambda1
                v = v1
            end
            r = norm(A*v-lambda*v)
        end
        lambda, v
    end
end
