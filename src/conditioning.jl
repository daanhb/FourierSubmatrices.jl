
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
