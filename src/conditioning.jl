
"""
Start from an approximate eigenvalue `λ` and eigenvector `v` of `A` and
refine iteratively to produce a higher accuracy estimation.
"""
function refine_eigenvalue(A, lambda, v, K=5)
    if min(size(A)...) == 1
        A[1], [one(eltype(A))]
    else
        for k in 1:K
            u = (A - lambda*I) \ v
            v1 = u/norm(u)
            lambda1 = (v1'*A*v1)[1]
            if !isnan(lambda1)
                lambda = lambda1
                v = v1
            end
        end
        lambda, v
    end
end

function LinearAlgebra.cond(A::DFTBlock{T}) where T
    N = dftlength(A)
    p,q = size(A)
    fourier_submatrix_cond(N, p, q, T)
end

function fourier_submatrix_cond(N, p, q, T = Float64)
    D_q = dft_diagonal_scaling(N, q, -(p-1)/2, T)
    omega = twiddle(N, T)

    if p < q
        return fourier_submatrix_cond(N, q, p, T)
    end

    J1 = jacobi_prolate(N, p, q, Float64)
    f_E1,f_V1 = eigen(J1, 1:1)
    f_E2,f_V2 = eigen(J1, q:q)
    if T != Float64
        J1_big = jacobi_prolate(N, p, q, BigFloat)
        E1,JV1 = refine_eigenvalue(J1_big, BigFloat(f_E1[1]), BigFloat.(f_V1[:,1]))
        E2,JV2 = refine_eigenvalue(J1_big, BigFloat(f_E2[1]), BigFloat.(f_V2[:,1]))
    else
        E1,JV1 = f_E1[1], f_V1[:,1]
        E2,JV2 = f_E2[1], f_V2[:,1]
    end

    V1 = D_q*JV1
    V2 = D_q*JV2

    s1 = norm(fft([V1; zeros(N-q)])[1:p])
    s2 = norm(fft([V2; zeros(N-q)])[1:p])

    s1/s2
end
