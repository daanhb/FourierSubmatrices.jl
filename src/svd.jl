
"""
Compute the diagonal scaling matrix of length `p` associated with the DFT
of length `N`, raised to the power `l`.
"""
function dft_diagonal_scaling(N, p, l = 1, ::Type{T} = Float64) where T
    omega = twiddle(N, T)
    Diagonal([omega^(-(j-1)*l) for j in 1:p])
end


"""
    Dp, Dq, c = blockshift_top_to_center(N, p, q[, T])

Compute the map from the topleft `p x q` subblock `B = A[1:p,1:q]` of the
`N × N` DFT matrix `A`, to the centered block `C` with the same size as `B`.

The relation is given by `C = 1/c * Dp * B * Dq`.
"""
function blockshift_top_to_center(N, p, q, ::Type{T} = Float64) where T
    ω = twiddle(N, T)
    pivot_p = (p-1) / T(2)
    pivot_q = (q-1) / T(2)
    Dp = dft_diagonal_scaling(N, p, -pivot_q, T)
    Dq = dft_diagonal_scaling(N, q, -pivot_p, T)
    c = ω^(pivot_p * pivot_q)
    Dp, Dq, c
end

"""
    Dp, Dq, c = blockshift_top_to_sub(N, Ip, Iq[, T])

Compute the map from the topleft `p x q` subblock `B = A[1:p,1:q]` of the
`N × N` DFT matrix `A`, to a subblock `C` with the same size as `B`.

The relation is given by `C = 1/c * Dp * B * Dq`.
"""
function blockshift_top_to_sub(N, Ip, Iq, ::Type{T} = Float64) where T
    ω = twiddle(N, T)
    shift_q = first(Iq)-1
    shift_p = first(Ip)-1
    Dp = dft_diagonal_scaling(N, length(Ip), shift_q, T)
    Dq = dft_diagonal_scaling(N, length(Iq), shift_p, T)
    c = ω^(shift_q*(first(Ip)-1))
    Dp, Dq, c
end

"""
    Dp, Dq, c = blockshift_center_to_sub(N, Ip, Iq[, T])

Compute the map from the centered `p x q` subblock `B` of the
`N × N` DFT matrix `A`, to a subblock `C` with the same size as `B`.

The relation is given by `C = 1/c * Dp * B * Dq`.
"""
function blockshift_sub_to_center(N, Ip, Iq, ::Type{T} = Float64) where T
    Dp1, Dq1, c1 = blockshift_top_to_center(N, length(Ip), length(Iq), T)
    Dp2, Dq2, c2 = blockshift_top_to_sub(N, Ip, Iq, T)
    Dp1*Dp2', Dq1*Dq2', c1*c2'
end

"The inverse of `blockshift_sub_to_center`."
function blockshift_center_to_sub(N, Ip, Iq, ::Type{T} = Float64) where T
    Dp, Dq, c = blockshift_sub_to_center(N, Ip, Iq, T)
    Dp', Dq', c'
end


"Matrix-vector product with the topleft subblock of the DFT matrix."
block_mv(N, p, q, x) = fft([x; zeros(eltype(x),N-q)])[1:p]

function LinearAlgebra.mul!(y::AbstractVector, A::CenteredBlock{T}, x::AbstractVector) where T
    N = dftlength(A)
    p,q = size(A)
    Dp, Dq, c = blockshift_top_to_center(N, p, q, T)
    x1 = Dq*x/c
    y1 = block_mv(N, p, q, x1)
    y[:] = Dp*y1
    y
end

istopblock(A::DFTBlock) = first(A.Ip) == first(A.Iq) == 1

function LinearAlgebra.mul!(y::AbstractVector, A::DFTBlock{T}, x::AbstractVector) where T
    N = dftlength(A)
    p,q = size(A)
    if istopblock(A)
        y[:] = block_mv(N, p, q, x)
    else
        Dp, Dq, c = blockshift_top_to_sub(N, A.Ip, A.Iq, T)
        x1 = Dq*x/c
        y1 = block_mv(N, p, q, x1)
        y[:] = Dp*y1
    end
    y
end

"""
Compute the singular value corresponding to the given singular vectors of `A`.

The vectors are only accurate up to a complex phase shift. The routine returns
a positive, real singular value and adjusts the left singular vector accordingly.
"""
function compute_singular_value_and_correct(A::CenteredBlock, uk, vk)
    sigma0 = uk' * (A*vk)
    sigma = abs(sigma0)
    uk = uk * sigma0/sigma
    uk, sigma, vk
end

"Compute the singular value to go with the given right singular vector."
function compute_singular_value(A::AbstractArray, vk)
    uk = A*vk
    norm(uk)
end

function compute_singular_value(A::AbstractArray, uk, vk)
    # p,q = size(A)
    # # mid = max(1,min(p,q)>>1)
    # M,I = findmax(abs.(uk))
    # Z = sum(A[I,k]*vk[k] for k in 1:length(vk))
    # abs(Z/uk[I])
    abs(uk'*(A*vk))
end

function LinearAlgebra.svd(A::CenteredBlock{T}) where T
    N = dftlength(A)
    p,q = size(A)
    Pleft = DiscreteProlateMatrix{T}(N, q, p)
    Pright = DiscreteProlateMatrix{T}(N, p, q)
    U = complex(pdpss(Pleft))
    V = complex(pdpss(Pright))
    K = min(p,q)
    S = zeros(T,K)
    for k in 1:K
        U[:,k],S[k],vk = compute_singular_value_and_correct(A, U[:,k], V[:,k])
    end
    U,S,collect(V')'
end

function pdpss(A::CenteredBlock{T}, range) where T
    N = dftlength(A)
    p,q = size(A)
    Pleft = DiscreteProlateMatrix{T}(N, q, p)
    Pright = DiscreteProlateMatrix{T}(N, p, q)
    Vleft = pdpss(Pleft, range)
    Vright = pdpss(Pright, range)
    Vleft, Vright
end

function LinearAlgebra.svd(A::DFTBlock{T}) where T
    u,s,v = svd(centered(A))
    Dp, Dq, c = blockshift_center_to_sub(A.N, A.Ip, A.Iq, T)
    Dp*u/c, s, (v'*Dq)'
end

function pdpss(A::DFTBlock{T}, range) where T
    Vleft,Vright = pdpss(centered(A), range)
    Dp, Dq, c = blockshift_sub_to_center(A.N, A.Ip, A.Iq, T)
    Dp'*Vleft, Dq*Vright
end
