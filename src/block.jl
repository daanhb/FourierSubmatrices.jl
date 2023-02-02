
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

function LinearAlgebra.svd(A::DFTBlock{T}) where T
    u,s,v = svd(centered(A))
    Dp, Dq, c = blockshift_center_to_sub(A.N, A.Ip, A.Iq, T)
    Dp*u/c, s, (v'*Dq)'
end

"""
Representation of a complex-symmetric subblock of the length `N` DFT matrix. The
block has dimensions `p × q`.

Upon construction, this representation computes and stores a set of prolate
vectors associated with the plunge region.
"""
struct CenteredDFTPlan{T} <: AbstractMatrix{Complex{T}}
    N   ::  Int
    p   ::  Int
    q   ::  Int
    U   ::  Matrix{T}
    S   ::  Diagonal{Complex{T},Vector{Complex{T}}}
    V   ::  Matrix{T}
    I   ::  UnitRange{Int}

    function CenteredDFTPlan{T}(N, p, q) where {T}
        U, S, V, I = centered_pdpss_plunge(N, p, q, T)
        new(N, p, q, U, S, V, I)
    end
end

CenteredDFTPlan(N, p, q) = CenteredDFTPlan{Float64}(N, p, q)

Base.size(A::CenteredDFTPlan) = (A.p,A.q)

function Base.getindex(A::CenteredDFTPlan, k::Int, l::Int)
    checkbounds(A, k, l)
    centered_dft_entry(A.N, A.p, A.q, k, l, prectype(A))
end

function centered_pdpss(N, p, q, ::Type{T} = Float64) where T
    PV = DiscreteProlateMatrix{T}(N, p, q)
    PU = DiscreteProlateMatrix{T}(N, p, q)
    V = pdpss(PV)
    U = pdpss(PU)
    S = zeros(Complex{T}, min(p,q))
    if q < p
        U = U[:,1:q]
    elseif q > p
        V = V[:,1:p]
    end
    mid = round(Int, p*q/N)
    for l in 1:min(p,q)
        if abs(U[mid,l]) > 1e-3
            S[l] = sum(centered_dft_entry(N, p, q, mid, k, T)*V[k,l] for k in 1:q) / U[mid,l]
        else
            idx = findfirst(abs.(U[:,l]) .> 1e-3)
            S[l] = sum(centered_dft_entry(N, p, q, idx, k, T)*V[k,l] for k in 1:q) / U[idx,l]
        end
    end
    U, Diagonal(S), V
end

function centered_pdpss_plunge(N, p, q, ::Type{T} = Float64) where {T}
    PV = DiscreteProlateMatrix{T}(N, p, q)
    PU = DiscreteProlateMatrix{T}(N, p, q)
    V, I = pdpss_plunge(PV)
    U = pdpss(PU, I)
    S = zeros(Complex{T}, length(I))
    mid = ceil(Int, p*q/N)
    for l in 1:length(I)
        S[l] = sum(centered_dft_entry(N, p, q, mid, k, T)*V[k,l] for k in 1:q) / U[mid,l]
    end
    U, Diagonal(S), V, I
end



"""
Compute the map from a DFT subblock `B` with the given range to the top-left
block `A` with the same size. The relation is `A = Dp * B * Dq`.
"""
function blockshift_topleft(N, Ip, Iq, ::Type{T} = Float64) where {T}
    ω = twiddle(N, T)

    shift_N = first(Iq)-1
    if shift_N > 0
        Dp = Diagonal(ω.^(shift_N * (Ip .- 1)))
    else
        Dp = Diagonal(ones(Complex{T}, length(Ip)))
    end
    shift_M = first(Ip)-1
    N = length(Iq)
    if shift_M > 0
        Dq = Diagonal(ω.^(shift_M * (0:N-1)))
    else
        Dq = Diagonal(ones(Complex{T}, length(Iq)))
    end
    Dp, Dq
end

"""
Compute the map from a block with the given range to a centered block with the
same size.
"""
function blockshift(N, Ip, Iq, ::Type{T} = Float64) where {T}
    Dtl_M, Dtl_N = blockshift_topleft(N, Ip, Iq, T)
    Dc_M, Dc_N, c = blockshift_top_to_center(N, length(Ip), length(Iq))
    Dc_M*Dtl_M, Dc_N*Dtl_N, c
end


function block_dft_pdpss(N, Ip, Iq, ::Type{T} = Float64) where {T}
    U, S, V = centered_pdpss(N, length(Ip), length(Iq), T)
    Dp, Dq, c = blockshift(N, Ip, Iq, T)
    U, S, V, Dp, Dq, c
end

function block_dft_pdpss_plunge(N, Ip, Iq, ::Type{T} = Float64) where {T}
    U, S, V, I = centered_pdpss_plunge(N, length(Ip), length(Iq), T)
    Dp, Dq, c = blockshift(N, Ip, Iq, T)
    U, S, V, Dp, Dq, c, I
end


"A subblock of the length `N` DFT matrix."
struct DFTBlockPlan{T} <: AbstractMatrix{Complex{T}}
    N       ::  Int
    Ip      ::  UnitRange{Int}
    Iq      ::  UnitRange{Int}

    center_block    ::  CenteredDFTPlan{T}
    Dp              ::  Diagonal{Complex{T}, Vector{Complex{T}}}
    Dq              ::  Diagonal{Complex{T}, Vector{Complex{T}}}
    c               ::  Complex{T}
end

DFTBlockPlan(N, Ip, Iq) = DFTBlockPlan{Float64}(N, Ip, Iq)

DFTBlockPlan{T}(N::Int, p::Int, q::Int) where {T} = DFTBlockPlan(N, 1:p, 1:q)
function DFTBlockPlan{T}(N, Ip::UnitRange, Iq::UnitRange) where {T}
    center_block = CenteredDFTPlan{T}(N, length(Ip), length(Iq))
    Dp, Dq, c = blockshift(N, Ip, Iq, T)
    DFTBlockPlan{T}(N, Ip, Iq, center_block, Dp, Dq, c)
end

function DFTBlockPlan{T}(N, Ip, Iq, center) where {T}
    Dp, Dq, c = blockshift(N, Ip, Iq, T)
    DFTBlockPlan{T}(N, Ip, Iq, center, Dp, Dq, c)
end


Base.size(A::DFTBlockPlan) = (length(A.Ip),length(A.Iq))
function Base.getindex(A::DFTBlockPlan, k::Int, l::Int)
    checkbounds(A, k, l)
    dft_entry(A.N, A.Ip[k], A.Iq[l], prectype(A))
end

blockshift(A::DFTBlockPlan) = blockshift(A.N, A.Ip, A.Iq, prectype(A))

pdpss(A::DFTBlockPlan) = block_dft_pdpss(A.N, A.Ip, A.Iq, prectype(A))
pdpss_plunge(A::DFTBlockPlan) = block_dft_pdpss_plunge(A.N, A.Ip, A.Iq, prectype(A))

function LinearAlgebra.svd(A::DFTBlockPlan)
    T = prectype(A)
    N = A.N
    p,q = size(A)
    dft_submatrix_svd(N, p, q, T)
end

function dft_submatrix_svd(N, p, q, ::Type{T} = Float64) where T
    J1 = jacobi_prolate(N, p, q, T)
    J2 = jacobi_prolate(N, q, p, T)
    D_p = dft_diagonal_scaling(N, p, (q-1)/2, T)
    D_q = dft_diagonal_scaling(N, q, -(p-1)/2, T)
    omega = twiddle(N, T)
    E1,V1 = eigen(J1)
    E2,V2 = eigen(J2)
    K = min(p,q)
    V = zeros(Complex{T},q,K)
    for i in 1:K
        V[:,i] = omega^(-(p-1)/2*(q-1)/2) * D_q*V1[:,i]
    end
    U = zeros(Complex{T},p,K)
    for i in 1:K
        U[:,i] = D_p*V2[:,i]
    end
    S = diag(U' * A * V)
    factors = abs.(S) ./ S
    s = abs.(S)
    for i in 1:K
        U[:,i] = U[:,i] / factors[i]
    end
    U, s, collect(V')'
end

function dft_submatrix_svd_with_fft(N, p, q, T = Float64)
    J1 = jacobi_prolate(N, p, q, T)
    D_q = diagonal_scaling(N, q, -(p-1)/2, T)
    E1,V1 = eigen(J1)
    omega = twiddle(N, T)
    K = min(p,q)
    V = zeros(Complex{T},q,K)
    U = zeros(Complex{T},p,K)
    S = zeros(T, K)
    for i in 1:K
        V[:,i] = omega^(-(p-1)/2*(q-1)/2) * D_q*V1[:,i]
        uu = fft([V[:,i]; zeros(Complex{T}, N-q)])[1:p]
        s = norm(uu)
        U[:,i] = uu / s
        S[i] = s
    end
    U, S, collect(V')'
end
