
"""
Compute the diagonal scaling matrix of length `p` associated with the DFT
of length `N`, raised to the power `l`.
"""
function dft_diagonal_scaling(N, p, l = 1, ::Type{T} = Float64) where T
    omega = twiddle(N, T)
    Diagonal([omega^(-(j-1)*l) for j in 1:p])
end

"""
Representation of a complex-symmetric subblock of the length `L` DFT matrix. The
block has dimensions `M x N`.

Upon construction, this representation computes and stores a set of prolate
vectors associated with the plunge region.
"""
struct CenteredDFTPlan{T} <: AbstractMatrix{Complex{T}}
    L   ::  Int
    M   ::  Int
    N   ::  Int
    U   ::  Matrix{T}
    S   ::  Diagonal{Complex{T},Vector{Complex{T}}}
    V   ::  Matrix{T}
    I   ::  UnitRange{Int}

    function CenteredDFTPlan{T}(L, M, N) where {T}
        U, S, V, I = centered_pdpss_plunge(L, M, N, T)
        new(L, M, N, U, S, V, I)
    end
end

CenteredDFTPlan(L, M, N) = CenteredDFTPlan{Float64}(L, M, N)

Base.size(A::CenteredDFTPlan) = (A.M,A.N)

function Base.getindex(A::CenteredDFTPlan, k::Int, l::Int)
    checkbounds(A, k, l)
    centered_dft_entry(A.L, A.M, A.N, k, l, prectype(A))
end

function centered_pdpss(L, M, N, ::Type{T} = Float64) where T
    PV = DiscreteProlateMatrix{T}(L, M, N)
    PU = DiscreteProlateMatrix{T}(L, N, M)
    V = pdpss(PV)
    U = pdpss(PU)
    S = zeros(Complex{T}, min(N,M))
    if N < M
        U = U[:,1:N]
    elseif N > M
        V = V[:,1:M]
    end
    mid = round(Int, M*N/L)
    for l in 1:min(N,M)
        if abs(U[mid,l]) > 1e-3
            S[l] = sum(centered_dft_entry(L, M, N, mid, k, T)*V[k,l] for k in 1:N) / U[mid,l]
        else
            idx = findfirst(abs.(U[:,l]) .> 1e-3)
            S[l] = sum(centered_dft_entry(L, M, N, idx, k, T)*V[k,l] for k in 1:N) / U[idx,l]
        end
    end
    U, Diagonal(S), V
end

function centered_pdpss_plunge(L, M, N, ::Type{T} = Float64) where {T}
    PV = DiscreteProlateMatrix{T}(L, M, N)
    PU = DiscreteProlateMatrix{T}(L, N, M)
    V, I = pdpss_plunge(PV)
    U = pdpss_range(PU, I)
    S = zeros(Complex{T}, length(I))
    mid = ceil(Int, M*N/L)
    for l in 1:length(I)
        S[l] = sum(centered_dft_entry(L, M, N, mid, k, T)*V[k,l] for k in 1:N) / U[mid,l]
    end
    U, Diagonal(S), V, I
end


"""
Compute the map from the topleft `M x N` subblock `B = A[1:M,1:N]` of the
`L x L` DFT matrix `A`, to the centered block `C` with the same size as `B`.

The relation is given by `C = D_M * B * D_N / c`.
"""
function blockshift_center(L, M, N, ::Type{T} = Float64) where {T}
    ω = twiddle(L, T)
    pivot_N = (N-1) / T(2)
    pivot_M = (M-1) / T(2)

    D_M = Diagonal(ω.^(pivot_N * (0:M-1)))
    D_N = Diagonal(ω.^(pivot_M * (0:N-1)))
    c = ω^(pivot_N * pivot_M)
    D_M, D_N, c
end


"""
Compute the map from a DFT subblock `B` with the given range to the top-left
block `A` with the same size. The relation is `A = D_M * B * D_N`.
"""
function blockshift_topleft(L, I_M, I_N, ::Type{T} = Float64) where {T}
    ω = twiddle(L, T)

    shift_N = first(I_N)-1
    if shift_N > 0
        D_M = Diagonal(ω.^(shift_N * (I_M .- 1)))
    else
        D_M = Diagonal(ones(Complex{T}, length(I_M)))
    end
    shift_M = first(I_M)-1
    N = length(I_N)
    if shift_M > 0
        D_N = Diagonal(ω.^(shift_M * (0:N-1)))
    else
        D_N = Diagonal(ones(Complex{T}, length(I_N)))
    end
    D_M, D_N
end

"""
Compute the map from a block with the given range to a centered block with the
same size.
"""
function blockshift(L, I_M, I_N, ::Type{T} = Float64) where {T}
    Dtl_M, Dtl_N = blockshift_topleft(L, I_M, I_N, T)
    Dc_M, Dc_N, c = blockshift_center(L, length(I_M), length(I_N))
    Dc_M*Dtl_M, Dc_N*Dtl_N, c
end


function block_dft_pdpss(L, I_M, I_N, ::Type{T} = Float64) where {T}
    U, S, V = centered_pdpss(L, length(I_M), length(I_N), T)
    D_M, D_N, c = blockshift(L, I_M, I_N, T)
    U, S, V, D_M, D_N, c
end

function block_dft_pdpss_plunge(L, I_M, I_N, ::Type{T} = Float64) where {T}
    U, S, V, I = centered_pdpss_plunge(L, length(I_M), length(I_N), T)
    D_M, D_N, c = blockshift(L, I_M, I_N, T)
    U, S, V, D_M, D_N, c, I
end


"A subblock of the length `L` DFT matrix."
struct DFTBlockPlan{T} <: AbstractMatrix{Complex{T}}
    L       ::  Int
    I_M     ::  UnitRange{Int}
    I_N     ::  UnitRange{Int}

    center_block    ::  CenteredDFTPlan{T}
    D_M             ::  Diagonal{Complex{T}, Vector{Complex{T}}}
    D_N             ::  Diagonal{Complex{T}, Vector{Complex{T}}}
    c               ::  Complex{T}
end

DFTBlockPlan(L, I_M, I_N) = DFTBlockPlan{Float64}(L, I_M, I_N)

DFTBlockPlan{T}(L::Int, p::Int, q::Int) where {T} = DFTBlockPlan(L, 1:p, 1:q)
function DFTBlockPlan{T}(L, I_M::UnitRange, I_N::UnitRange) where {T}
    center_block = CenteredDFTPlan{T}(L, length(I_M), length(I_N))
    D_M, D_N, c = blockshift(L, I_M, I_N, T)
    DFTBlockPlan{T}(L, I_M, I_N, center_block, D_M, D_N, c)
end

function DFTBlockPlan{T}(L, I_M, I_N, center) where {T}
    D_M, D_N, c = blockshift(L, I_M, I_N, T)
    DFTBlockPlan{T}(L, I_M, I_N, center, D_M, D_N, c)
end


Base.size(A::DFTBlockPlan) = (length(A.I_M),length(A.I_N))
function Base.getindex(A::DFTBlockPlan, k::Int, l::Int)
    checkbounds(A, k, l)
    dft_entry(A.L, A.I_M[k], A.I_N[l], prectype(A))
end

blockshift(A::DFTBlockPlan) = blockshift(A.L, A.I_M, A.I_N, prectype(A))

pdpss(A::DFTBlockPlan) = block_dft_pdpss(A.L, A.I_M, A.I_N, prectype(A))
pdpss_plunge(A::DFTBlockPlan) = block_dft_pdpss_plunge(A.L, A.I_M, A.I_N, prectype(A))

function LinearAlgebra.svd(A::DFTBlockPlan)
    T = prectype(A)
    N = A.L
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
