
function dft_entry(L, k, l, ::Type{T} = Float64) where {T}
    Tpi = convert(T, pi)
    exp(-2*Tpi*im*(k-1)*(l-1)/L)
end

function idft_entry(L, k, l, ::Type{T} = Float64) where {T}
    Tpi = convert(T, pi)
    exp(2*Tpi*im*(k-1)*(l-1)/L)/L
end


"The DFT matrix of length `L`."
struct DFTMatrix{T} <: AbstractMatrix{Complex{T}}
    L       ::  Int
end
DFTMatrix(L) = DFTMatrix{Float64}(L)
Base.size(A::DFTMatrix) = (A.L, A.L)
function Base.getindex(A::DFTMatrix, k::Int, l::Int)
    checkbounds(A, k, l)
    dft_entry(A.L, k, l, numtype(A))
end

"The inverseDFT matrix of length `L`."
struct iDFTMatrix{T} <: AbstractMatrix{Complex{T}}
    L       ::  Int
end
iDFTMatrix(L) = iDFTMatrix{Float64}(L)
Base.size(A::iDFTMatrix) = (A.L, A.L)
function Base.getindex(A::iDFTMatrix, k::Int, l::Int)
    checkbounds(A, k, l)
    idft_entry(A.L, k, l, numtype(A))
end




"""
Representation of a complex-symmetric subblock of the length `L` DFT matrix. The
block has dimensions `M x N`.

Upon construction, this representation computes and stores a set of prolate
vectors associated with the plunge region.
"""
struct CenteredDFTBlock{T} <: AbstractMatrix{Complex{T}}
    L   ::  Int
    M   ::  Int
    N   ::  Int
    U   ::  Matrix{T}
    S   ::  Diagonal{Complex{T},Vector{Complex{T}}}
    V   ::  Matrix{T}
    I   ::  UnitRange{Int}

    function CenteredDFTBlock{T}(L, M, N) where {T}
        U, S, V, I = centered_pdpss_plunge(L, M, N, T)
        new(L, M, N, U, S, V, I)
    end
end

CenteredDFTBlock(L, M, N) = CenteredDFTBlock{Float64}(L, M, N)

Base.size(A::CenteredDFTBlock) = (A.M,A.N)

function Base.getindex(A::CenteredDFTBlock, k::Int, l::Int)
    checkbounds(A, k, l)
    centered_dft_entry(A.L, A.M, A.N, k, l, numtype(A))
end

function centered_dft_entry(L, M, N, k, l, ::Type{T} = Float64) where {T}
    Tpi = convert(T, pi)
    pivot_N = (N - 1) / T(2)
    pivot_M = (M - 1) / T(2)
    exp(-2*Tpi*im * (k-1-pivot_M) * (l-1-pivot_N) / L)
end

function centered_pdpss(L, M, N, ::Type{T} = Float64) where {T}
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
    Tpi = convert(T, pi)
    twiddle = exp(2*Tpi*im/L)
    pivot_N = (N-1) / T(2)
    pivot_M = (M-1) / T(2)

    D_M = Diagonal(twiddle.^(pivot_N * (0:M-1)))
    D_N = Diagonal(twiddle.^(pivot_M * (0:N-1)))
    c = twiddle^(pivot_N * pivot_M)
    D_M, D_N, c
end


"""
Compute the map from a DFT subblock `B` with the given range to the top-left
block `A` with the same size. The relation is `A = D_M * B * D_N`.
"""
function blockshift_topleft(L, I_M, I_N, ::Type{T} = Float64) where {T}
    Tpi = convert(T, pi)
    twiddle = exp(2*Tpi*im/L)

    shift_N = first(I_N)-1
    if shift_N > 0
        D_M = Diagonal(twiddle.^(shift_N * (I_M .- 1)))
    else
        D_M = Diagonal(ones(Complex{T}, length(I_M)))
    end
    shift_M = first(I_M)-1
    N = length(I_N)
    if shift_M > 0
        D_N = Diagonal(twiddle.^(shift_M * (0:N-1)))
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
struct DFTBlock{T} <: AbstractMatrix{Complex{T}}
    L       ::  Int
    I_M     ::  UnitRange{Int}
    I_N     ::  UnitRange{Int}

    center  ::  CenteredDFTBlock{T}
    D_M     ::  Diagonal{Complex{T}, Vector{Complex{T}}}
    D_N     ::  Diagonal{Complex{T}, Vector{Complex{T}}}
    c       ::  Complex{T}
end

DFTBlock(L, I_M, I_N) = DFTBlock{Float64}(L, I_M, I_N)
function DFTBlock{T}(L, I_M, I_N) where {T}
    center = CenteredDFTBlock{T}(L, length(I_M), length(I_N))
    D_M, D_N, c = blockshift(L, I_M, I_N, T)
    DFTBlock{T}(L, I_M, I_N, center, D_M, D_N, c)
end

function DFTBlock{T}(L, I_M, I_N, center) where {T}
    D_M, D_N, c = blockshift(L, I_M, I_N, T)
    DFTBlock{T}(L, I_M, I_N, center, D_M, D_N, c)
end


Base.size(A::DFTBlock) = (length(A.I_M),length(A.I_N))
function Base.getindex(A::DFTBlock, k::Int, l::Int)
    checkbounds(A, k, l)
    dft_entry(A.L, A.I_M[k], A.I_N[l], numtype(A))
end

blockshift(A::DFTBlock) = blockshift(A.L, A.I_M, A.I_N, numtype(A))

pdpss(A::DFTBlock) = block_dft_pdpss(A.L, A.I_M, A.I_N, numtype(A))
pdpss_plunge(A::DFTBlock) = block_dft_pdpss_plunge(A.L, A.I_M, A.I_N, numtype(A))
