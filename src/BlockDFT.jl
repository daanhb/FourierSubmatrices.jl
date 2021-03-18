module BlockDFT

using LinearAlgebra, FFTW

export DFT_matrix, IDFT_matrix,
    DFTBlock,
    sinc_matrix,
    jacobi_prolate,
    CenteredDFTBlock,
    DFTBlock,
    pdpss,
    pdpss_range,
    pdpss_plunge


# TODO: fix this routine
estimate_plunge_size(N, L, threshold) =
    max(1,min(round(Int, 10*log(N^2/L)),N))

estimate_plunge_size(L, M, N, threshold) =
    max(1,min(round(Int, 9*log(M*N/L)),N))


numtype(A) = real(eltype(A))

"""
Representation of a periodic discrete prolate (PDP) matrix.

The parameters are `L`, `M` and `N`. The matrix has size `N x N`. Element `A[i,j]`
is given by `sin(M*(i-j)*pi/L) / (M*sin((i-j)*pi/L))` when `i` differs from `j`,
and `1` when `i=j`.
"""
struct DiscreteProlateMatrix{T} <: AbstractMatrix{T}
    L   ::  Int                 # length of the underlying DFT
    M   ::  Int                 # frequency of the sinc numerator
    N   ::  Int                 # dimension of the PDP matrix
end

DiscreteProlateMatrix(L, M, N) = DiscreteProlateMatrix{Float64}(L, M, N)

Base.size(A::DiscreteProlateMatrix) = (A.N,A.N)
function Base.getindex(A::DiscreteProlateMatrix, k::Int, l::Int)
    checkbounds(A, k, l)
    pdpss_matrix_entry(A.L, A.M, A.N, k, l, numtype(A))
end

function pdpss_matrix_entry(L, M, N, k, l, ::Type{T} = Float64) where {T}
    Tpi = convert(T, pi)
    k == l ? one(T) : sin(M*(k-l)*Tpi/L) / (M*sin((k-l)*Tpi/L))
end

"Compute the periodic discrete prolate spheroidal sequences."
function pdpss(A::DiscreteProlateMatrix)
    L = A.L; M = A.M; N = A.N
    T = numtype(A)
    J = jacobi_prolate(L, M, N, T)
    E,V = eigen(J)
    V
end

"Compute the given range of periodic discrete prolate spheroidal sequences."
function pdpss_range(A::DiscreteProlateMatrix, range)
    L = A.L; M = A.M; N = A.N
    T = numtype(A)
    J = jacobi_prolate(L, M, N, T)
    E,V = eigen(J, range)
    V
end

"Compute the periodic discrete prolate spheroidal sequences associated with the plunge region."
function pdpss_plunge(A::DiscreteProlateMatrix, threshold = eps(numtype(A)))
    L = A.L; M = A.M; N = A.N
    mid = round(Int, M*N/L)
    plunge_size = estimate_plunge_size(L, M, N, threshold)
    indices = max(1,mid-plunge_size>>1):min(N,mid+plunge_size>>1)
    V = pdpss_range(A, indices)
    V, indices
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
    S   ::  Vector{Complex{T}}
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
    U, S, V
end

function centered_pdpss_plunge(L, M, N, ::Type{T} = Float64) where {T}
    PV = DiscreteProlateMatrix{T}(L, M, N)
    PU = DiscreteProlateMatrix{T}(L, N, M)
    V, I = pdpss_plunge(PV)
    U = pdpss_range(PU, I)
    S = zeros(Complex{T}, length(I))
    mid = round(Int, M*N/L)
    for l in 1:length(I)
        S[l] = sum(centered_dft_entry(L, M, N, mid, k, T)*V[k,l] for k in 1:N) / U[mid,l]
    end
    U, S, V, I
end

function mv(A::CenteredDFTBlock, x)
    L = A.L; M = A.M; N = A.N
    # Let's deal with this case for now
    @assert N == M
    T = numtype(A)
    D_M, D_N, c = blockshift(L, 1:M, 1:N, T)

    v2 = A.V' * x
    u2 = A.U * (A.S .* v2)

    x13 = x - A.V * v2
    w13 = fft(D_N * x13)
    Q = round(Int, N*M/L)
    x1 = D_N' * ifft([w13[1:Q]; zeros(N-Q)])

    step = floor(Int, L/N)
    while (step*N > L || round(Int,L/step)*step != L)
        step -= 1
        if step == 0
            error("L should be divisible by a number greater than 1")
        end
    end
    N2 = round(Int, L/step)
    x1_e = [D_N * x1; zeros(N2-N)]
    t1 = fft(x1_e)
    t2 = zeros(Complex{T}, N2)
    t2[1:step:step*Q] = t1[1:Q]
    z1 = ifft(t2)
    z1[Q+1:end] .= 0
    u1 = D_M * fft(z1)[1:N]*step / c

    u1 + u2
end

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
Base.size(A::DFTMatrix) = (A.L, A.L)
function Base.getindex(A::DFTMatrix, k::Int, l::Int)
    checkbounds(A, k, l)
    dft_entry(A.L, k, l, numtype(A))
end

"The inverseDFT matrix of length `L`."
struct iDFTMatrix{T} <: AbstractMatrix{Complex{T}}
    L       ::  Int
end
Base.size(A::iDFTMatrix) = (A.L, A.L)
function Base.getindex(A::iDFTMatrix, k::Int, l::Int)
    checkbounds(A, k, l)
    idft_entry(A.L, k, l, numtype(A))
end


function blockshift(L, I_M, I_N, ::Type{T} = Float64) where {T}
    Tpi = convert(T, pi)
    pivot_N = (last(I_N)-first(I_N)) / T(2)
    pivot_M = (last(I_M)-first(I_M)) / T(2)
    twiddle = exp(2*Tpi*im/L)
    D_M = Diagonal(twiddle.^(pivot_N * (I_M .- 1)))
    D_N = Diagonal(twiddle.^(pivot_M * (I_N .- 1)))
    c = exp(2*Tpi*im * pivot_M * pivot_N / L)
    D_M, D_N, c
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
end

DFTBlock(L, I_M, I_N) = DFTBlock{Float64}(L, I_M, I_N)

Base.size(A::DFTBlock) = (length(A.I_M),length(A.I_N))
function Base.getindex(A::DFTBlock, k::Int, l::Int)
    checkbounds(A, k, l)
    dft_entry(A.L, A.I_M[k], A.I_N[l], numtype(A))
end

blockshift(A::DFTBlock) = blockshift(A.L, A.I_M, A.I_N, numtype(A))

pdpss(A::DFTBlock) = block_dft_pdpss(A.L, A.I_M, A.I_N, numtype(A))
pdpss_plunge(A::DFTBlock) = block_dft_pdpss_plunge(A.L, A.I_M, A.I_N, numtype(A))


"A subblock of the length `L` iDFT matrix."
struct iDFTBlock{T} <: AbstractMatrix{Complex{T}}
    L       ::  Int
    I_M     ::  UnitRange{Int}
    I_N     ::  UnitRange{Int}
end

iDFTBlock(L, I_M, I_N) = iDFTBlock{Float64}(L, I_M, I_N)

Base.size(A::iDFTBlock) = (length(A.I_M),length(A.I_N))
function Base.getindex(A::iDFTBlock, k::Int, l::Int)
    checkbounds(A, k, l)
    idft_entry(A.L, A.I_M[k], A.I_N[l], numtype(A))
end



DFT_matrix(L, ::Type{T} = Float64) where {T} = collect(DFTMatrix{T}(L))
IDFT_matrix(L, ::Type{T} = Float64) where {T} = collect(iDFTMatrix{T}(L))

function sinc_matrix(L, N, T = Float64)
    A = zeros(T,N,N)
    Tpi = convert(T, pi)
    for i in 1:N
        for j in 1:N
            if i==j
                A[i,j] = 1;
            else
                A[i,j] = sin(N*(i-j)*Tpi/L) / (N*sin((i-j)*Tpi/L))
            end
        end
    end
    A
end

function jacobi_prolate(L, M, N, T = Float64)
    k_c = 0:N-1
    k_b = 0:(N-2)
    Tpi = convert(T, pi)

    c = cos.(Tpi/L*(2*k_c .- (N-1))) * cos(Tpi*M/L)
    b = -sin.(Tpi/L*(k_b .+ 1)) .* sin.(Tpi/L*(N-1 .- k_b))

    SymTridiagonal(c, b)
end


end # module
