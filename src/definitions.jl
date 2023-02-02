
"""
The twiddle factor associated with `N` is the complex root of unity
`ω = exp(2πi/N)`.
"""
twiddle(N, ::Type{T} = Float64) where T = exp(2*convert(T, pi)*im/N)

"The `(k,l)` entry of the `N × N` DFT matrix."
function dft_entry(N, k, l, ::Type{T} = Float64) where T
    ω = twiddle(N, T)
    ω^(-(k-1)*(l-1))
end

"The `(k,l)` entry of the `N × N` inverse DFT matrix."
function idft_entry(N, k, l, ::Type{T} = Float64) where T
    ω = twiddle(N, T)
    ω^((k-1)*(l-1))/N
end

"The `(k,l)` entry of a centered DFT subblock of size `p × q`."
function centered_dft_entry(N, p, q, k, l, ::Type{T} = Float64) where T
    pivot_q = (q - 1) / T(2)
    pivot_p = (p - 1) / T(2)
    ω = twiddle(N, T)
    ω^(-(k-1-pivot_p)*(l-1-pivot_q))
end



"Supertype of discrete Fourier-type matrices."
abstract type FourierMatrix{T} <: AbstractMatrix{Complex{T}} end

"The DFT matrix of size `N × N`."
struct DFTMatrix{T} <: FourierMatrix{T}
    N       ::  Int
end

DFTMatrix(N) = DFTMatrix{Float64}(N)

dftlength(A::DFTMatrix) = A.N
Base.size(A::DFTMatrix) = (A.N, A.N)

function Base.getindex(A::DFTMatrix, k::Int, l::Int)
    checkbounds(A, k, l)
    dft_entry(A.N, k, l, prectype(A))
end


"The inverseDFT of size `N × N`."
struct iDFTMatrix{T} <: FourierMatrix{T}
    N       ::  Int
end

iDFTMatrix(N) = iDFTMatrix{Float64}(N)

dftlength(A::iDFTMatrix) = A.N
Base.size(A::iDFTMatrix) = (A.N, A.N)

function Base.getindex(A::iDFTMatrix, k::Int, l::Int)
    checkbounds(A, k, l)
    idft_entry(A.N, k, l, prectype(A))
end


"""
Representation of a complex-symmetric subblock of the length `N` DFT matrix. The
block has dimensions `p × q`.
"""
struct CenteredBlock{T} <: FourierMatrix{T}
    N   ::  Int
    p   ::  Int
    q   ::  Int

    function CenteredBlock{T}(N, p, q) where T
        @assert 1 <= p <= N
        @assert 1 <= q <= N
        new(N, p, q)
    end
end

CenteredBlock(N, p, q) = CenteredBlock{Float64}(N, p, q)

dftlength(A::CenteredBlock) = A.N
Base.size(A::CenteredBlock) = (A.p,A.q)

function Base.getindex(A::CenteredBlock, k::Int, l::Int)
    checkbounds(A, k, l)
    centered_dft_entry(A.N, A.p, A.q, k, l, prectype(A))
end


"""
A subblock of the `N × N` DFT matrix.

The row and column selection are determined by unit ranges.
"""
struct DFTBlock{T} <: FourierMatrix{T}
    N       ::  Int
    Ip      ::  UnitRange{Int}  # row range
    Iq      ::  UnitRange{Int}  # column range

    function DFTBlock{T}(N, Ip, Iq) where T
        @assert length(Ip) <= N
        @assert length(Iq) <= N
        new(N, Ip, Iq)
    end
end

DFTBlock(N, P, Q) = DFTBlock{Float64}(N, P, Q)

DFTBlock{T}(N::Int, p::Int, q::Int) where T = DFTBlock{T}(N, 1:p, 1:q)

dftlength(A::DFTBlock) = A.N
Base.size(A::DFTBlock) = (length(A.Ip),length(A.Iq))

function Base.getindex(A::DFTBlock{T}, k::Int, l::Int) where T
    checkbounds(A, k, l)
    dft_entry(A.N, A.Ip[k], A.Iq[l], T)
end

function centered(A::DFTBlock{T}) where T
    N = dftlength(A)
    p,q = size(A)
    CenteredBlock{T}(N, p, q)
end
