
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



"Supertype of discrete Fourier-type matrices."
abstract type AbstractFourierMatrix{T} <: AbstractMatrix{Complex{T}} end

"The DFT matrix of size `N × N`."
struct DFTMatrix{T} <: AbstractFourierMatrix{T}
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
struct iDFTMatrix{T} <: AbstractFourierMatrix{T}
    N       ::  Int
end

iDFTMatrix(N) = iDFTMatrix{Float64}(N)

dftlength(A::iDFTMatrix) = A.N
Base.size(A::iDFTMatrix) = (A.N, A.N)

function Base.getindex(A::iDFTMatrix, k::Int, l::Int)
    checkbounds(A, k, l)
    idft_entry(A.N, k, l, prectype(A))
end
