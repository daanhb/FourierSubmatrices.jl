

"Estimate the size of the plunge region with the given parameters for a given threshold."
estimate_plunge_size(L, N, threshold) = estimate_plunge_size(L, N, N, threshold)

estimate_plunge_size(L, M, N, threshold) =
    max(1,min(round(Int, 10*log(M*N/L)),N))

"Return the Jacobi matrix that commutes with a discrete prolate matrix."
function jacobi_prolate(L, M, N, T = Float64)
    k_c = 0:N-1
    k_b = 0:(N-2)
    Tpi = convert(T, pi)

    c = cos.(Tpi/L*(2*k_c .- (N-1))) * cos(Tpi*M/L)
    b = -sin.(Tpi/L*(k_b .+ 1)) .* sin.(Tpi/L*(N-1 .- k_b))

    SymTridiagonal(c, b)
end



"""
Representation of a periodic discrete prolate (PDP) matrix.

The parameters are `L`, `M` and `N`. The matrix has size `N x N`. Elements are
`A[i,j] = sin(M*(i-j)*pi/L) / (M*sin((i-j)*pi/L))` when `i` differs from `j`,
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
    pdpss_matrix_entry(A.L, A.M, A.N, k, l, prectype(A))
end

function pdpss_matrix_entry(L, M, N, k, l, ::Type{T} = Float64) where {T}
    Tpi = convert(T, pi)
    k == l ? one(T) : sin(M*(k-l)*Tpi/L) / (M*sin((k-l)*Tpi/L))
end

"Compute the periodic discrete prolate spheroidal sequences."
function pdpss(A::DiscreteProlateMatrix)
    L = A.L; M = A.M; N = A.N
    T = prectype(A)
    J = jacobi_prolate(L, M, N, T)
    E,V = eigen(J)
    V
end

"Compute the given range of periodic discrete prolate spheroidal sequences."
function pdpss_range(A::DiscreteProlateMatrix, range)
    L = A.L; M = A.M; N = A.N
    T = prectype(A)
    J = jacobi_prolate(L, M, N, T)
    E,V = eigen(J, range)
    V
end

"Compute the periodic discrete prolate spheroidal sequences associated with the plunge region."
function pdpss_plunge(A::DiscreteProlateMatrix, threshold = eps(prectype(A)))
    L = A.L; M = A.M; N = A.N
    mid = ceil(Int, M*N/L)
    plunge_size = estimate_plunge_size(L, M, N, threshold)
    indices = max(1,mid-plunge_size>>1):min(N,mid+plunge_size>>1)
    V = pdpss_range(A, indices)
    V, indices
end
