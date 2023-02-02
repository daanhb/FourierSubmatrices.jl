

"Estimate the size of the plunge region with the given parameters for a given threshold."
estimate_plunge_size(N, p, threshold) = estimate_plunge_size(N, p, p, threshold)

estimate_plunge_size(N, p, q, threshold) =
    max(1,min(round(Int, 10*log(p*q/N)),min(p,q)))

"Return the Jacobi matrix that commutes with a discrete prolate matrix."
function jacobi_prolate(N, p, q, T = Float64)
    k_c = 0:q-1
    k_b = 0:(q-2)
    F = T(pi)/N

    c = cos.(F*(2*k_c .- (q-1))) * cos(F*p)
    b = -sin.(F*(k_b .+ 1)) .* sin.(F*(q-1 .- k_b))

    SymTridiagonal(c, b)
end


"""
Representation of a periodic discrete prolate (PDP) matrix.

The parameters are `L`, `M` and `N`. The matrix has size `N x N`. Elements are
`A[i,j] = sin(M*(i-j)*pi/L) / (M*sin((i-j)*pi/L))` when `i` differs from `j`,
and `1` when `i=j`.
"""
struct DiscreteProlateMatrix{T} <: AbstractMatrix{T}
    N   ::  Int                 # length of the underlying DFT
    p   ::  Int                 # frequency of the sinc numerator
    q   ::  Int                 # dimension of the PDP matrix
end

DiscreteProlateMatrix(N, p, q) = DiscreteProlateMatrix{Float64}(N, p, q)

dftlength(A::DiscreteProlateMatrix) = A.N
Base.size(A::DiscreteProlateMatrix) = (A.q,A.q)

function Base.getindex(A::DiscreteProlateMatrix{T}, k::Int, l::Int) where T
    checkbounds(A, k, l)
    pdpss_matrix_entry(A.N, A.p, A.q, k, l, T)
end

function pdpss_matrix_entry(N, p, q, k, l, ::Type{T} = Float64) where T
    F = T(pi)/N
    k == l ? one(T) : sin(F*p*(k-l)) / (p*sin(F*(k-l)))
end

"Compute the periodic discrete prolate spheroidal sequences."
function pdpss(A::DiscreteProlateMatrix{T}) where T
    N = A.N; p = A.p; q = A.q
    J = jacobi_prolate(N, p, q, T)
    E,V = eigen(J)
    V
end

"Compute the given range of periodic discrete prolate spheroidal sequences."
function pdpss_range(A::DiscreteProlateMatrix{T}, range) where T
    N = A.N; p = A.p; q = A.q
    J = jacobi_prolate(N, p, q, T)
    E,V = eigen(J, range)
    V
end

"Compute the periodic discrete prolate spheroidal sequences associated with the plunge region."
function pdpss_plunge(A::DiscreteProlateMatrix, threshold = eps(prectype(A)))
    N = A.N; p = A.p; q = A.q
    mid = ceil(Int, p*q/N)
    plunge_size = estimate_plunge_size(N, p, q, threshold)
    indices = max(1,mid-plunge_size>>1):min(N,mid+plunge_size>>1)
    V = pdpss_range(A, indices)
    V, indices
end
