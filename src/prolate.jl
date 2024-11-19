
## Part I: Periodic discrete prolate sequences

"Estimate the size of the plunge region with the given parameters for a given threshold."
estimate_plunge_size(N, p, threshold) = estimate_plunge_size(N, p, p, threshold)

estimate_plunge_size(N, p, q, threshold) =
    max(1,min(round(Int, 10*log(p*q/N)),min(p,q)))

"A tridiagonal matrix that commutes with a periodic discrete prolate matrix."
function pdpss_tridiag_matrix(N, p, q, T = Float64)
    k_c = 0:q-1
    k_b = 0:(q-2)
    F = T(pi)/N

    c = cos.(F*(2*k_c .- (q-1))) * cos(F*p)
    b = -sin.(F*(k_b .+ 1)) .* sin.(F*(q-1 .- k_b))

    SymTridiagonal(c, b)
end


"""
Representation of a periodic discrete prolate (PDP) matrix.

The parameters are `N`, `p` and `q`. The matrix has size `q x q`. Elements are
`A[k,l] = sin(p*(k-l)*pi/N) / (p*sin((k-l)*pi/N))` when `k` differs from `l`,
and `1` when `k=l`.
"""
struct DiscreteProlateMatrix{T} <: AbstractMatrix{T}
    N   ::  Int                 # length of the underlying DFT
    p   ::  Int                 # frequency of the sinc numerator
    q   ::  Int                 # dimension of the PDP matrix
end

DiscreteProlateMatrix(N, p, q) = DiscreteProlateMatrix{Float64}(N, p, q)

dftlength(A::DiscreteProlateMatrix) = A.N
Base.size(A::DiscreteProlateMatrix) = (A.q, A.q)

function Base.getindex(A::DiscreteProlateMatrix{T}, k::Int, l::Int) where T
    checkbounds(A, k, l)
    pdpss_matrix_entry(A.N, A.p, A.q, k, l, T)
end

function pdpss_matrix_entry(N, p, q, k, l, ::Type{T} = Float64) where T
    F = T(pi)/N
    k == l ? one(T) : sin(F*p*(k-l)) / (p*sin(F*(k-l)))
end

"""
    pdpss(A::DiscreteProlateMatrix[, range])

Compute the periodic discrete prolate spheroidal sequences.

The PDPSS are computed from the eigenvalue decomposition of a commuting
tridiagonal matrix. Optionally, a range of indices can be supplied and only
the corresponding eigenvectors are computed.
"""
function pdpss(A::DiscreteProlateMatrix{T}) where T
    N = A.N; p = A.p; q = A.q
    J = pdpss_tridiag_matrix(N, p, q, T)
    E,V = eigen(J)
    V
end

function pdpss(A::DiscreteProlateMatrix{T}, range) where {T<:Base.IEEEFloat}
    N = A.N; p = A.p; q = A.q
    J = pdpss_tridiag_matrix(N, p, q, T)
    E,V = eigen(J, range)
    V
end

function pdpss(A::DiscreteProlateMatrix{T}, range) where T
    N = A.N; p = A.p; q = A.q
    J1 = pdpss_tridiag_matrix(N, p, q, Float64)
    E1,V1 = eigen(J1, range)
    E = similar(E1, T)
    V = similar(V1, T)
    J = pdpss_tridiag_matrix(N, p, q, T)
    for k in 1:length(E1)
        E[k],V[:,k] = refine_eigenvalue(J, E1[k], V1[:,k])
    end
    V
end

"Compute the periodic discrete prolate spheroidal sequences associated with the plunge region."
function pdpss_plunge(A::DiscreteProlateMatrix, threshold = eps(prectype(A)))
    N = A.N; p = A.p; q = A.q
    mid = ceil(Int, p*q/N)
    plunge_size = estimate_plunge_size(N, p, q, threshold)
    indices = max(1,mid-plunge_size>>1):min(N,mid+plunge_size>>1)
    V = pdpss(A, indices)
    V, indices
end


## Part II: Discrete prolate sequences


"A tridiagonal matrix that commutes with the discrete prolate matrix."
function prolate_tridiag_matrix(N, W)
    T = typeof(W)
    I = 0:N-1
    J = 1:N-1
    PI = T(pi)

    c = ((N-one(T))/2 .- I).^2 * cos(2*PI*W)
    b = one(T)/2 * J .* (N .- J)
    SymTridiagonal(c, b)
end


"""
Representation of a discrete prolate matrix with parameters `N` and `W`.

We use the original notation of Slepian. The prolate matrix has entries
`A[k,l] = sin(2πW*(k-l)) / (π*(k-l))`.
"""
struct ProlateMatrix{T} <: AbstractMatrix{T}
    N   ::  Int
    W   ::  T
end


Base.size(A::ProlateMatrix) = (A.N, A.N)

function Base.getindex(A::ProlateMatrix{T}, k::Int, l::Int) where T
    checkbounds(A, k, l)
    prolate_matrix_entry(A.N, A.W, k, l)
end

function prolate_matrix_entry(N, W, k, l)
    PI = one(W)*pi
    k == l ? 2W : sin(2*PI*W*(k-l)) / (PI*(k-l))
end

"""
    dpss(N::Int, W[, range])
    dpss(A::ProlateMatrix[, range])

Compute the discrete prolate spheroidal sequences.

The DPSS are computed from the eigenvalue decomposition of a commuting
tridiagonal matrix. Optionally, a range of indices can be supplied and only
the corresponding eigenvectors are computed.
"""
dpss(A::ProlateMatrix, args...) = dpss(A.N, A.W, args...)

function dpss(N::Int, W::AbstractFloat)
    J = prolate_tridiag_matrix(N, W)
    E,V = eigen(J)
    V[:,N:-1:1]
end

function dpss(N::Int, W::Base.IEEEFloat, range)
    J = prolate_tridiag_matrix(N, W)
    fliprange = (N+1-last(range)):(N+1-first(range))
    E,V = eigen(J, fliprange)
    V[:,end:-1:1]
end

function dpss(N::Int, W, range)
    J1 = prolate_tridiag_matrix(N, Float64(W))
    fliprange = (N+1-last(range)):(N+1-first(range))
    E1,V1 = eigen(J1, fliprange)
    T = typeof(W)
    E = similar(E1, T)
    V = similar(V1, T)
    J = prolate_tridiag_matrix(N, W)
    for k in 1:length(E1)
        E[k],V[:,k] = refine_eigenvalue(J, E1[k], V1[:,k])
    end
    V[:,end:-1:1]
end
