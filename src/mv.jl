function mv(A::CenteredDFTPlan, x)
    L = A.L; M = A.M; N = A.N
    # Let's deal with this case for now
    @assert N == M
    T = prectype(A)
    D_M, D_N, c = blockshift(L, 1:M, 1:N, T)

    v2 = A.V' * x
    u2 = A.U * (A.S * v2)

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

# In the regular variant we assume that L is a multiple of p
function mv_regular(A::CenteredDFTPlan, x)
    L = A.L; M = A.M; N = A.N
    p = round(Int, L/N)
    # Let's deal with this case for now
    @assert N == M
    @assert N*p == L
    T = prectype(A)
    D_M, D_N, c = blockshift(L, 1:M, 1:N, T)

    v2 = A.V' * x
    u2 = A.U * (A.S * v2)

    x13 = x - A.V * v2
    w13 = fft(D_N * x13)
    Q = round(Int, N*M/L)
    t2 = zeros(Complex{T}, N)
    t2[1:p:N] = w13[1:Q]
    z1 = ifft(t2)
    z1[Q+1:end] .= 0
    u1 = D_M * fft(z1)[1:N]*p / c

    u1 + u2
end


function mv(A::DFTBlockPlan, x)
    x_shift = conj(A.D_N) * x
    y_shift = mv(A.center_block, x_shift)
    y = conj(A.D_M) * y_shift * A.c
end

function mv_regular(A::DFTBlockPlan, x)
    x_shift = conj(A.D_N) * x
    y_shift = mv_regular(A.center_block, x_shift)
    y = conj(A.D_M) * y_shift * A.c
end


function mv!(y, A::DFTBlockPlan, x, D_M1, D_N1)
    T = eltype(x); RT = real(T); CT = Complex{RT}
    N = length(A.I_N)
    M = length(A.I_M)
    @assert M==N
    L = A.L
    p = round(Int, L/N)
    Q = round(Int, N/p)
    center = A.center_block
    block_D_M = A.D_M
    block_D_N = A.D_N
    c = A.c
    U2 = center.U
    S2 = center.S
    V2 = center.V
    I2 = center.I

    x_shift = conj(block_D_N) * x
    v2 = V2' * x_shift
    u2 = U2 * (S2 * v2)

    x13 = x_shift - V2 * v2
    w13 = fft(D_N1 * x13)
    t2 = zeros(CT, N)
    t2[1:p:N] = w13[1:Q]
    z1 = ifft(t2)
    z1[Q+1:end] .= 0
    u1 = D_M1 * fft(z1)[1:N]*p / c

    y[:] = conj(block_D_M) * (u1+u2) * c
end

"""
Extend the first `Q` samples of x to a vector of length `N` by upsampling, followed
by an ideal low-pass filter (in the frequency domain).
"""
function upsample!(y, x, Q, p, FFT!, IFFT!)
    fill!(y, 0)
    # y[1:p:N] .= x[1:Q]
    for k in 1:Q
        y[1+(k-1)*p] = x[k]
    end
    IFFT! * y
    y[Q+1:end] .= 0
    FFT! * y
end


# Results are returned in y_reduced and t_plunge
function mv!_part1(y_reduced, A::DFTBlockPlan, x, D_N1, t1, t_plunge, FFT!)
    Q = length(y_reduced)
    block_D_N = A.D_N
    center = A.center_block
    S2 = center.S
    V2 = center.V

    # t1 = conj(block_D_N) * x
    # division avoids having to take conjugates here
    ldiv!(t1, block_D_N, x)
    # t_plunge = V2' * t1
    mul!(t_plunge, V2', t1)
    # t1 = t1 - V2 * t_plunge
    mul!(t1, V2, t_plunge, -1, 1)
    # t_plunge = S2 * t_plunge
    mul!(t_plunge, S2, t_plunge)   # aliasing t_plunge is ok because S2 is diagonal
    # t1 = D_N1 * t1
    mul!(t1, D_N1, t1)
    FFT! * t1
    for i in 1:Q
        y_reduced[i] = t1[i]
    end
    nothing
end


function mv!_part2(y, A::DFTBlockPlan, y_reduced, D_M1, t1, t2, t3, t_plunge, FFT!, IFFT!)
    L = A.L
    N = length(A.I_N)
    p = round(Int, L/N)
    Q = length(y_reduced)
    block_D_M = A.D_M
    c = A.c
    center = A.center_block
    U2 = center.U

    # t3 = U2 * t_plunge
    mul!(t3, U2, t_plunge)
    upsample!(t2, t1, Q, p, FFT!, IFFT!)
    # t2 = D_M1 * t2
    mul!(t2, D_M1, t2)
    # t3 = c*t3 + p*t2
    axpby!(p, t2, c, t3)
    # y = conj(block_D_M) * t3
    ldiv!(y, block_D_M, t3)
end


"A subblock of the length `L` iDFT matrix."
struct iDFTBlockPlan{T} <: AbstractMatrix{Complex{T}}
    L       ::  Int
    I_M     ::  UnitRange{Int}
    I_N     ::  UnitRange{Int}
end

iDFTBlockPlan(L, I_M, I_N) = iDFTBlockPlan{Float64}(L, I_M, I_N)

Base.size(A::iDFTBlockPlan) = (length(A.I_M),length(A.I_N))
function Base.getindex(A::iDFTBlockPlan, k::Int, l::Int)
    checkbounds(A, k, l)
    idft_entry(A.L, A.I_M[k], A.I_N[l], prectype(A))
end



DFT_matrix(L, ::Type{T} = Float64) where {T} = collect(DFTMatrix{T}(L))
IDFT_matrix(L, ::Type{T} = Float64) where {T} = collect(iDFTMatrix{T}(L))


function blockdft_blocks(N, p, ::Type{T} = Float64) where {T}
    L = N*p
    cb = CenteredDFTPlan{T}(L, N, N)
    dftblocks = [DFTBlockPlan{T}(L, (k-1)*N+1:k*N, (l-1)*N+1:l*N, cb) for k in 1:p, l in 1:p]
end


function blockdft(L, p, x::AbstractVector)
    N = round(Int, L/p)
    @assert N*p == L

    T = eltype(x)
    RT = real(T)
    CT = Complex{RT}

    dftblocks = blockdft_blocks(N, p, RT)
    x_blocks = [x[(k-1)*N+1:k*N] for k in 1:p]
    y_blocks = [zeros(CT, N) for k in 1:p]
    for k in 1:p
        for l in 1:p
            y_blocks[k] += mv_regular(dftblocks[k,l], x_blocks[l])
        end
    end
    y = vcat(y_blocks...)
end


abstract type DFTBlockPlanArray{T} <: AbstractBlockMatrix{T} end

const FFTPLAN{RT} = FFTW.cFFTWPlan{Complex{RT},-1,true,1,UnitRange{Int64}}
const IFFTPLAN{RT} = AbstractFFTs.ScaledPlan{Complex{RT},FFTW.cFFTWPlan{Complex{RT},1,true,1,UnitRange{Int64}},RT}

struct RegularDFTBlockPlanArray{T,RT} <: DFTBlockPlanArray{Complex{RT}}
    N           ::  Int
    p           ::  Int
    blocks      ::  Matrix{DFTBlockPlan{RT}}
    # Precomputed length-N FFT plans follow
    FFT!        ::  FFTPLAN{RT}
    IFFT!       ::  IFFTPLAN{RT}
    # Temporary storage for an allocation-free matrix-vector product
    t1          ::  Vector{Complex{RT}}
    t2          ::  Vector{Complex{RT}}
    t3          ::  Vector{Complex{RT}}
    t4          ::  Vector{Complex{RT}}
    t_plunge    ::  Vector{Complex{RT}}
    y_reduced   ::  Vector{Complex{RT}}

    function RegularDFTBlockPlanArray{T,RT}(N, p) where {T,RT}
        CT = Complex{RT}
        blocks = blockdft_blocks(N, p, RT)
        FFT! = plan_fft!(zeros(CT, N))
        IFFT! = plan_ifft!(zeros(CT, N))
        plungesize = length(blocks[1].center_block.I)
        Q = round(Int, N/p)
        new(N, p, blocks, FFT!, IFFT!,
            zeros(CT,N), zeros(CT,N), zeros(CT,N), zeros(CT,N),
            zeros(CT,plungesize), zeros(CT, Q))
    end
end

RegularDFTBlockPlanArray{T}(args...) where {T} = RegularDFTBlockPlanArray{T,real(T)}(args...)

Base.axes(A::RegularDFTBlockPlanArray) = map(blockedrange, (Fill(A.N, A.p), Fill(A.N, A.p)))
function Base.getindex(A::DFTBlockPlanArray, k::Int, l::Int)
    checkbounds(A, k, l)
    dft_entry(size(A,1), k, l, prectype(A))
end

Base.getindex(A::RegularDFTBlockPlanArray, blockindex::Block{2}) =
    A.blocks[blockindex.n[1],blockindex.n[2]]

function mv(A::RegularDFTBlockPlanArray{T,RT}, x::BlockVector) where {T,RT}
    y = similar(x,Complex{RT})
    mv!(y, A, x)
end


function mv!(y::BlockVector, A::RegularDFTBlockPlanArray, x::BlockVector)
    fill!(y, 0)

    block1 = A.blocks[1]
    D_M1 = block1.D_M
    D_N1 = block1.D_N
    yb = A.t4

    y_reduced = A.y_reduced
    t1 = A.t1; t2 = A.t2; t3 = A.t3; t_plunge = A.t_plunge
    FFT! = A.FFT!
    IFFT! = A.IFFT!

    p = A.p
    for l in 1:p
        for k in 1:p
            # yb = A.blocks[k,l] * x[Block(l)]
            mv!_part1(y_reduced, A.blocks[k,l], x[Block(l)], D_N1, t1, t_plunge, FFT!)
            mv!_part2(yb, A.blocks[k,l], y_reduced, D_M1, t1, t2, t3, t_plunge, FFT!, IFFT!)
            # y[Block(k)] += yb
            axpy!(1, yb, y[Block(k)])
        end
    end
    y
end


"A plan for a block-based DFT transform."
struct BlockDFTPlan{T}
    N       ::  Int
    p       ::  Int
    blocks  ::  Array{DFTBlockPlan{T},2}
end
