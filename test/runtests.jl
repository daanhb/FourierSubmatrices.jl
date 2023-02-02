
using Test, BlockArrays, FFTW, FillArrays, LinearAlgebra

using BlockDFT

N = 2048
p = 16
L = N*p
x = rand(L)
y = BlockDFT.blockdft(L, p, x)
@test norm(y-fft(x))/norm(y) < 1e-7

N = 235
p = 7
L = p*N
r = BlockDFT.RegularDFTBlockArray{Float64}(N,p)
xb = BlockArray{ComplexF64}(undef, Fill(N,p))
xb .= rand(ComplexF64, length(xb))
yb = similar(xb)
BlockDFT.mv!(yb, r, xb)
@test norm(yb-fft(xb))/norm(yb) < 1e-8

N = 128
M = 128
p = 8
L = p*N
A = collect(DFTMatrix(L))
Ainv = collect(iDFTMatrix(L))
AMN = A[1:M,1:N]
twiddle = exp(2*pi*im/L)
pivot_N = (N-1)/2
pivot_M = (M-1)/2
D_M = Diagonal(twiddle.^(0:M-1))
D_N = Diagonal(twiddle.^(0:N-1))
c = twiddle^(pivot_M*pivot_N)
BMN = D_M^(pivot_N) * AMN * D_N^(pivot_M) / c

cb = BlockDFT.CenteredDFTBlock(L, M, N)
@test norm(cb-BMN) < 1e-10

P1 = BlockDFT.DiscreteProlateMatrix(L, M, N)
G1 = BMN' * BMN
@test norm(G1/M-P1) < 1e-10

P2 = BlockDFT.DiscreteProlateMatrix(L, N, M)
G2 = BMN * BMN'
@test norm(G2/N-P2) < 1e-10

cU,cS,cV = BlockDFT.centered_pdpss(L, M, N)
@test norm(diag(cU'*cb*cV)-diag(cS)) < 1e-10

pU,pS,pV,pI = BlockDFT.centered_pdpss_plunge(L, M, N)
@test norm(diag(pU'*BMN*pV)-diag(pS)) < 1e-8

bl = BlockDFT.DFTBlock(L, 1:M, 1:N)
@test norm(bl - AMN) < 1e-10

D_M, D_N, c = BlockDFT.blockshift(bl)
@test norm(D_M*bl*D_N /c - cb) < 1e-9

U, S, V, D_M, D_N, c, I = BlockDFT.pdpss_plunge(bl)
@test norm(diag(U'*D_M*bl*D_N*V/c)-diag(S)) < 1e-8

dU,dS,dV,dM,dN,dc = BlockDFT.pdpss(bl)
@test norm(dM'*dU*dS*dV'*dN'*dc - bl) < 1e-10

x = rand(N)
yA = AMN * x
yc = cb * x
yl = bl * x
@test norm(yl - c*D_M'*cb*D_N'x) < 1e-10

x2 = cb.V * (cb.V' * x)
x13 = x - x2
w13 = fft(D_N * x13)
Q = round(Int,N*M/L)
x1 = real(D_N' * ifft([w13[1:Q]; zeros(N-Q)]))
@test norm(cb * (x1+x2) - yc) < 5e-7

y2 = cb * x2
v2 = cb.V' * x
x2 = cb.V * v2
u2 = cb.U * (cb.S * v2)
@test norm(u2-y2)  < 1e-9

y1 = cb * x1
x_ext = [D_N * x1; zeros(L-N)]
wx_ext = D_M * fft(x_ext)[1:M] / c
@test norm(wx_ext - y1) < 1e-9
