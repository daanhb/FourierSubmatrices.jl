
using Test, BlockArrays, FFTW, FillArrays, LinearAlgebra

using BlockDFT

N = 2048
p = 16
L = N*p
x = rand(L)
y = BlockDFT.blockdft(L, p, x);
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
