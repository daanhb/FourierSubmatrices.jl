
using Test, FFTW, LinearAlgebra

using BlockDFT

N = 2048
p = 16
L = N*p
x = rand(L)
y = BlockDFT.blockdft(L, p, x);
@test norm(y-fft(x))/norm(y) < 1e-7
