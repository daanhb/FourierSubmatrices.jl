# FourierSubmatrices.jl

This is a package to manipulate subblocks of the DFT matrix. The main functionality is an accurate computation of the singular value decomposition of the subblock.

## Usage

The main type is a `DFTBlock`. One can create the top-left `p x q` subblock or a subblock corresponding to a range of rows and columns as follows:
```julia
julia> using FourierSubmatrices

julia> A = DFTBlock(500, 14, 39); size(A)
(14, 39)

julia> A2 = DFTBlock(500, 10:25, 39:117); size(A2)
(16, 79)
```

## Definition

The Fourier submatrix `A` of size `p x q` agrees with a subblock of the `N x N` DFT matrix that corresponds with the `fft`. Multiplication with `A` corresponds to extension by zeros to a vector of length `N`, followed by the `fft`, followed by a restriction to the subset of rows:
```julia
julia> using FFTW

julia> N = 230; p = 112; q = 65;

julia> A = DFTBlock(N, p, q);

julia> x = rand(q); x_ext = [x; zeros(N-q)]; y_ext = fft(x_ext); y = y_ext[1:p];

julia> norm(y - A*x)
0.0
```

## Singular value decomposition

### Singular vectors

The singular vectors of Fourier submatrices have been studied under the name "periodic discrete prolate spheroidal sequences" (P-DPPSS). Standard SVD methods are not accurate for Fourier submatrices, because the singular values are clustered. That makes computation of the individual singular vectors ill-conditioned.

This package implements an alternative algorithm called the Commuting Tridiagonal Algorithm (CTA). The singular vectors are computed accurately, regardless of the corresponding singular value, as the eigenvectors of a tridiagonal matrix that commutes with `A'A` or with `AA'`.

The algorithm is invoked simply with the standard `svd` command:
```julia
julia> using LinearAlgebra

julia> u,s,v = svd(DFTBlock(N, p, q)); s[40]
0.004578961704651389
```

### Singular values

The singular values can only be computed accurately in standard precision if they are not too small. For small singular values, higher-precision arithmetic is required. The computation also uses the FFT of length `N`, for which in the following example we load the `GenericFFT` package:
```julia
julia> using GenericFFT

julia> u,s,v = svd(DFTBlock(N, p, q)); s[end]
1.6502882656979275e-16

julia> u,s,v = svd(DFTBlock{BigFloat}(N, p, q)); s[end]
7.204726671893715713779103966796263392431543875558953205738556163196036672198492e-26
```

## Condition number

A topic of considerable theoretical and practical interest is the condition number of Fourier submatrices. An efficient algorithm is available to compute condition numbers with the standard `cond` command even if they are very large:
```julia
julia> setprecision(BigFloat, 1024)
1024

julia> N = 1037; p = 357; q = 562;

julia> A = DFTBlock{BigFloat}(N, p, q);

julia> @time cond(A)
  0.535734 seconds (4.57 M allocations: 455.887 MiB, 13.65% gc time)
7.353651340014338386308637101137516042764778837567350581850384364782791623259092763241859087743000175330434038100621859474328287467408841602814990133320708022569636091180354558022514052969923258546843864368919080111560367115514093579259392688465550800952961456390592770313138054296597528736581884657351588412831e+131

julia> @time cond(collect(A))
494.699396 seconds (2.27 G allocations: 219.593 GiB, 24.24% gc time)
7.353651340014338386308637101137516042764778837567350581850384364782791623259092763241859087743000175330434038100621859474328287467408841602814990133320708022569636091180354557959778294039718607645544693277263712453710701320865676191157343345455473358959564562256902661467175654918318115580709482220000284105052e+131

```
The latter statement involving `collect` computes the condition number of the dense `BigFloat` matrix, instead of using CTA. It is slower by a factor of `1000` on a contemporary laptop. Computation of the condition number in standard double floating point precision is limited to condition numbers smaller than `1e15`.


## References

The methods of this package are described in the paper "On the computation of the SVD of Fourier submatrices" by S. Dirckx, D. Huybrechs and R. Ongenae. The paper is available on the (arXiv)[https://arxiv.org/abs/2208.12583]. A recent (reference)[https://epubs.siam.org/doi/10.1137/20M1336837] for lower bounds on the condition number of Fourier submatrices is "How exponentially ill-conditioned are contiguous submatrices of the Fourier matrix?" by Alex Barnett.
