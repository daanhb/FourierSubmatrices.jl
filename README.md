# FourierSubmatrices.jl

This is a package to manipulate subblocks of the DFT matrix. The main functionality is an accurate computation of the singular value decomposition of the subblock.

## Usage

The main type is a `DFTBlock`. One can create the top-left `p \times q` subblock, or a subblock corresponding to a range of rows and columns as follows:
```julia
julia> using FourierSubmatrices, LinearAlgebra

julia> A = DFTBlock(500, 14, 39); size(A)

julia> A2 = DFTBlock(500, 10:25, 39:117); size(A2)
(16, 79)
```

The block agrees with a subblock `A` of the DFT as implemented by `fft`. Multiplication with `A` corresponds to extension by zeros to a vector of length `N`, followed by the `fft`, followed by a restriction to the subset of rows:
```julia
julia> N = 230; p = 112; q = 65;

julia> A = DFTBlock(N, p, q);

julia> x = rand(q); x_ext = [x; zeros(N-q)]; y_ext = fft(x_ext); y = y_ext[1:p];

julia> norm(y - A*x)
0.0
```

## Singular value decomposition

### Singular vectors

The singular vectors of Fourier submatrices have been studied under the name "periodic discrete prolate spheroidal sequences" (P-DPPSS). Standard SVD methods are not accurate for Fourier submatrices, because the singular values are clustered. In this package, an alternative algorithm is implemented called the Commuting Tridiagonal Algorithm (CTA). The singular vectors are computed accurately regardless of the corresponding singular value.

### Singular values


## References

The methods of this package are described in the paper "On the computation of the SVD of Fourier submatrices" by S. Dirckx, D. Huybrechs and R. Ongenae. The paper is available on the (arXiv)[https://arxiv.org/abs/2208.12583].
