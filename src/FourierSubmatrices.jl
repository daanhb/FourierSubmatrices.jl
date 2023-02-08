module FourierSubmatrices

using BlockArrays, FFTW
using LinearAlgebra, GenericLinearAlgebra

export twiddle,
    DFTMatrix,
    iDFTMatrix,
    DFTBlock,
    CenteredBlock,
    DiscreteProlateMatrix,
    jacobi_prolate,
    pdpss,
    pdpss_plunge


prectype(A) = real(eltype(A))

include("definitions.jl")
include("prolate.jl")
include("svd.jl")
include("cond.jl")

end # module
