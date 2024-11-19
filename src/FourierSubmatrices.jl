module FourierSubmatrices

using BlockArrays, FFTW
using LinearAlgebra, GenericLinearAlgebra

export twiddle,
    DFTMatrix,
    iDFTMatrix,
    DFTBlock,
    CenteredBlock,
    DiscreteProlateMatrix,
    pdpss,
    pdpss_plunge,
    pdpss_tridiag_matrix,
    ProlateMatrix,
    dpss,
    prolate_tridiag_matrix


prectype(A) = real(eltype(A))

include("definitions.jl")
include("prolate.jl")
include("svd.jl")
include("cond.jl")

include("deprecated.jl")

end # module
