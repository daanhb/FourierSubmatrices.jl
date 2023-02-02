module BlockDFT

using BlockArrays, FFTW
using LinearAlgebra, GenericLinearAlgebra

export twiddle,
    DFTMatrix,
    iDFTMatrix,
    DFTBlock,
    DiscreteProlateMatrix,
    jacobi_prolate,
    pdpss,
    pdpss_plunge


prectype(A) = real(eltype(A))

include("definitions.jl")
include("prolate.jl")
include("block.jl")
include("conditioning.jl")
include("mv.jl")

end # module
