module BlockDFT

using BlockArrays, FillArrays, FFTW, LinearAlgebra

export twiddle,
    DFTMatrix,
    iDFTMatrix,
    DFTBlock,
    CenteredDFTBlock,
    DiscreteProlateMatrix,
    jacobi_prolate,
    pdpss,
    pdpss_range,
    pdpss_plunge


prectype(A) = real(eltype(A))

include("definitions.jl")
include("prolate.jl")
include("block.jl")
include("conditioning.jl")
include("mv.jl")

end # module
