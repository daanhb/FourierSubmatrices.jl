
using Test, BlockArrays, FillArrays
using FFTW, GenericFFT
using LinearAlgebra, GenericLinearAlgebra

using BlockDFT
using BlockDFT: DFTBlockPlan

include("test_prolates.jl")
include("test_mv.jl")
