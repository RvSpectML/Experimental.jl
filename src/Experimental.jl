module Experimental
using RvSpectMLBase
using EchelleInstruments
using EchelleCCFs
using DataFrames, Query
using Dates
using Statistics

CCFs = EchelleCCFs

include("pipeline/pipeline.jl")

include("interp/interp.jl")
#=
include("interp/gp/temporalgps.jl")
export TemporalGPInterpolation
export construct_gp_posterior, gp_marginal, predict_gp
export predict_mean, predict_deriv, predict_deriv2, predict_mean_and_deriv, predict_mean_and_derivs
=#

include("line_finder/line_finder.jl")
export LineFinderPlan

include("alg/project_flux_common_wavelengths.jl")
include("alg/make_template_spectrum.jl")

#include("alg/dcpca.jl")
#include("alg/ppcca.jl")
#include("alg/rvs_from_gp_pairs.jl")
end
