module Experimental
using RvSpectMLBase
using EchelleInstruments
using EchelleCCFs
using DataFrames, Query
using Dates

CCFs = EchelleCCFs

include("pipeline/pipeline.jl")

include("line_finder.jl")
export LineFinderPlan

include("interp/gp/temporalgps.jl")
export TemporalGPInterpolation
export construct_gp_posterior, gp_marginal, predict_gp
export predict_mean, predict_deriv, predict_deriv2, predict_mean_and_deriv, predict_mean_and_derivs


end
