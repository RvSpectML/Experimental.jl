using RvSpectMLBase
import EchelleInstruments
import EchelleInstruments: EXPRES
#using EchelleCCFs
using Experimental
#using DataFrames, Query
using Stheno, TemporalGPs
using Statistics
using Optim

all_spectra = include(joinpath(pkgdir(EchelleInstruments),"examples/read_expres_data_101501.jl"))

order_list_timeseries = extract_orders(all_spectra,pipeline_plan, recalc=true )

function calc_logλ_grid(chunklist_timeseries, obs_idx, obs_jdx, chunk_idx; sample_scale_factor::Real = 1, Δv::Real = 0, skip_edge_pixels::Integer = 100)
	@assert 1+skip_edge_pixels < length(chunklist_timeseries[obs_idx][chunk_idx].λ)-skip_edge_pixels

	minlogλ = log( max(chunklist_timeseries[obs_idx][chunk_idx].λ[1+skip_edge_pixels], chunklist_timeseries[obs_jdx][chunk_idx].λ[1+skip_edge_pixels] ))
	maxlogλ = log( min(chunklist_timeseries[obs_idx][chunk_idx].λ[end-skip_edge_pixels], chunklist_timeseries[obs_jdx][chunk_idx].λ[end-skip_edge_pixels] ))
	logλ_grid = range(minlogλ, stop=maxlogλ, length = 1+2*(ceil(Int,sample_scale_factor*length(chunklist_timeseries[obs_idx][chunk_idx].λ)//2)-skip_edge_pixels) )
end

function calc_logλ_grid_all_pairs(chunklist_timeseries, chunk_idx; sample_scale_factor::Real = 1, Δv::Real = 0, skip_edge_pixels::Integer = 100)
	@assert 1+skip_edge_pixels < length(chunklist_timeseries[1][chunk_idx].λ)-skip_edge_pixels

	minlogλ = log( maximum(map(obs_idx->chunklist_timeseries[obs_idx][chunk_idx].λ[1+skip_edge_pixels], 1:length(chunklist_timeseries)) ))
	maxlogλ = log( minimum(map(obs_idx->chunklist_timeseries[obs_idx][chunk_idx].λ[end-skip_edge_pixels], 1:length(chunklist_timeseries)) ))
	logλ_grid = range(minlogλ, stop=maxlogλ, length = 1+2*(ceil(Int,sample_scale_factor*length(chunklist_timeseries[1][chunk_idx].λ)//2)-skip_edge_pixels) )
end

function calc_cross_cor_obs_chunklist(chunklist_timeseries, obs_idx, obs_jdx, chunk_idx; sample_scale_factor::Real = 1, Δv::Real = 0, skip_edge_pixels::Integer = 100)
	logλ_grid = calc_logλ_grid(chunklist_timeseries, obs_idx, obs_jdx, chunk_idx, sample_scale_factor=sample_scale_factor, Δv=Δv, skip_edge_pixels=skip_edge_pixels )
	Δlogλ = log(RvSpectMLBase.calc_doppler_factor(Δv))
	(pixels, order) = chunklist_timeseries[obs_idx][chunk_idx].flux.indices
	blaze = chunklist_timeseries.metadata[obs_idx][:blaze][pixels,order]
	pred_spec1 = Experimental.TemporalGPInterpolation.predict_mean( log.(chunklist_timeseries[obs_idx][chunk_idx].λ).+0.5*Δlogλ,
										chunklist_timeseries[obs_idx][chunk_idx].flux ./ blaze,
										logλ_grid, sigmasq_obs=chunklist_timeseries[obs_idx][chunk_idx].var ./ blaze.^2,
										use_logx = false, use_logy = false, smooth_factor = 1)
	(pixels, order) = chunklist_timeseries[obs_jdx][chunk_idx].flux.indices
	blaze = chunklist_timeseries.metadata[obs_jdx][:blaze][pixels,order]
	pred_spec2 = Experimental.TemporalGPInterpolation.predict_mean( log.(chunklist_timeseries[obs_jdx][chunk_idx].λ).-0.5*Δlogλ,
										chunklist_timeseries[obs_jdx][chunk_idx].flux ./ blaze,
										logλ_grid, sigmasq_obs=chunklist_timeseries[obs_jdx][chunk_idx].var ./ blaze.^2,
										use_logx = false, use_logy = false, smooth_factor = 1)
	scale_factor = mean(pred_spec1)/mean(pred_spec2)
	result = mean( (pred_spec1 .- scale_factor.*pred_spec2 ).^2 )
	return result
end


function calc_cross_cor_obs_chunklist_alt(chunklist_timeseries, obs_idx, obs_jdx, chunk_idx; sample_scale_factor::Real = 1, Δv::Real = 0, skip_edge_pixels::Integer = 100)
	logλ_grid = calc_logλ_grid(chunklist_timeseries, obs_idx, obs_jdx, chunk_idx, sample_scale_factor=sample_scale_factor, Δv=Δv, skip_edge_pixels=skip_edge_pixels )
	Δlogλ = log(RvSpectMLBase.calc_doppler_factor(Δv))
	(pixels, order) = chunklist_timeseries[obs_idx][chunk_idx].flux.indices
	blaze = chunklist_timeseries.metadata[obs_idx][:blaze][pixels,order]
	spec1_gp = TemporalGPInterpolation.construct_gp_posterior( log.(chunklist_timeseries[obs_idx][chunk_idx].λ).+0.5*Δlogλ,
										chunklist_timeseries[obs_idx][chunk_idx].flux ./ blaze,
										logλ_grid, sigmasq_obs=chunklist_timeseries[obs_idx][chunk_idx].var ./ blaze.^2,
										use_logx = false, use_logy = false, smooth_factor = 1)
	marginal_gp1 = marginals(spec1_gp(logλ_grid))
	(pixels, order) = chunklist_timeseries[obs_jdx][chunk_idx].flux.indices
	blaze = chunklist_timeseries.metadata[obs_jdx][:blaze][pixels,order]
	spec2_gp = TemporalGPInterpolation.construct_gp_posterior( log.(chunklist_timeseries[obs_jdx][chunk_idx].λ).-0.5*Δlogλ,
										chunklist_timeseries[obs_jdx][chunk_idx].flux ./ blaze,
										logλ_grid, sigmasq_obs=chunklist_timeseries[obs_jdx][chunk_idx].var ./ blaze.^2,
										use_logx = false, use_logy = false, smooth_factor = 1)
	marginal_gp2 = marginals(spec2_gp(logλ_grid))
	#scale_factor = mean(pred_spec1)/mean(pred_spec2)
	scale_factor = mean(mean.(marginal_gp1)./mean.(marginal_gp2))
	mean_diff = (mean.(marginal_gp1) .- scale_factor .* mean.(marginal_gp2))
	var_diff = (var.(marginal_gp1) .+ scale_factor .* var.(marginal_gp2))
	result = mean(mean_diff.^2 ./ var_diff)
end



function calc_cross_cor_obs_chunklist_all_pairs(chunklist_timeseries, chunk_idx; sample_scale_factor::Real = 1, Δv::Real = 0, skip_edge_pixels::Integer = 100, max_num_obs::Integer = 0)
	logλ_grid = calc_logλ_grid_all_pairs(chunklist_timeseries, chunk_idx, sample_scale_factor=sample_scale_factor, Δv=Δv, skip_edge_pixels=skip_edge_pixels )
	num_obs = max_num_obs > 0 ? max_num_obs : length(chunklist_timeseries)
	spec_gps = Any[]
	for obs_idx in 1:num_obs
		(pixels, order) = chunklist_timeseries[obs_idx][chunk_idx].flux.indices
		blaze = chunklist_timeseries.metadata[obs_idx][:blaze][pixels,order]
		spec1_gp = Experimental.TemporalGPInterpolation.construct_gp_posterior( log.(chunklist_timeseries[obs_idx][chunk_idx].λ),
										chunklist_timeseries[obs_idx][chunk_idx].flux ./ blaze,
										sigmasq_obs=chunklist_timeseries[obs_idx][chunk_idx].var ./ blaze.^2,
										use_logx = false, smooth_factor = 1)
	   push!(spec_gps,spec1_gp)
   end
   χ² = zeros(num_obs,num_obs)
   Δrv = zeros(num_obs,num_obs)
   σrv = zeros(num_obs,num_obs)
   rv_eps = 0.01
   #logλ_grid = collect(logλ_grid)
   for obs_idx in 1:num_obs
	   #marginal_gp1 = marginals(spec_gps[obs_idx](logλ_grid.-0.5*Δlogλ))
	   marginal_gp1 = marginals(spec_gps[obs_idx](logλ_grid))
	   for obs_jdx in 1:num_obs
		   	if obs_idx == obs_jdx   continue   end
			#marginal_gp2 = marginals(spec_gps[obs_jdx](logλ_grid.+0.5*Δlogλ))
			function func_to_minimize(θ)
				@assert length(θ) == 1
				dv = θ[1]
				Δlogλ = log(RvSpectMLBase.calc_doppler_factor(dv))
				marginal_gp2 = marginals(spec_gps[obs_jdx](logλ_grid.+Δlogλ))
				scale_factor = mean(mean.(marginal_gp1)./mean.(marginal_gp2))
				mean_diff = (mean.(marginal_gp1) .- scale_factor .* mean.(marginal_gp2))
				var_diff = (var.(marginal_gp1) .+ scale_factor .* var.(marginal_gp2))
				chi2 = sum(mean_diff.^2 ./ var_diff)
				return chi2
			end
			#=
			res = optimize(func_to_minimize, [0.0], Newton(), Optim.Options(x_tol=rv_eps) )
			if !Optim.converged(res)
				@warn "optim didn't converged for $obs_idx and $obs_jdx ."
				println(summary(res))
			else
				χ²[obs_idx,obs_jdx] = minimum(res)
				rv_bf = Optim.minimizer(res)[1]
				#dchi2drv = (func_to_minimize(rv_bf+rv_eps)-func_to_minimize(rv_bf-rv_eps))/(2*rv_eps)
				d2chi2drv2 = (func_to_minimize(rv_bf+rv_eps)-2func_to_minimize(rv_bf)+func_to_minimize(rv_bf-rv_eps))/(2*rv_eps)^2
				Δrv[obs_idx,obs_jdx] = rv_bf
				σrv[obs_idx,obs_jdx] = 2.0 /d2chi2drv2
			end
			=#
			rv_bf = 0
			chisq_v0 = func_to_minimize(rv_bf)
			dchi2drv = (func_to_minimize(rv_bf+rv_eps)-func_to_minimize(rv_bf-rv_eps))/(2*rv_eps)
			d2chi2drv2 = (func_to_minimize(rv_bf+rv_eps)-2func_to_minimize(rv_bf)+func_to_minimize(rv_bf-rv_eps))/(2*rv_eps)^2
			v_at_min_of_quadratic = -dchi2drv/(2*d2chi2drv2)
			Δrv[obs_idx,obs_jdx] = v_at_min_of_quadratic
			σrv[obs_idx,obs_jdx] = 2.0 /d2chi2drv2
			χ²[obs_idx,obs_jdx] = func_to_minimize(v_at_min_of_quadratic)
			#=
			marginal_gp2 = marginals(spec_gps[obs_jdx](logλ_grid.+Δlogλ))
			scale_factor = mean(mean.(marginal_gp1)./mean.(marginal_gp2))
			mean_diff = (mean.(marginal_gp1) .- scale_factor .* mean.(marginal_gp2))
			var_diff = (var.(marginal_gp1) .+ scale_factor .* var.(marginal_gp2))
			results[obs_idx,obs_jdx] = sum(mean_diff.^2 ./ var_diff)
			=#
		end
  end
  return (Δrv=Δrv, σrv=σrv, χ²=χ²)
end


function plot_Δ_chunk(chunklist_timeseries, obs_idx, obs_jdx, chunk_idx; sample_scale_factor::Real = 1, Δv::Real = 0, skip_edge_pixels::Integer = 100)
	logλ_grid = calc_logλ_grid(chunklist_timeseries, obs_idx, obs_jdx, chunk_idx, sample_scale_factor=sample_scale_factor, Δv=Δv, skip_edge_pixels=skip_edge_pixels )
	Δlogλ = log(RvSpectMLBase.calc_doppler_factor(Δv))
	(pixels, order) = chunklist_timeseries[obs_idx][chunk_idx].flux.indices
	blaze = chunklist_timeseries.metadata[obs_idx][:blaze][pixels,order]
	pred_spec1 = Experimental.TemporalGPInterpolation.predict_mean( log.(chunklist_timeseries[obs_idx][chunk_idx].λ).+0.5*Δlogλ,
										chunklist_timeseries[obs_idx][chunk_idx].flux ./ blaze,
										logλ_grid, sigmasq_obs=chunklist_timeseries[obs_idx][chunk_idx].var ./ blaze.^2,
										use_logx = false, use_logy = true, smooth_factor = 1)

	(pixels, order) = chunklist_timeseries[obs_jdx][chunk_idx].flux.indices
	blaze = chunklist_timeseries.metadata[obs_jdx][:blaze][pixels,order]
	pred_spec2 = Experimental.TemporalGPInterpolation.predict_mean( log.(chunklist_timeseries[obs_jdx][chunk_idx].λ).-0.5*Δlogλ,
										chunklist_timeseries[obs_jdx][chunk_idx].flux ./ blaze,
										logλ_grid, sigmasq_obs=chunklist_timeseries[obs_jdx][chunk_idx].var  ./ blaze.^2,
										use_logx = false, use_logy = true, smooth_factor = 1)
	result = pred_spec1 .- pred_spec2 .* (mean(pred_spec1)/mean(pred_spec2))
end

using StatsBase
function compute_ave_Δrv_weighted(Δrv::AbstractArray{T1,2}, σ_rv::AbstractArray{T2,2}) where {T1<:Real, T2<:Real}
    @assert size(Δrv) == size(σ_rv)
    n = size(Δrv)[1]
    rv_out = zeros(n)
    weight = zeros(n)
    σ_rv_out = zeros(n)
    for i in 1:n
        wsum = 0.0
        for j in 1:n
            if i==j   continue   end
            weight[j] = 1 / σ_rv[i,j]^2
            #rv_out[i] += Δrv[i,j] * weight[i]
            #wsum += weight[i]
        end
        #rv_out[i] /= wsum
		rv_out[i] = mean(Δrv[:,i], Weights(weight))
        #σ_rv_out[i] = std(Δrv[:,i], Weights(weight), mean=rv_out[i])
		σ_rv_out[i] = std(Δrv[:,i], Weights(weight), mean=rv_out[i])
		println(i,": rv: ", rv_out[i], "  σ_rv: ", σ_rv_out[i] )
    end
    return (rvs=rv_out, σ_rv=σ_rv_out)
end

@time order5_cross_cor_gp_results = calc_cross_cor_obs_chunklist_all_pairs(order_list_timeseries, 5, Δv=0 ) #, max_num_obs =3)
compute_ave_Δrv_weighted(order5_cross_cor_gp_results.Δrv, order5_cross_cor_gp_results.σrv)

using JLD2
using Distributed

num_chunks_fit = num_chunks(order_list_timeseries)
results = pmap(ch-> calc_cross_cor_obs_chunklist_all_pairs(order_list_timeseries, ch, Δv=0), 1:num_chunks_fit)
 order_cross_cor_gp_results = results
 gp_pair_rvs = pmap(ch-> compute_ave_Δrv_weighted(results[ch].Δrv, results[ch].σrv), 1:num_chunks_fit)
 @save "gp_all_pairs_fits.jld2" order_cross_cor_gp_results gp_pair_rvs
 std_of_gp_pair_rvs = map(obsid->std(map(ch->gp_pair_rvs[ch].rvs[obsid], 1:num_chunks_fit) ), 1:length(order_list_timeseries) )
 mean_of_gp_pair_σ_rvs = map(obsid->mean(map(ch->gp_pair_rvs[ch].σ_rv[obsid], 1:num_chunks_fit) ), 1:length(order_list_timeseries) )
 mean_of_gp_pair_rvs = map(obsid->mean(map(ch->gp_pair_rvs[ch].rvs[obsid], 1:num_chunks_fit) ), 1:length(order_list_timeseries) )


scatter(mean_of_gp_pair_rvs)

#=
using Plots
plt = plot()
 logλ_grid = calc_logλ_grid(order_list_timeseries, 5, 6, 5 )
 plot!(exp.(logλ_grid),plot_Δ_chunk(order_list_timeseries, 5, 6, 5, Δv=0))
 plot!(exp.(logλ_grid),plot_Δ_chunk(order_list_timeseries, 5, 6, 5, Δv=100))
 plot!(exp.(logλ_grid),plot_Δ_chunk(order_list_timeseries, 5, 6, 5, Δv=-100))
 xlims!(5360,5370)

plot_Δ_chunk(order_list_timeseries, 5, 6, 5, Δv=0)

Δv_list = range(-10,stop=10,length=51)
 score_list_5 = map(Δv->calc_cross_cor_obs_chunklist(order_list_timeseries, 5, 6, 5, Δv=Δv), Δv_list)
 score_list_6 = map(Δv->calc_cross_cor_obs_chunklist(order_list_timeseries, 5, 6, 6, Δv=Δv), Δv_list)
 score_list_7 = map(Δv->calc_cross_cor_obs_chunklist(order_list_timeseries, 5, 6, 7, Δv=Δv), Δv_list)
 plt = plot()
 plot!(Δv_list, score_list_5./minimum(score_list_5))
 plot!(Δv_list, score_list_6./minimum(score_list_6))
 plot!(Δv_list, score_list_7./minimum(score_list_7))

using Plots

plt = plot()
plot!(exp.(logλ_grid),pred_spec1)
plot!(exp.(logλ_grid),pred_spec2)
display(plt)
=#




using Distributed
using ProgressMeter

num_chunks_fit = 10 # num_chunks(order_list_timeseries)

@time order_cross_cor_gp_results =  @showprogress pmap(order->calc_cross_cor_obs_chunklist_all_pairs(order_list_timeseries, order, Δv=0), 1:num_chunks_fit )
@time gp_pair_rvs = @showprogress pmap(ch->compute_ave_Δrv_weighted(order_cross_cor_gp_results[ch].Δrv,order_cross_cor_gp_results[ch].σrv), 1:num_chunks_fit )


std_of_gp_pair_rvs = map(obsid->std(map(ch->gp_pair_rvs[ch].rvs[obsid], 1:num_chunks_fit) ), 1:length(order_list_timeseries) )
mean_of_gp_pair_rvs = map(obsid->mean(map(ch->gp_pair_rvs[ch].rvs[obsid], 1:num_chunks_fit) ), 1:length(order_list_timeseries) )



using Plots
plt = plot(); map(i->scatter!(plt,gp_pair_rvs[i].rvs, label=:none),1:num_chunks(order_list_timeseries)) ; display(plt)

scatter(std_of_gp_pair_rvs)
scatter(mean_of_gp_pair_rvs)



using JLD2
@save "gp_all_pairs_fits.jld2" order_cross_cor_gp_results
