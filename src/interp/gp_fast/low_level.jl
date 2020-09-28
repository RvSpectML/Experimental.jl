
""" `construct_gp_prior()`
Returns a GP prior using a Matern 5/2 kernel and specified parameters
Optional Inputs:
- smooth_factor: Multiplies variance and length scale for GP kernel (1)
- σ²: Variance for GP kernel (0.5)
- l: length scale for GP kernel (5.8e-5)
"""
function construct_gp_prior(; smooth_factor::Real = 1, σ²::Real = 0.5, l::Real = 5.8e-5)
	σ² *= smooth
	l *= smooth
	k = σ² * stretch(Matern52(), 1 / l)
	f_naive = GP(k, GPC())
	#f = to_sde(f_naive)   # if develop issues with StaticArrays could revert to this
	f = to_sde(f_naive, SArrayStorage(Float64))
end

#=
function construct_gp_prior_kernel_param_as_vector(gp_param = [0.5, 5.8e-5 ])
	σ², l = gp_param
	k = σ² * stretch(Matern52(), 1 / l)
	f_naive = GP(k, GPC())
	#f = to_sde(f_naive)   # if develop issues with StaticArrays could revert to this
	f = to_sde(f_naive, SArrayStorage(Float64))
end
=#

""" `construct_gp_posterior(xobs, yobs, xpred; sigmasq_obs, use_logx, use_logy, smooth_factor, boost_factor )`
Inputs:
- xobs: x locations where data is provided
- yobs: y values of data to condition on
Optional Inputs:
- sigmasq_obs: variances for y values being conditioned on
- use_logx: If true, take log's of x values before fitting GP
- use_logy: If true, perform log transform on y's
- smooth_factor: scales GP hyperparameters so as to result in smoother GP posterior (1)
- boost_factor: scales xobs by 1/boost_factor (1)
Returns:
- Posterior GP at locations xpred given training data
"""
function construct_gp_posterior(xobs::AA1, yobs::AA2; sigmasq_obs::AA4 = 1e-16*ones(length(xobs)) #=, sigmasq_cor::Real=1.0, rho::Real=1.0)  =#
			, use_logx::Bool = true, use_logy::Bool = true, smooth_factor::Real = 1, boost_factor::Real = 1 )   where {
				T1<:Real, AA1<:AbstractArray{T1,1}, T2<:Real, AA2<:AbstractArray{T2,1}, T3<:Real, AA3<:AbstractArray{T3,1}, T4<:Real, AA4<:AbstractArray{T4,1} }
	f = construct_gp_prior( smooth_factor=smooth_factor )
	xobs_trans = use_logx ? log.(xobs./boost_factor) : xobs./boost_factor
    yobs_trans = use_logy ? log.(yobs) : yobs
    sigmasq_obs_trans = use_logy ? sigmasq_obs./yobs.^2 : sigmasq_obs
	fx = f(xobs_trans, sigmasq_obs_trans)
	f_posterior = posterior(fx, yobs_trans )
end

""" predict_gp_mean_var(gp, xpred ; use_logx, use_logy)
Inputs:
- gp:
- xpred: Locations to predict GP at
Optional inputs:
- use_logx: If true, apply log transform to xpred before evaluating GP
- use_logy: If true, apply exp transform after evaluating GP
Returns tuple with vector of means and variance of GP posterior at locations in xpred.
"""
function predict_gp_mean_var(gp::AGP,  xpred::AA3, use_logx::Bool = true, use_logy::Bool = true )   where {
		AGP<:AbstractGP, T3<:Real, AA3<:AbstractArray{T3,1} }
		gp_post = use_log_x ? gp(log.(xpred)) : gp(xpred)
		m = mean.(gp_post)
		v = var.(gp_post)
		if use_logy
			m = exp.(m)
			v .*= m.^2
		end
		return m, v
end

""" predict_gp_mean(gp, xpred ; use_logx, use_logy)
Inputs:
- gp:
- xpred: Locations to predict GP at
Optional inputs:
- use_logx: If true, apply log transform to xpred before evaluating GP
- use_logy: If true, apply exp transform after evaluating GP
Returns vector of means of GP posterior at locations in xpred.
"""
function predict_mean(gp::AGP,  xpred::AA3, use_logx::Bool = true, use_logy::Bool = true )   where { AGP<:Fx_PosteriorType, T3<:Real, AA3<:AbstractArray{T3,1} }
	gp_post = use_log_x ? gp(log.(xpred)) : gp(xpred)
	m = mean.(gp_post)
	if use_logy
		m = exp.(m)
	end
	return m
end

function predict_deriv(gp::AGP, xpred::AA3; use_logx::Bool = true, use_logy::Bool = true )   where { AGP<:Fx_PosteriorType,
				T3<:Real, AA3<:AbstractArray{T3,1}  }
  #kobs = make_kernel_data(xobs, kernel=kernel, sigmasq_obs=sigmasq_obs, sigmasq_cor=sigmasq_cor, rho=rho)
  #kobs_pred_deriv = make_kernel_obs_pred(xobs,xpred, kernel=dkerneldx, sigmasq_cor=sigmasq_cor, rho=rho)
  #alpha = kobs \ yobs
  #pred_deriv = kobs_pred_deriv' * alpha
  xpred_trans = use_logx ? log.(xpred) : xpred
  m = predict_mean(gp, use_logy=use_logy)
  #if use_logy   m .= exp(m)   end
  return numerical_deriv(xpred_transt,m)
  #=
  dfluxdlnλ = zeros(size(m))
  dfluxdlnλ[1] = (m[2]-m[1])/(xpred_trans[2]-xpred_trans[1])
  dfluxdlnλ[2:end-1] .= (m[3:end].-m[1:end-2])./(xpred_trans[3:end].-xpred_trans[1:end-2]) # exp.(m[2:end-1]).*
  dfluxdlnλ[end] = (m[end]-m[end-1])/(xpred_trans[end]-xpred_trans[end-1])
  return dfluxdlnλ
	=#
end

function predict_deriv2(gp::AGP, xpred::AA3; use_logx::Bool = true, use_logy::Bool = true )   where { AGP<:Fx_PosteriorType,
				T3<:Real, AA3<:AbstractArray{T3,1}  }
	#kobs = make_kernel_data(xobs, kernel=kernel, sigmasq_obs=sigmasq_obs, sigmasq_cor=sigmasq_cor, rho=rho)
	#kobs_pred_deriv2 = make_kernel_obs_pred(xobs,xpred, kernel=d2kerneldx2, sigmasq_cor=sigmasq_cor, rho=rho)
	#alpha = kobs \ yobs
	#pred_deriv = kobs_pred_deriv2' * alpha

	xpred_trans = use_logx ? log.(xpred) : xpred
	m = predict_mean(gp,use_logy=use_logy)
	#if use_logy   m .= exp(m)   end
	d2fluxdlnλ2 = zeros(size(m))
	d2fluxdlnλ2[2:end-1] .= (m[3:end].+m[1:end-2].-2.0.*m[2:end-1])./(xpred_trans[3:end].-xpred_trans[1:end-2]).^2
	d2fluxdlnλ2[1] = d2fluxdlnλ2[2]
	d2fluxdlnλ2[end] = d2fluxdlnλ2[end-1]

	# TODO add some sort of check based on second derivative and dx that the finite difference is good enough?
	return d2fluxdlnλ2
end

function predict_mean(xobs::AA1, yobs::AA2, xpred::AA3;	sigmasq_obs::AA4 = 1e-16*ones(length(xobs)) #=, sigmasq_cor::Real=1.0, rho::Real=1.0)  =#
			, use_logx::Bool = true, use_logy::Bool = true, smooth_factor::Real = 1, boost_factor::Real = 1  )   where {
				T1<:Real, AA1<:AbstractArray{T1,1}, T2<:Real, AA2<:AbstractArray{T2,1}, T3<:Real, AA3<:AbstractArray{T3,1}, T4<:Real, AA4<:AbstractArray{T4,1} }
	# global ncalls += 1
	@assert size(xobs) == size(yobs) == size(sigmasq_obs)
  	#println("# predict_mean (TemporalGPs): size(xobs) = ",size(xobs), "  size(xpred) = ", size(xpred))
	tstart = now()
	f_posterior = construct_gp_posterior(xobs,yobs,xpred,sigmasq_obs=sigmasq_obs, use_logx=use_logx, use_logy=use_logy, smooth_factor=smooth_factor, boost_factor=boost_factor )
	#println("typeof(f_posterior) = ",typeof(f_posterior))
	#println("f_posterior <: AbstractGP = ",typeof(f_posterior) <: AbstractGP )
	#println("f_posterior <: AbstractMvNormal = ",typeof(f_posterior) <: AbstractMvNormal )
	xpred_trans = use_logx ? log.(xpred) : xpred
	fx_posterior = f_posterior(xpred_trans)
	#println("typeof(fx_posterior) = ",typeof(fx_posterior))
	#println("f_posterior <: AbstractGP = ",typeof(fx_posterior) <: AbstractGP )
	#println("f_posterior <: AbstractMvNormal = ",typeof(fx_posterior) <: AbstractMvNormal )
	#println("f_posterior <: Fx_PosteriorType = ",typeof(fx_posterior) <: Fx_PosteriorType )
	#output = predict_mean(f_posterior(xpred_trans), xpred_trans ) #, use_logx=use_logx,use_logy=use_logy)
	output = predict_mean(f_posterior(xpred_trans), use_logy=use_logy)
	#println("# predict_mean (TemporalGPs) runtime: ", now()-tstart)
	return output
end

# NEED TO TEST
function predict_deriv(xobs::AA1, yobs::AA2, xpred::AA3; sigmasq_obs::AA4 = 1e-16*ones(length(xobs))
						, use_logx::Bool = true, use_logy::Bool = true, smooth_factor::Real = 1  )    where { T1<:Real, AA1<:AbstractArray{T1,1}, T2<:Real, AA2<:AbstractArray{T2,1}, T3<:Real, AA3<:AbstractArray{T3,1}, T4<:Real, AA4<:AbstractArray{T4,1}  }
#			kernel::Function = matern52_sparse_kernel, dkerneldx::Function = dkerneldx_matern52_sparse,
#			sigmasq_cor::Real=1.0, rho::Real=1.0)
  #kobs = make_kernel_data(xobs, kernel=kernel, sigmasq_obs=sigmasq_obs, sigmasq_cor=sigmasq_cor, rho=rho)
  #kobs_pred_deriv = make_kernel_obs_pred(xobs,xpred, kernel=dkerneldx, sigmasq_cor=sigmasq_cor, rho=rho)
  #alpha = kobs \ yobs
  #pred_deriv = kobs_pred_deriv' * alpha
  println("# predict_deriv (TemporalGPs): size(xobs) = ",size(xobs), "  size(xpred) = ", size(xpred))
  xpred_trans = use_logx ? log.(xpred) : xpred
  tstart = now()
  f_posterior = construct_gp_posterior(xobs,yobs,xpred,sigmasq_obs=sigmasq_obs, use_logx=use_logx, use_logy=use_logy, smooth_factor=smooth_factor)
  output = predict_deriv(f_posterior(xpred_trans), vec(xpred_trans), use_logx=use_logx,use_logy=use_logy)
  println("# predict_deriv (TemporalGPs) runtime: ", now()-tstart)
	return output
end

# NEED TO TEST
function predict_deriv2(xobs::AA1, yobs::AA2, xpred::AA3;sigmasq_obs::AA4 = 1e-16*ones(length(xobs))
						, use_logx::Bool = true, use_logy::Bool = true, smooth_factor::Real = 1  )   where { T1<:Real, AA1<:AbstractArray{T1,1}, T2<:Real, AA2<:AbstractArray{T2,1}, T3<:Real, AA3<:AbstractArray{T3,1}, T4<:Real, AA4<:AbstractArray{T4,1}  }
#			kernel::Function = matern52_sparse_kernel, d2kerneldx2::Function = d2kerneldx2_matern52_sparse,
#			,	sigmasq_cor::Real=1.0, rho::Real=1.0
  #kobs = make_kernel_data(xobs, kernel=kernel, sigmasq_obs=sigmasq_obs, sigmasq_cor=sigmasq_cor, rho=rho)
  #kobs_pred_deriv2 = make_kernel_obs_pred(xobs,xpred, kernel=d2kerneldx2, sigmasq_cor=sigmasq_cor, rho=rho)
  #alpha = kobs \ yobs
  #pred_deriv = kobs_pred_deriv2' * alpha

  println("# predict_deriv2 (TemporalGPs): size(xobs) = ",size(xobs), "  size(xpred) = ", size(xpred))
  tstart = now()
  xpred_trans = use_logx ? log.(xpred) : xpred
  f_posterior = construct_gp_posterior(xobs,yobs,xpred,sigmasq_obs=sigmasq_obs, use_logx=use_logx, use_logy=use_logy, smooth_factor=smooth_factor )
  output = predict_deriv2(f_posterior(xpred_trans), xpred_trans, use_logx=use_logx,use_logy=use_logy)
  println("# predict_deriv2 (TemporalGPs) runtime: ", now()-tstart)
    return output
end

function predict_mean_and_deriv(xobs::AA1, yobs::AA2, xpred::AA3;sigmasq_obs::AA4 = 1e-16*ones(length(xobs))
	 							, use_logx::Bool = true, use_logy::Bool = true, smooth_factor::Real = 1 ) where { T1<:Real, AA1<:AbstractArray{T1,1}, T2<:Real, AA2<:AbstractArray{T2,1}, T3<:Real, AA3<:AbstractArray{T3,1}, T4<:Real, AA4<:AbstractArray{T4,1}  }
	#		kernel::Function = matern52_sparse_kernel, dkerneldx::Function = dkerneldx_matern52_sparse,
	#		sigmasq_cor::Real=1.0, rho::Real=1.0)
  #kobs = make_kernel_data(xobs, kernel=kernel, sigmasq_obs=sigmasq_obs, sigmasq_cor=sigmasq_cor, rho=rho)
  #alpha = kobs \ yobs
  #kobs_pred = make_kernel_obs_pred(xobs,xpred, kernel=kernel, sigmasq_cor=sigmasq_cor, rho=rho)
  #pred_mean = kobs_pred' * alpha
  #kobs_pred_deriv = make_kernel_obs_pred(xobs,xpred, kernel=dkerneldx, sigmasq_cor=sigmasq_cor, rho=rho)
  #pred_deriv = kobs_pred_deriv' * alpha
  println("# predict_mean_and_deriv (TemporalGPs): size(xobs) = ",size(xobs), "  size(xpred) = ", size(xpred))
  xobs_trans = use_logx ? log.(xobs) : xobs
  yobs_trans = use_logy ? log.(yobs) : yobs
  xpred_trans = use_logx ? log.(xpred) : xpred
  sigmasq_obs_trans = use_logy ? sigmasq_obs./yobs.^2 : sigmasq_obs
  tstart = now()
  f_posterior = construct_gp_posterior(xobs,yobs,xpred,sigmasq_obs=sigmasq_obs, use_logx=use_logx, use_logy=use_logy, smooth_factor=smooth_factor )
	# TODO Opt:  Avoid repeated calculating of mean
  pred_mean = predict_mean(f_posterior(xpred_trans), use_logy=use_logy)
  #pred_deriv = predict_deriv(f_posterior(xpred_trans), xpred_trans, use_logx=use_logx,use_logy=use_logy)
  pred_deriv = numerical_deriv(xpred_trans,pred_mean)
  println("# predict_mean_and_deriv (TemporalGPs) runtime: ", now()-tstart)

  return (mean=pred_mean, deriv=pred_deriv)
end

function predict_mean_and_derivs(xobs::AA1, yobs::AA2, xpred::AA3; sigmasq_obs::AA4 = 1e-16*ones(length(xobs))
								, use_logx::Bool = true, use_logy::Bool = true, smooth_factor::Real = 1  ) where { T1<:Real, AA1<:AbstractArray{T1,1}, T2<:Real, AA2<:AbstractArray{T2,1}, T3<:Real, AA3<:AbstractArray{T3,1}, T4<:Real, AA4<:AbstractArray{T4,1}  }
	#		kernel::Function = matern52_sparse_kernel, dkerneldx::Function = dkerneldx_matern52_sparse, d2kerneldx2::Function = d2kerneldx2_matern52_sparse,
	#		,	sigmasq_cor::Real=1.0, rho::Real=1.0)
  #=
  kobs = make_kernel_data(xobs, kernel=kernel, sigmasq_obs=sigmasq_obs, sigmasq_cor=sigmasq_cor, rho=rho)
  alpha = kobs \ yobs
  kobs_pred = make_kernel_obs_pred(xobs,xpred, kernel=kernel, sigmasq_cor=sigmasq_cor, rho=rho)
  pred_mean = kobs_pred' * alpha
  kobs_pred_deriv = make_kernel_obs_pred(xobs,xpred, kernel=dkerneldx, sigmasq_cor=sigmasq_cor, rho=rho)
  pred_deriv = kobs_pred_deriv' * alpha
  kobs_pred_deriv2 = make_kernel_obs_pred(xobs,xpred, kernel=d2kerneldx2, sigmasq_cor=sigmasq_cor, rho=rho)
  pred_deriv2 = kobs_pred_deriv2' * alpha
  =#
  #println("# predict_mean_and_derivs (TemporalGPs): size(xobs) = ",size(xobs), "  size(xpred) = ", size(xpred))
  xobs_trans = use_logx ? log.(xobs) : xobs
  yobs_trans = use_logy ? log.(yobs) : yobs
  xpred_trans = use_logx ? log.(xpred) : xpred
  sigmasq_obs_trans = use_logy ? sigmasq_obs./yobs.^2 : sigmasq_obs
  #mean_yobs_trans = mean(yobs_trans)
  #yobs_trans .-= mean_yobs_trans  # Since TemporalGPs don't like non-zero mean

  tstart = now()
  f_posterior = construct_gp_posterior(xobs_trans,yobs_trans,xpred_trans,sigmasq_obs=sigmasq_obs_trans, use_logx=false, use_logy=false, smooth_factor=smooth_factor )
  pred_mean = predict_mean(f_posterior(xpred_trans), use_logy=false)  # doesn't need use_logx
  #pred_mean .+= mean_yobs_trans   # Since TemporalGPs don't like non-zero mean
  if use_logy
	pred_mean = exp.(pred_mean)
  end
  pred_deriv = calc_dfluxdlnlambda(pred_mean,xpred_trans)
  pred_deriv2 = calc_d2fluxdlnlambda2(pred_mean,xpred_trans)
  #pred_deriv = predict_deriv(f_posterior, xpred_trans, use_logx=false,use_logy=false)
  #pred_deriv2 = predict_deriv2(f_posterior, xpred_trans, use_logx=false,use_logy=false)
  #println("# predict_mean_and_derivs (TemporalGPs) runtime: ", now()-tstart)
  return (mean=pred_mean, deriv=pred_deriv, deriv2=pred_deriv2)
end

function log_pdf_gp_posterior(xobs::AA, yobs::AA #, kernel::Function;
			; sigmasq_obs::AA = 1e-16*ones(length(xobs)),
			use_logx::Bool = true, use_logy::Bool = true, smooth_factor::Real = 1  )  where { T<:Real, AA<:AbstractArray{T,1} }
  	#kobs = make_kernel_data(xobs, kernel=kernel, sigmasq_obs=sigmasq_obs, sigmasq_cor=sigmasq_cor, rho=rho)
  	# -0.5*( invquad(kobs, yobs) + logdet(kobs) + length(xobs)*log(2pi) )
	xobs_trans = use_logx ? log.(xobs) : xobs
    yobs_trans = use_logy ? log.(yobs) : yobs
	# TODO Opt check if logs are being recalculated twice or otpimized away by compiler
    sigmasq_obs_trans = use_logy ? sigmasq_obs./yobs.^2 : sigmasq_obs
	f_posterior = construct_gp_posterior(xobs,yobs,xobs,sigmasq_obs=sigmasq_obs, use_logx=use_logx, use_logy=use_logy, smooth_factor=smooth_factor )
	return -logpdf(f_posterior(xobs_trans, sigmasq_obs_trans), yobs_trans)

end
