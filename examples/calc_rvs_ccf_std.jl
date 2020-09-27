using RvSpectMLBase
import EchelleInstruments
import EchelleInstruments: EXPRES
using EchelleCCFs
using Experimental
#using DataFrames, Query

all_spectra = include(joinpath(pkgdir(EchelleInstruments),"examples/read_expres_data_101501.jl"))

order_list_timeseries = extract_orders(all_spectra,pipeline_plan, recalc=true )

line_list_df = prepare_line_list_pass1(linelist_for_ccf_filename, all_spectra, pipeline_plan,  v_center_to_avoid_tellurics=ccf_mid_velocity, Δv_to_avoid_tellurics = 30e3, recalc=true)

(ccfs, v_grid) = ccf_total(order_list_timeseries, line_list_df, pipeline_plan,  mask_scale_factor=10.0, ccf_mid_velocity=ccf_mid_velocity, recalc=true)


line_width = RvSpectMLBase.calc_line_width(v_grid,view(ccfs,:,1),frac_depth=0.05)

line_list_df = prepare_line_list_pass1(linelist_for_ccf_filename, all_spectra, pipeline_plan,  v_center_to_avoid_tellurics=ccf_mid_velocity, Δv_to_avoid_tellurics = line_width)

(ccfs, v_grid) = ccf_total(order_list_timeseries, line_list_df, pipeline_plan,  mask_type=:gaussian, mask_scale_factor=10.0, ccf_mid_velocity=ccf_mid_velocity, recalc=true)

(order_ccfs, v_grid_order_ccfs) = ccf_orders(order_list_timeseries, line_list_df, pipeline_plan)

#rvs_ccf = calc_rvs_from_ccf_total(ccfs, pipeline_plan, v_grid=v_grid, times = order_list_timeseries.times, recalc=true)

#=
if need_to(pipeline_plan,:scalpels)
   rvs_scalpels = map(n->Scalpels.clean_rvs_scalpels(rvs_ccf, ccfs, num_basis=n), 1:5)
   println("RMS RVs cleaned by Scalpels: ",std.(rvs_scalpels) )
   dont_need_to!(pipeline_plan,:scalpels)
end
=#
