linelist_for_ccf_filename = "G8.espresso.mas"
ccf_mid_velocity = -5e3
max_spectra_to_use = 100
df_files
df_files_use = df_files |>
    @filter( _.target == fits_target_str ) |>
    @take(max_spectra_to_use) |>
    DataFrame
