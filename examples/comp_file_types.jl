

for fn_fits in fns
    local file = FITS(fn_fits)
    local l1 = read(file[2],"wavelength")
    local f1 = read(file[2],"spectrum")
    local v1 = read(file[2],"uncertainty")
    fn_jld2 = replace(fn_fits,"fits"=>"jld2")
    println(fn_jld2)
    @save fn_jld2 λ=l1 flux=f1 var=v1
end


@time begin
    for fn_fits in fns
    fn_jld2 = replace(fn_fits,"fits"=>"jld2")
    local f = jldopen(fn_jld2, "r")
    local l1 = f["λ"]
    local f1 = f["flux"]
    local v1 = f["var"]
    println("sum(l) = ", sum(view(l1[:,10:20],.!isnan.(l1[:,10:20]))),
    " sum(f) = ", sum(view(f1[:,10:20],.!isnan.(f1[:,10:20]))),
    " sum(v) = ", sum(view(v1[:,10:20],.!isnan.(v1[:,10:20]))) )
end
end

@time begin for fn_fits in fns
    local file = FITS(fn_fits)
    local l1 = read(file[2],"wavelength")
    local f1 = read(file[2],"spectrum")
    local v1 = read(file[2],"uncertainty")
    println("sum(l) = ", sum(view(l1[:,10:20],.!isnan.(l1[:,10:20]))),
    " sum(f) = ", sum(view(f1[:,10:20],.!isnan.(f1[:,10:20]))),
    " sum(v) = ", sum(view(v1[:,10:20],.!isnan.(v1[:,10:20]))) )
end
end

@time begin
    for fn_fits in fns
    fn_jld2 = replace(fn_fits,"fits"=>"jld2")
    local f = h5open(fn_jld2, "r")
    local l1 = readmmap(f["λ"])
    local f1 = readmmap(f["flux"])
    local v1 = readmmap(f["var"])
    println("sum(l) = ", sum(view(l1[:,10:20],.!isnan.(l1[:,10:20]))),
    " sum(f) = ", sum(view(f1[:,10:20],.!isnan.(f1[:,10:20]))),
    " sum(v) = ", sum(view(v1[:,10:20],.!isnan.(v1[:,10:20]))) )
end
end
