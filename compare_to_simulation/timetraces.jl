using LaTeXStrings
import Plots as plt
import NCDatasets as NCDF
using Plots.PlotMeasures
using Dierckx
import LsqFit

#
max_t = 7.0 #[ms]
plot_time_trace = true
plot_z_profile = true
plot_r_profile = true
plot_NEUT = false
last_IPS_step = 32
lowres_timepoints = false
rho_min = 0.1
NBI_off_time = 2.0 #[ms]
fit_decay = false

#
if NBI_off_time > max_t
    fit_decay = false
end

# cd to simulation folder
ROOT = dirname(@__FILE__)
SIM_RESULTS = joinpath(ROOT,"..","simulation_results")

# Kunal - Change it for now to get something working
SIM_RESULTS = "/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/second_round/nneut_1e15_gb_2e18_NBI_800kW_ECH_0kW/simulation_results/"

cd(SIM_RESULTS)

# parse directories, find time steps
dir_names = readdir()
times_str = Vector{String}()
for dir in dir_names
    try
        check_float = eval(Meta.parse.(dir))
        append!(times_str, [dir])
    catch
    else
    end
end
times_float = eval(Meta.parse.(times_str))

sort_idx = sortperm(times_float)
times_float = times_float[sort_idx]
times_str = times_str[sort_idx]

times_str = times_str[times_float .<= last_IPS_step]
times_float = times_float[times_float .<= last_IPS_step]

#
mutable struct CQL_OUTPUT
    restart_time::Array{Float64}
    cql_time::Array{Float64}
    rya::Array{Float64}
    Z::Array{Float64}
    R_midplane::Array{Float64}

    ne::Array{Float64}
    ni::Array{Float64}
    Ee::Array{Float64}
    Ei::Array{Float64}

    Pabs_NBI::Array{Float64}
    Ei_volavg::Array{Float64}
    beta_max::Array{Float64}
    W::Array{Float64}
    ni_lavg::Array{Float64}
    B_midplane::Array{Float64}

    tau_cql::Array{Float64}
    linavg_ni_at_Z::Array{Float64}

    beta0::Array{Float64}
    beta_avg::Array{Float64}
end

function initialize_CQL_OUTPUT(nrya,nz)
    cql_output = CQL_OUTPUT(
        zeros(1),
        zeros(1),
        zeros(1),
        zeros(1),
        zeros(nrya),

        zeros(nz,nrya),
        zeros(nz,nrya),
        zeros(nz,nrya),
        zeros(nz,nrya),

        zeros(1),
        zeros(1),
        zeros(1),
        zeros(1),
        zeros(1),
        zeros(nrya),

        zeros(1),
        zeros(nz),

        zeros(1),
        zeros(1)
    )
    return cql_output
end

mutable struct KN1D_OUTPUT
    RH::Array{Float64}
    rhoH::Array{Float64}
    nH::Array{Float64}
    TH::Array{Float64}
    Pwall::Array{Float64}
    nHwall::Array{Float64}

    Sion::Array{Float64}
    Si_CX::Array{Float64}
    tau_CX::Array{Float64}
end

function initialize_KN1D_OUTPUT(nrya)
    kn1d_output = KN1D_OUTPUT(
        zeros(nrya),
        zeros(nrya),
        zeros(nrya),
        zeros(nrya),
        zeros(1),
        zeros(1),

        zeros(nrya),
        zeros(nrya),
        zeros(1)
    )
    return kn1d_output
end

#
for (time_idx, time) in enumerate(times_float)

    if time_idx == 1
        CQL_FILE = joinpath(".",times_str[time_idx],"components","fp__cql3dm_4","WHAM.nc")
        CQLdat = NCDF.Dataset(CQL_FILE, "r"); 
        #println(CQLdat)
        rya = CQLdat["rya"][:] # normalized radial coord
        Z = CQLdat["z"][:,1] .* 1e-2 # axial distance from midplane [cm] -> [m]
        close(CQLdat)
        NRYA = length(rya)
        NZ = length(Z)
        global cql_output = initialize_CQL_OUTPUT(NRYA,NZ)
        cql_output.rya = rya
        cql_output.Z = Z
        global rho_min_idx = argmin(abs.(cql_output.rya .- rho_min))

    end


    CQL_FILE = joinpath(".",times_str[time_idx],"components","fp__cql3dm_4","WHAM.nc")
    CQLdat = NCDF.Dataset(CQL_FILE, "r")
    Z = CQLdat["z"][:,1] # axial distance from midplane [cm]

    density = CQLdat["densz1"][:,:,:,:] # density [cm^-3] gen_species_dim × zdim × r0dim × tdim
    density = density .* 1e6 # [cm^-3] -> [m^-3]

    #energy = CQLdat["energy"][:,:,:] # FSA energy [keV] gen_species_dim × r0dim × tdim
    energy = CQLdat["energyz"][:,:,:,:] # energy [keV] zdim × gen_species_dim × r0dim × tdim
    energy = permutedims(energy, [2,1,3,4]) # match dimension ordering with "density"
    energy = energy

    energy_FSA = CQLdat["energy"][:,:,:] # energy [keV] gen_species_dim × r0dim × tdim
    beta_max = maximum(sum(CQLdat["betaz"][:,1,:,end],dims=1)) # energy [keV] gen_species_dim × zdim × r0dim × tdim

    powers = CQLdat["powers"][:,:,:,:] .* 1e6 #  rdim × thirteendim × gen_species_dim × tdim [W/cm^3] -> [W/m^3]
    powers_ion_particle = powers[:,6,:,:] #power from ion particle source

    ephiz = CQLdat["ephiz"][:,:,:] .* 1e6 #  zdim × r0dim × tdim Electric potential along field line [kV]

    cql_time = CQLdat["time"][:]            # time in seconds

    R_midplane = CQLdat["solrz"][1,:] .* 1e-2 # radius at midplane [cm] -> [m]

    ni_midplane = CQLdat["densz1"][1,1,:,:] .* 1e6 #[cm-3] -> [m-3]
    dR_midplane = diff(cat([0.0], R_midplane, dims=1))

    r_edge = 0.9
    r_edge_idx = argmin(abs.(r_edge .- cql_output.rya))
    Pabs_NBI = 0.0
    Pabs_edge = 0.0
    try
        Pabs_NBI = CQLdat["sorpw_nbii"][1,end] #[W]
        Pabs_edge = (CQLdat["sorpw_nbii"][1,end] - CQLdat["sorpw_nbii"][1,end-4])
    catch
    else
    end
    println("$(round(Pabs_edge * 1e-3, sigdigits=3))kW of NBI power deposited at edge, out of a total $(round(Pabs_NBI .* 1e-3, sigdigits=3)) deposited.")

    dvol = CQLdat["dvol"][:] .* 1e-6 #[m^3]

    W_i = (CQLdat["energy"][1,:,:] ./ 1.5) .* CQLdat["density"][1,:,:] .* (1.6e-16 * 1e6) # (keV * cm-3) -> (J * m-3)

    W_i_tot = zeros(size(W_i)[2])
    for i in collect(1:size(W_i)[2])
        W_i[:,i] = W_i[:,i] .* dvol
        W_i_tot[i] = cumsum(W_i[:,i])[end]
    end

    W_e = (CQLdat["energy"][2,:,:] ./ 1.5) .* CQLdat["density"][2,:,:] .* (1.6e-16 * 1e6) # (keV * cm-3) -> (J * m-3)
    W_e_tot = zeros(size(W_e)[2])
    for i in collect(1:size(W_e)[2])
        W_e[:,i] = W_e[:,i] .* dvol
        W_e_tot[i] = cumsum(W_e[:,i])[end]
    end

    W = W_i_tot .+ W_e_tot #total plasma energy

    ni_lavg = zeros(size(ni_midplane)[2])
    for i in collect(1:size(ni_midplane)[2])
        ni_lavg[i] = sum(ni_midplane[:,i] .* dR_midplane) / R_midplane[end]
    end

    B_midplane = CQLdat["bmidplne"][:] .* 1e-4 # [Gauss] -> [T]

    _Ei_volavg = zeros(length(cql_time))
    for (cql_time_idx, cql_time_step) in enumerate(cql_time)
        ridx = last.(Tuple.(findall(CQLdat["density"][1,:,cql_time_idx] .* 1e6 .> 1e15)))
        if length(ridx) > 0
            _Ei_volavg[cql_time_idx] = cumsum(energy_FSA[1,ridx,cql_time_idx] .* dvol[ridx] .* CQLdat["density"][1,ridx,cql_time_idx])[end] ./ cumsum(dvol[ridx] .* CQLdat["density"][1,ridx,cql_time_idx])[end]
        end
    end
    tau_cql = cumsum(CQLdat["tauc_code"][1,rho_min_idx:end,end] .* dvol[rho_min_idx:end] .* CQLdat["density"][1,rho_min_idx:end,end])[end] ./ cumsum(dvol[rho_min_idx:end] .* CQLdat["density"][1,rho_min_idx:end,end])[end] .* 1e3 #[ms]
    #println(CQLdat["tauc_code"][1,rho_min_idx:end,end])


    # get line-averaged ion density at each z

    linavg_ni_at_Z = zeros(length(Z))
    for (i,_Z) in enumerate(Z)
        R = CQLdat["solrz"][i,:] .* 1e-2 # radius at midplane [cm] -> [m]
        dR = diff(cat([0.0], R, dims=1))
        #println("A ",CQLdat["densz1"][1,i,:,end])
        #println("B ",dR)
        linavg_ni_at_Z[i] = sum((CQLdat["densz1"][1,i,:,end] .* 1e6) .* dR) / R[end] #[m-3]
    end

    # get beta0
    _beta = sum(CQLdat["betaz"][:,1,:,end],dims=1)[1,:] # energy [keV] gen_species_dim × zdim × r0dim × tdim
    beta0 = _beta[1]

    # get area averaged beta
    darea = CQLdat["darea"][:] .* 1e-4 #[m^2]
    beta_avg = cumsum(_beta .* darea)[end] / cumsum(darea)[end]

    close(CQLdat)

    if time_idx == 1
        cql_output.cql_time = copy(cql_time)
        cql_output.restart_time[1] = cql_time[1]
    else
        cql_time_total = cql_output.cql_time[end] .+ cql_time
        cql_output.cql_time = cat(cql_output.cql_time, cql_time_total, dims=1)
        cql_output.restart_time = cat(cql_output.restart_time, cql_time_total[1], dims = 1)
    end

    for (cql_time_idx, cql_time_step) in enumerate(cql_time)
        cql_output.ni = cat(cql_output.ni, density[1,:,:,cql_time_idx], dims = 3)
        cql_output.ne = cat(cql_output.ne, density[2,:,:,cql_time_idx], dims = 3)
        cql_output.Ei = cat(cql_output.Ei, energy[1,:,:,cql_time_idx], dims = 3)
        cql_output.Ee = cat(cql_output.Ee, energy[2,:,:,cql_time_idx], dims = 3)
    end

    cql_output.R_midplane = cat(cql_output.R_midplane, R_midplane, dims = 2)
    cql_output.Pabs_NBI = cat(cql_output.Pabs_NBI, Pabs_NBI, dims = 1)
    cql_output.beta_max = cat(cql_output.beta_max, beta_max, dims = 1)
    cql_output.beta0 = cat(cql_output.beta0, beta0, dims = 1)
    cql_output.beta_avg = cat(cql_output.beta_avg, beta_avg, dims = 1)
    cql_output.W = cat(cql_output.W, W, dims = 1)
    cql_output.ni_lavg = cat(cql_output.ni_lavg, ni_lavg, dims = 1)
    cql_output.B_midplane = cat(cql_output.B_midplane, B_midplane, dims = 2)
    cql_output.Ei_volavg = cat(cql_output.Ei_volavg, _Ei_volavg, dims = 1)
    cql_output.tau_cql = cat(cql_output.tau_cql, tau_cql, dims = 1)
    cql_output.linavg_ni_at_Z = cat(cql_output.linavg_ni_at_Z, linavg_ni_at_Z, dims = 2)

end

cql_output.ni = cql_output.ni[:,:,2:end]
cql_output.ne = cql_output.ne[:,:,2:end]
cql_output.Ei = cql_output.Ei[:,:,2:end]
cql_output.Ee = cql_output.Ee[:,:,2:end]
cql_output.R_midplane = cql_output.R_midplane[:,2:end]
cql_output.Pabs_NBI = cql_output.Pabs_NBI[2:end]
cql_output.beta_max = cql_output.beta_max[2:end]
cql_output.Ei_volavg = cql_output.Ei_volavg[2:end]
cql_output.W = cql_output.W[2:end]
cql_output.ni_lavg[1] = cql_output.ni_lavg[2]
cql_output.B_midplane = cql_output.B_midplane[:,2:end]
cql_output.tau_cql = cql_output.tau_cql[2:end]
cql_output.linavg_ni_at_Z = cql_output.linavg_ni_at_Z[:,2:end]



##_______________________________________________

if plot_NEUT

    for (time_idx, time) in enumerate(times_float)
        NRH = 100

        if time_idx == 1
            KN1D_FILE = joinpath(".",times_str[time_idx],"components","neut__kn1dc_5","kn1dc.nc")
            KN1Ddat = NCDF.Dataset(KN1D_FILE, "r")
            RH = KN1Ddat["rhoH"][:] # R [m]
            nH = KN1Ddat["nH"][:] # neutral density [m-3]
            if length(RH) != length(nH)
                RH = KN1Ddat["rhoH_lowres"][:] # R [m]
            end
            #println(KN1Ddat)
            close(KN1Ddat)
            #NRH = length(RH)
            global kn1d_output = initialize_KN1D_OUTPUT(NRH)
        end

        #=
        try
            KN1D_FILE = joinpath(".",times_str[time_idx],"components","neut__kn1dc_5","kn1dc.nc")
            KN1Ddat = NCDF.Dataset(KN1D_FILE, "r")
        catch
        else
            KN1D_FILE = joinpath(".",times_str[1],"components","neut__kn1dc_5","kn1dc.nc")
            KN1Ddat = NCDF.Dataset(KN1D_FILE, "r")
        end
        =#
        KN1D_FILE = joinpath(".",times_str[time_idx],"components","neut__kn1dc_5","kn1dc.nc")
        println(KN1D_FILE)
        KN1Ddat = NCDF.Dataset(KN1D_FILE, "r")

        RH = KN1Ddat["rhoH"][:] # R [m]
        nH = KN1Ddat["nH"][:] # neutral density [m-3]
        ne = KN1Ddat["ne_cold"][:] # electron density [m-3]
        println("Edge neutral density is: ", round(nH[end],sigdigits=4), " [m^-3]")
        if length(RH) != length(nH)
            RH = KN1Ddat["rhoH_lowres"][:] # R [m]
        end
        TH = KN1Ddat["TH"][:] .* 1e-3 # neutral temp [eV] -> [keV]
        Pwall = KN1Ddat["Pressure_wall"][1] * 7.5 # [mtorr]
        Sion = KN1Ddat["Sion"][:] # [m^-3 s^-1]
        Si_CX = KN1Ddat["Si_CX"][:] # [m^-3 s^-1]
        nHwall = nH[end] # [mtorr]

        rho_min_for_vol_avg = 0.1
        rho_max_for_vol_avg = copy(KN1Ddat["r_lim"][1])
        rho_max_idx = argmin(abs.(RH .- rho_max_for_vol_avg))
        rho_min_idx = argmin(abs.(RH .- rho_min_for_vol_avg))
        drho = append!([0.0], RH[2:end] .- RH[1:end-1])
        tau_CX = 1e3 .* (2 * sum(drho[rho_min_idx:rho_max_idx] .* RH[rho_min_idx:rho_max_idx] .* ne[rho_min_idx:rho_max_idx] ./ Si_CX[rho_min_idx:rho_max_idx] ) / (rho_max_for_vol_avg^2 - rho_min_for_vol_avg^2))
        
        close(KN1Ddat)

        RH_fine = LinRange(RH[1], RH[end], NRH)

        nH = Spline1D(RH, nH,k=1).(RH_fine)
        TH = Spline1D(RH, TH,k=1).(RH_fine)
        Sion = Spline1D(RH, Sion,k=1).(RH_fine)
        Si_CX = Spline1D(RH, Si_CX,k=1).(RH_fine)

        cql_R_midplane = cql_output.R_midplane[:,time_idx]
        func_R_to_rho = Spline1D(cql_R_midplane, cql_output.rya,k=1,bc="extrapolate")
        rhoH = func_R_to_rho.(RH_fine)
        println("rho at R = 0.15m : ", func_R_to_rho(0.05))

        kn1d_output.rhoH = cat(kn1d_output.rhoH, rhoH, dims = 2)
        kn1d_output.RH = cat(kn1d_output.RH, RH_fine, dims = 2)
        kn1d_output.nH = cat(kn1d_output.nH, nH, dims = 2)
        kn1d_output.TH = cat(kn1d_output.TH, TH, dims = 2)
        kn1d_output.Sion = cat(kn1d_output.Sion, Sion, dims = 2)
        kn1d_output.Si_CX = cat(kn1d_output.Si_CX, Si_CX, dims = 2)

        kn1d_output.Pwall = cat(kn1d_output.Pwall, Pwall, dims = 1)
        kn1d_output.nHwall = cat(kn1d_output.nHwall, nHwall, dims = 1)
        kn1d_output.tau_CX = cat(kn1d_output.tau_CX, tau_CX, dims = 1)
    end

    kn1d_output.rhoH = kn1d_output.rhoH[:,2:end]
    kn1d_output.RH = kn1d_output.RH[:,2:end]
    kn1d_output.nH = kn1d_output.nH[:,2:end]
    kn1d_output.TH = kn1d_output.TH[:,2:end]
    kn1d_output.Sion = kn1d_output.Sion[:,2:end]
    kn1d_output.Si_CX = kn1d_output.Si_CX[:,2:end]
    kn1d_output.tau_CX = kn1d_output.tau_CX[2:end]

    kn1d_output.Pwall = kn1d_output.Pwall[2:end]
    kn1d_output.nHwall = kn1d_output.nHwall[2:end]
end

##_______________________________________________
plt.gr(framestyle = :box, lw = 2, minorgrid = true,
    right_margin = 5mm, left_margin=5mm, bottom_margin=3mm, top_margin=5mm,
    tickfontsize=12,guidefontsize=15,
    fontfamily="Computer Modern", legendfontsize=12)

max_t = max_t .* 1e-3
NBI_off_time = NBI_off_time .* 1e-3
##_______________________________________________
cql_times_idx = last.(Tuple.(findall(cql_output.cql_time .<= max_t)))
restart_times_idx = last.(Tuple.(findall(cql_output.restart_time .<= max_t)))

if plot_time_trace
    p_den0 = plt.plot(yscale=:log10)
    plt.plot!(p_den0, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.ni[1,rho_min_idx,cql_times_idx], c="black", label="i")
    plt.plot!(p_den0, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.ne[1,rho_min_idx,cql_times_idx], c="red", label="e")
    plt.vline!(cql_output.restart_time .* 1e3, lw=0.6, c="blue", linestyle=:dash, label = nothing)
    plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
    plt.xlabel!("t [ms]")
    plt.ylabel!(L"$\left[ m^{-3} \right]$")
    plt.title!("On-axis density")
    plt.xlims!(0,max_t .* 1e3)
    #plt.ylims!(0,5e19)

    ##
    p_dena = plt.plot(legend=false, yscale=:log10)
    plt.plot!(p_dena, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.ni[1,end,cql_times_idx], c="black", label="i")
    plt.plot!(p_dena, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.ne[1,end,cql_times_idx], c="red", label="e")
    plt.vline!(cql_output.restart_time .* 1e3, lw=0.6, c="blue", linestyle=:dash)
    plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
    plt.xlabel!("t [ms]")
    plt.ylabel!(L"$\left[ m^{-3} \right]$")
    plt.title!("Edge density")
    plt.xlims!(0,max_t .* 1e3)
    #plt.ylims!(0,2e19)

    p_den_together = plt.plot(p_den0, p_dena, layout = (2,1))
    plt.display(p_den_together)
    ###_______________________________________________
    p_E0 = plt.plot()
    plt.plot!(p_E0, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.Ei[1,rho_min_idx,cql_times_idx], c="black", label="i")
    plt.plot!(p_E0, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.Ee[1,rho_min_idx,cql_times_idx], c="red", label="e")
    plt.vline!(cql_output.restart_time .* 1e3, lw=0.6, c="blue", linestyle=:dash, label = nothing)
    plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
    plt.xlabel!("t [ms]")
    plt.ylabel!("[keV]")
    plt.title!("On-axis Energy")
    plt.xlims!(0,max_t .* 1e3)
    plt.ylims!(0,Inf)

    p_Ea = plt.plot(legend=false)
    plt.plot!(p_Ea, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.Ei[1,end,cql_times_idx], c="black", label="i")
    plt.plot!(p_Ea, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.Ee[1,end,cql_times_idx], c="red", label="e")
    plt.vline!(cql_output.restart_time .* 1e3, lw=0.6, c="blue", linestyle=:dash)
    plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
    plt.xlabel!("t [ms]")
    plt.ylabel!("[keV]")
    plt.title!("Edge Energy")
    plt.xlims!(0,max_t .* 1e3)
    plt.ylims!(0,Inf)

    p_E_together = plt.plot(p_E0, p_Ea, layout = (2,1))
    plt.display(p_E_together)

    println(" ")
    p_Eivolavg = plt.plot(legend=false)
    plt.plot!(p_Eivolavg, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.Ei_volavg[cql_times_idx], c="black")
    plt.vline!(cql_output.restart_time .* 1e3, lw=0.6, c="blue", linestyle=:dash, label = nothing)
    plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
    plt.xlabel!("t [ms]")
    plt.ylabel!("[keV]")
    plt.title!(L"\langle E_{\textrm{i}} \rangle")
    plt.xlims!(0,max_t .* 1e3)
    plt.ylims!(0,Inf)
    plt.display(p_Eivolavg)

    p_W = plt.plot(legend=true)
    plt.plot!(p_W, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.W[cql_times_idx], c="black", label = nothing)
    plt.vline!(cql_output.restart_time .* 1e3, lw=0.3, c="blue", linestyle=:dash, label = nothing)
    plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)

    if fit_decay
        #off = last.(Tuple.(findall(cql_output.cql_time .<= NBI_off_time)))
        cql_times_after_NBI_off = last.(Tuple.(findall(cql_output.cql_time .>= NBI_off_time)))
    
        t_data = cql_output.cql_time[cql_times_after_NBI_off] .* 1e3
        y_data = cql_output.W[cql_times_after_NBI_off]
        model(t, p) = cql_output.W[cql_times_after_NBI_off[1]] * exp.(-t/p[1])
        fit = LsqFit.curve_fit(model, t_data .- t_data[1], y_data, [0.03])
        println("Decay time : ", round(-fit.param[1], sigdigits=3), " [ms]")
        _label = L"\tau = "*" $(round(fit.param[1], sigdigits=2)) [ms]"
        plt.plot!(p_W, t_data, model.(t_data .- t_data[1], Ref(fit.param)), c="red", label=_label)
    end
    
    plt.xlabel!("t [ms]")
    plt.ylabel!("[J]")
    plt.title!("Plasma stored energy")
    plt.xlims!(0,max_t .* 1e3)
    plt.ylims!(0,Inf)
    plt.display(p_W)
end

##_______________________________________________
if plot_z_profile
    p_den0i_v_Z = plt.plot(legend=:outerright, size = (500,500),  colorbartitle = "\n t [ms]")
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_den0i_v_Z, cql_output.Z, cql_output.ni[:,rho_min_idx,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Core ion density $\left[ m^{-3} \right]$")
        plt.ylims!(0,1e19)
        plt.xlabel!("Z [m]")
    end

    p_den0e_v_Z = plt.plot(legend=:outerright, size = (500,500),  colorbartitle = "\n t [ms]")
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_den0e_v_Z, cql_output.Z, cql_output.ne[:,rho_min_idx,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Core electron density $\left[ m^{-3} \right]$")
        plt.ylims!(0,1e19)
        plt.xlabel!("Z [m]")
    end

    p_den0_v_Z = plt.plot(p_den0i_v_Z, p_den0e_v_Z, layout = (2,1), legend = false)
    plt.display(p_den0_v_Z)

    ##_______________________________________________
    p_linavgden_v_z = plt.plot(size = (500,300),  colorbartitle = "\n t [ms]", legend=false,colorbar=true)
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.restart_time))
        plt.plot!(p_linavgden_v_z, cql_output.Z, cql_output.linavg_ni_at_Z[:,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Line averaged ion density $\left[ m^{-3} \right]$")
        plt.xlabel!("Z [m]")
        plt.xlims!(0,0.75)
        #plt.ylims!(0,2e19)
    end
    plt.display(p_linavgden_v_z)

    ##_______________________________________________
    p_denai_v_Z = plt.plot(legend=:outerright, size = (500,500),  colorbartitle = "\n t [ms]")
    if lowres_timepoints
        for targ_time in cql_output.restart_time[restart_times_idx]
            targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
            plt.plot!(p_denai_v_Z, cql_output.Z, cql_output.ni[:,end,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
            plt.title!(L"Edge ion density $\left[ m^{-3} \right]$")
            plt.ylims!(0,1e19)
            plt.xlabel!("Z [m]")
        end
    else
        for (targ_time_idx, targ_time) in enumerate(cql_output.cql_time[cql_times_idx])
            #targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
            plt.plot!(p_denai_v_Z, cql_output.Z, cql_output.ni[:,end,targ_time_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
            plt.title!(L"Edge ion density $\left[ m^{-3} \right]$")
            plt.ylims!(0,1e19)
            plt.xlabel!("Z [m]")
        end
    end

    p_denae_v_Z = plt.plot(legend=:outerright, size = (500,500),  colorbartitle = "\n t [ms]")
    if lowres_timepoints
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_denae_v_Z, cql_output.Z, cql_output.ne[:,end,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Edge electron density $\left[ m^{-3} \right]$")
        plt.ylims!(0,1e19)
        plt.xlabel!("Z [m]")
    end
    else
    for (targ_time_idx, targ_time) in enumerate(cql_output.cql_time[cql_times_idx])
        #targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_denae_v_Z, cql_output.Z, cql_output.ne[:,end,targ_time_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Edge electron density $\left[ m^{-3} \right]$")
        plt.ylims!(0,1e19)
        plt.xlabel!("Z [m]")
    end
    end

    p_dena_v_Z = plt.plot(p_denai_v_Z, p_denae_v_Z, layout = (2,1), legend = false)
    plt.display(p_dena_v_Z)

    ##_______________________________________________
    p_E0i_v_Z = plt.plot(legend=:outerright, size = (500,500),  colorbartitle = "\n t [ms]")
    if lowres_timepoints
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_E0i_v_Z, cql_output.Z, cql_output.Ei[:,rho_min_idx,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Core ion energy [keV]")
        plt.xlabel!("Z [m]")
    end
    else
    for (targ_time_idx, targ_time) in enumerate(cql_output.cql_time[cql_times_idx])
        #targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_E0i_v_Z, cql_output.Z, cql_output.Ei[:,rho_min_idx,targ_time_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Core ion energy [keV]")
        plt.xlabel!("Z [m]")
    end
    end

    p_E0e_v_Z = plt.plot(legend=:outerright, size = (500,500),  colorbartitle = "\n t [ms]")
    if lowres_timepoints
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_E0e_v_Z, cql_output.Z, cql_output.Ee[:,rho_min_idx,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Core electron energy [keV]")
        plt.xlabel!("Z [m]")
    end
    else
    for (targ_time_idx, targ_time) in enumerate(cql_output.cql_time[cql_times_idx])
        #targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_E0e_v_Z, cql_output.Z, cql_output.Ee[:,rho_min_idx,targ_time_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Core electron energy [keV]")
        plt.xlabel!("Z [m]")
    end
    end

    p_E0_v_Z = plt.plot(p_E0i_v_Z, p_E0e_v_Z, layout = (2,1), legend = false)
    plt.display(p_E0_v_Z)

    ##_______________________________________________
    p_Eai_v_Z = plt.plot(legend=:outerright, size = (500,500),  colorbartitle = "\n t [ms]")
    if lowres_timepoints
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_Eai_v_Z, cql_output.Z, cql_output.Ei[:,end,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Edge ion energy [keV]")
        plt.xlabel!("Z [m]")
    end
    else
    for (targ_time_idx, targ_time) in enumerate(cql_output.cql_time[cql_times_idx])
        #targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_Eai_v_Z, cql_output.Z, cql_output.Ei[:,end,targ_time_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Edge ion energy [keV]")
        plt.xlabel!("Z [m]")
    end
    end

    p_Eae_v_Z = plt.plot(legend=:outerright, size = (500,500),  colorbartitle = "\n t [ms]")
    if lowres_timepoints
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_Eae_v_Z, cql_output.Z, cql_output.Ee[:,end,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Edge electron energy [keV]")
        plt.xlabel!("Z [m]")
    end
    else
    for (targ_time_idx, targ_time) in enumerate(cql_output.cql_time[cql_times_idx])
        #targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_Eae_v_Z, cql_output.Z, cql_output.Ee[:,end,targ_time_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Edge electron energy [keV]")
        plt.xlabel!("Z [m]")
    end
    end

    p_Ea_v_Z = plt.plot(p_Eai_v_Z, p_Eae_v_Z, layout = (2,1), legend = false)
    plt.display(p_Ea_v_Z)
end

##_______________________________________________
if plot_r_profile
    p_deni_v_rya = plt.plot(legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm)
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_deni_v_rya, cql_output.rya, log10.(cql_output.ni[1,:,targ_idx]), label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Ion density $\left[m^{-3} \right]$")
        plt.xlabel!(L"\rho")
    end
    plt.xlims!(rho_min,Inf)
    #plt.ylims!(0,Inf)
    plt.display(p_deni_v_rya)

    ##
    p_dene_v_rya = plt.plot(legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm)
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_dene_v_rya, cql_output.rya, log10.(cql_output.ne[1,:,targ_idx]), label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Electron density $\left[m^{-3} \right]$")
        plt.xlabel!(L"\rho")
    end
    plt.xlims!(rho_min,Inf)
    #plt.ylims!(0,Inf)
    plt.display(p_dene_v_rya)



    ##_______________________________________________
    p_Ei_v_rya = plt.plot(legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=5mm)
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
        plt.plot!(p_Ei_v_rya, cql_output.rya, cql_output.Ei[1,:,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Ion energy [keV]")
        plt.xlabel!(L"\rho")
    end
    #plt.ylims!(0,1)
    plt.xlims!(rho_min,Inf)
    plt.display(p_Ei_v_rya)

    p_Ee_v_rya = plt.plot(legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=5mm)
    for targ_time in cql_output.restart_time[restart_times_idx]
        targ_idx = argmin(abs.(targ_time .- cql_output.cql_time)) .+ 0
        plt.plot!(p_Ee_v_rya, cql_output.rya, cql_output.Ee[1,:,targ_idx], label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Electron energy [keV]")
        plt.xlabel!(L"\rho")
    end
    plt.ylims!(0,Inf)
    plt.xlims!(rho_min,Inf)
    plt.display(p_Ee_v_rya)
end
##_______________________________________________

p_PabsNBI_v_time = plt.plot(legend=false)
plt.scatter!(p_PabsNBI_v_time, cql_output.restart_time[restart_times_idx] .* 1e3, cql_output.Pabs_NBI[restart_times_idx] .* 1e-3, c="black")
plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
plt.xlabel!("t [ms]")
plt.ylabel!("[kW]")
plt.title!("Absorbed NBI power")
plt.xlims!(0,max_t .* 1e3)
plt.ylims!(0,Inf)
plt.display(p_PabsNBI_v_time)

##_______________________________________________
if plot_NEUT
    ##_______________________________________________

    p_denH_v_rya = plt.plot(legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm, yscale=:log10)
    for (targ_time_idx, targ_time) in enumerate(cql_output.restart_time[restart_times_idx])
        _rhoH = (kn1d_output.rhoH[:,targ_time_idx])[kn1d_output.rhoH[:,targ_time_idx] .< 1.0]
        _nH = (kn1d_output.nH[:,targ_time_idx])[kn1d_output.rhoH[:,targ_time_idx] .< 1.0]
        plt.plot!(p_denH_v_rya, _rhoH, _nH, label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Neutral density $\left[ m^{-3} \right]$")
        plt.xlabel!(L"\rho")
    end
    plt.display(p_denH_v_rya)

    p_TH_v_rya = plt.plot(legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm)
    for (targ_time_idx, targ_time) in enumerate(cql_output.restart_time[restart_times_idx])
        _rhoH = (kn1d_output.rhoH[:,targ_time_idx])[kn1d_output.rhoH[:,targ_time_idx] .< 1.0]
        _TH = (kn1d_output.TH[:,targ_time_idx])[kn1d_output.rhoH[:,targ_time_idx] .< 1.0]
        plt.plot!(p_TH_v_rya, _rhoH, _TH, label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!("Neutral temp [keV]")
        plt.xlabel!(L"\rho")
    end
    plt.display(p_TH_v_rya)

    kn1d_output.Sion[kn1d_output.Sion .== 0.0] .= NaN
    p_Sion_v_rya = plt.plot(minorgrid=true,yscale=:log10, legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm)
    for (targ_time_idx, targ_time) in enumerate(cql_output.restart_time[restart_times_idx])
        _rhoH = (kn1d_output.rhoH[:,targ_time_idx])[kn1d_output.rhoH[:,targ_time_idx] .< 1.0]
        _Sion = (kn1d_output.Sion[:,targ_time_idx])[kn1d_output.rhoH[:,targ_time_idx] .< 1.0]
        plt.plot!(p_Sion_v_rya, _rhoH, _Sion, label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        plt.title!(L"Ionization rate $\left[m^{-3} s^{-1}\right]$")
        plt.xlabel!(L"\rho")
    end
    plt.display(p_Sion_v_rya)

    kn1d_output.Si_CX[kn1d_output.Si_CX .== 0.0] .= NaN
    p_Si_CX_v_rya = plt.plot(minorgrid=true,yscale=:log10, legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm)
    for (targ_time_idx, targ_time) in enumerate(cql_output.restart_time[restart_times_idx])
        _rhoH = (kn1d_output.rhoH[:,targ_time_idx])[kn1d_output.rhoH[:,targ_time_idx] .< 1.0]
        _Si_CX = (kn1d_output.Si_CX[:,targ_time_idx])[kn1d_output.rhoH[:,targ_time_idx] .< 1.0]
        plt.plot!(p_Si_CX_v_rya, _rhoH, _Si_CX, label=round(targ_time,sigdigits=2), linez = targ_time * 1e3)
        #plt.title!(L"\textrm{CX rate }\left[m^{-3} s^{-1}\right]")
        plt.title!(L"CX rate $\left[m^{-3} s^{-1}\right]$")
        plt.xlabel!(L"\rho")
    end
    plt.display(p_Si_CX_v_rya)

    p_Pwall_v_time = plt.plot(legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm)
    plt.plot!(p_Pwall_v_time, cql_output.restart_time[restart_times_idx] .* 1e3, kn1d_output.Pwall[restart_times_idx], color="black")
    plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
    plt.xlabel!("t [ms]")
    plt.ylabel!("[mtorr]")
    plt.title!("Wall neutral pressure")
    plt.xlims!(0,max_t*1e3)
    plt.display(p_Pwall_v_time)

    #Twall = 0.0257 #[eV] or 298K
    Twall = 4.118e-21 #[J]

    p_Pwall_v_time = plt.plot(legend=false, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm)
    plt.plot!(p_Pwall_v_time, cql_output.restart_time[restart_times_idx] .* 1e3, kn1d_output.nHwall[restart_times_idx] .* (7.5*Twall), color="black")
    plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
    plt.xlabel!("t [ms]")
    plt.ylabel!("[mtorr]")
    plt.title!("Wall neutral pressure assuming Twall=298K")
    plt.xlims!(0,max_t*1e3)
    plt.display(p_Pwall_v_time)
end


p_beta_v_time = plt.plot(legend=true, size = (500,300),  colorbartitle = "\n t [ms]", colorbar=true,right_margin=10mm)
plt.plot!(p_beta_v_time, cql_output.restart_time[restart_times_idx] .* 1e3, cql_output.beta_max[restart_times_idx], color="black", linestyle=:dash, label=L"\textrm{max}\left( \beta \right)")
plt.plot!(p_beta_v_time, cql_output.restart_time[restart_times_idx] .* 1e3, cql_output.beta0[restart_times_idx], color="red", linestyle=:solid, label=L"\beta_{0}")
plt.plot!(p_beta_v_time, cql_output.restart_time[restart_times_idx] .* 1e3, cql_output.beta_avg[restart_times_idx], color="black", linestyle=:solid, label=L"\langle \beta \rangle")
plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
plt.xlabel!("t [ms]")
plt.title!("Max beta")
plt.xlims!(0,max_t*1e3)
plt.display(p_beta_v_time)

## line averaged ion density vs time _______________________________________________
p_ni_lavg = plt.plot(legend=false, colorbar=false)
plt.plot!(p_ni_lavg, cql_output.cql_time[cql_times_idx] .* 1e3, cql_output.ni_lavg[cql_times_idx] ./ 1e19, color="black")
plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
plt.xlabel!("t [ms]")
plt.ylabel!(L"\left[ 10^{19} \textrm{m}^{-3} \right]")
plt.title!(L"Line averaged $n_{\textrm{i}}$")
plt.xlims!(0,max_t .* 1e3)
#plt.ylims!(0,1)
plt.display(p_ni_lavg)

## plot tau
Bm = 17 #Tesla
E_b = 25 #keV
L_p = 1 #m
p_tau = plt.plot()
R_m = Bm ./ cql_output.B_midplane[1,:]

tau_p = zeros(length(cql_output.restart_time[restart_times_idx])) #[ms]
tau_GDT = zeros(length(cql_output.restart_time[restart_times_idx])) #[ms]
tau_CX = zeros(length(cql_output.restart_time[restart_times_idx]))
tau_ii = zeros(length(cql_output.restart_time[restart_times_idx]))

for (tidx, targ_time) in enumerate(cql_output.restart_time[restart_times_idx])
    targ_idx = argmin(abs.(targ_time .- cql_output.cql_time))
    #tau_p[tidx] = 250*(E_b / 1e2)^(1.5) * log10.(R_m[tidx]) ./ (cql_output.ni_lavg[targ_idx] ./ 1e20)
    #tau_p[tidx] = 250*(cql_output.Ei_volavg[targ_idx] / 1.5 / 1e2)^(1.5) * log10.(R_m[tidx]) ./ (cql_output.ni_lavg[targ_idx] ./ 1e20)
    tau_p[tidx] = 250*(25 / 1.5 / 1e2)^(1.5) * log10.(R_m[tidx]) ./ (cql_output.ni_lavg[targ_idx] ./ 1e20)
    nu_ii = 4.8e-8*(2)^(-0.5)*(cql_output.Ei_volavg[targ_idx] * 1e3 / 1.5)^(-1.5) * 15 * (cql_output.ni_lavg[targ_idx] ./ 1e6)
    tau_ii[tidx] = (1/nu_ii) * 1e3 * log10.(R_m[tidx]) # [ms]

    tau_GDT[tidx] = 5.2 * R_m[tidx] * L_p * (cql_output.Ei_volavg[targ_idx] / 1.5)^(-0.5) / 1e3
end

tau_p[tau_p .<= 0.0] .= NaN
tau_p[tau_p .== 0.0] .= NaN
tau_GDT[tau_GDT .== Inf] .= NaN
tau_GDT[tau_GDT .== 0.0] .= NaN

p_taus = plt.plot(legend=:outerright, size = (500,300), colorbar=true,right_margin=10mm, yscale=:log10)
plt.plot!(p_taus, cql_output.restart_time[restart_times_idx] .* 1e3, tau_p, color="black",label="classic")
plt.plot!(p_taus, cql_output.restart_time[restart_times_idx] .* 1e3, tau_GDT, color="red",label="GDT")
plt.plot!(p_taus, cql_output.restart_time[restart_times_idx] .* 1e3, tau_ii, color="blue",label="i-i")
#if plot_NEUT
#    plt.plot!(p_taus, cql_output.restart_time[restart_times_idx] .* 1e3, kn1d_output.tau_CX[restart_times_idx], color="blue",label="CX")
#end
plt.plot!(p_taus, cql_output.restart_time[restart_times_idx] .* 1e3, cql_output.tau_cql[restart_times_idx], color="green",label="cql")
plt.vline!([NBI_off_time .* 1e3], lw=0.6, c="red", linestyle=:dash, label = nothing)
plt.xlabel!("t [ms]")
plt.ylabel!("[ms]")
plt.title!("Confinement times")
plt.xlims!(0,max_t*1e3)
plt.display(p_taus)