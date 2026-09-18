from axuv_postprocessing.axuv_read import AXUV_Read
from axuv_postprocessing.axuv_analysis import AXUV_analysis
# --- user settings ---
shotnum                = [260302079]#251007035]#, 251009049, 251009050, 251009055]#[251007014, 251007015, 251007016, 251007017]

# shotnum                = [250324073, 250324079,
#                           250324071, 250324075,
#                           250324074, 250324076,
#                           250324077, 250324072]#251007035]#, 251009049, 251009050, 251009055]#[251007014, 251007015, 251007016, 251007017]
selected_indices       = [1, 2, 3]    # Diode Array
plasmaedge             = 0.25         # m
time_window            = [0, 20]      # ms

# --- diode selection ---
diag    = "axuv"
# --- prepare jobs ---

results = {}
analysis_freq_t_window = [1, 50]      # ms, kHz
for sn in shotnum:
    print(f"Preparing shot {sn}")
    axuv = AXUV_Read(
        sn,
        selected_indices_input=selected_indices,
        plasma_edge=plasmaedge,
        recal=[-1.5, 0],
    )

    diode_keys = list(axuv.diodes.keys())
    if selected_indices is not None:
        selected_keys = [diode_keys[i - 1] for i in selected_indices]
    else:
        selected_keys = diode_keys

    analy_tw = analysis_freq_t_window[0]
    analy_f  = analysis_freq_t_window[1]

    for key in selected_keys:
        D = axuv.diodes[key]

        # run analysis (serial)
        D_result = AXUV_analysis(
            D,
            time_window,
            fft_dig_khz=analy_f,
            FFT_time_window=analy_tw,
        )

        results.setdefault(sn, {})[key] = D_result

print("✅ All jobs completed!")
print("✅ Results stored in dictionary `results[sn][key]`")
import numpy as np

def rd_to_npz_dict(rd):
    """Convert one AXUV_analysis result to a flat dict with better names."""
    d = {}

    # --- Scalars / metadata ---
    d["shot"] = int(rd.shotnum)
    d["sample_rate_MHz"] = int(rd.digi_Mhz)
    d["plasma_edge_m"] = float(rd.def_plasma_edge)
    d["plasma_length_m"] = float(rd.AXUV_plasma_len) / 100.0  # you said /100 -> meters

    d["diode_id"] = str(rd.diode_name)          # e.g. DIODEARRAY3
    d["diode_description"] = str(rd.da_name)    # your description string
    # diag_view_chord might be array or scalar; store as-is
    d["view_chord_deg"] = np.array(rd.diag_view_chord)

    d["filter_transmission"] = float(rd.attenuation)  # (confirm: 0-1 or %)
    d["power_cal_for_signal_padded_nA_kw_per_nA"] = float(rd.kW_nA)

    # --- Time series / derived quantities ---
    d["time_s"] = np.array(rd.time_coor)
    d["centroid_cm"] = np.array(rd.centroid_hr)
    d["radius_cm"] = np.array(rd.radius_hr)
    d["m1_polarization_ratio"] = np.array(rd.c1n1_cm1)

    # detv_list is a dict (often not uniform). Store as object array.
    d["macro_stability_chi_sample_rate_khz"] = np.array(rd.chi_signal_dig_khz)
    d["macro_stability_chi_and_time_cm2_s"] = np.array(rd.detv_list, dtype=object)

    # --- Mid-raw chord data ---
    d["impact_b_raw_m"] = np.array(rd.R_impact)
    d["signal_raw_nA"] = np.array(rd.signal)

    # --- Padded / G-filling ---
    d["impact_b_padded_m"] = np.array(rd.x_padded)
    d["signal_padded_nA"] = np.array(rd.y_padded)

    # --- Regularized / smoothed ---
    d["impact_b_smooth_m"] = np.array(rd.dnumhr)
    d["signal_smooth_nA"] = np.array(rd.signal_hr)
    d["signal_sym_smooth_nA"] = np.array(rd.sym_axuvData_cal_hr_smooth)
    d["signal_asym_smooth_nA"] = np.array(rd.asy_axuvData_cal_hr_smooth)

    # Positive-half b
    dnumhr = np.array(rd.dnumhr)
    d["impact_b_smooth_half_m"] = dnumhr[len(dnumhr)//2:]

    # Abel inversion emissivity profile
    # You wrote radial_profile.T is what you want; store that explicitly.
    d["emissivity_profile_mW_per_cm3"] = np.array(rd.radial_profile*1000).T

    return d


# ---- save many shots for one diode (DA index or explicit key) ----

for j in range(len(selected_indices)):
    DA = selected_indices[j]
    out = {}
    for sn in shotnum:
        diode_keys = list(results[sn].keys())
        key = diode_keys[DA - 1]
        rd = results[sn][key]

        # prefix each field with shot id so one NPZ can hold multiple shots
        d = rd_to_npz_dict(rd)
        for k, v in d.items():
            out[f"s{sn}_{k}"] = v

    np.savez("AXUV_results"+str(shotnum[0])+"_"+str(shotnum[-1])+"_DA"+str(DA)+".npz", **out)
    print("Saved:", "AXUV_results"+str(shotnum[0])+"_"+str(shotnum[-1])+"_DA"+str(DA)+".npz")