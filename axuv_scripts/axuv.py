# Directory Structure
# axuv_postprocessing/
# ├── __init__.py
# ├── axuv.py  # main script that takes shot number
# ├── axuv_read.py
# ├── axuv_analysis.py
# ├── axuv_plots.py
# └── object_init.py

# Inside axuv.py
from axuv_postprocessing.axuv_read import AXUV_Read
from axuv_postprocessing.axuv_analysis import AXUV_analysis
from axuv_postprocessing.axuv_plots import AXUV_plots

import MDSplus as mds
import matplotlib.pyplot as plt
# Optional utility to parse input
import argparse

axuvcc_diode_list = [1,2,3]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXUV Postprocessing Pipeline")
    parser.add_argument('-s', '--shotnum', type=int, nargs='+', required=False, help="Shot number(s)")
    parser.add_argument('-p','--plot', action='store_true',       help='Plot results after analysis')
    parser.add_argument('-m','--modedecomp', action='store_true', help='Do mode decomposition')
    def parse_diode_list(value):
        if value is None:
            return None
        try:
            return [int(v) for v in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid diode list: '{value}'. Use integers or comma-separated list like 1,2,3"
            )
    
    parser.add_argument(
        "-d", "--D",
        type=parse_diode_list,
        required=False,
        default=None,
        help="Diode index (1-based). Can be a single integer (e.g., 1) or comma-separated list (e.g., 1,2,3). "
             "If not set, defaults to [1,2,3]."
    )

    parser.add_argument('-tw', '--timewindow',
                        type=float, nargs=2,
                        required=False,
                        default=[0, 20],
                        help='Define the start and end of the analysis time window.e.g -tw st et'
                        )
    parser.add_argument('-pe', '--plasmaedge',
                        type=float,
                        default=None,
                        help='Optional plasma edge radius. If not set, will be read from tree or fall back to 0.25.'
                        )
    parser.add_argument('-rc', '--recal_in',
                        type=float, nargs=2,
                        default=None,
                        help='perspective angel of AXUV relative to Vessel, # [deg m]; -rc deg m'
                        )
    parser.add_argument('-pl', '--plot_cm_R_limit',
                        type=float, nargs=2,
                        default=None,
                        help='Set the centroid and radius plot limit. If not provided, the value will be CM[-4cm,4cm] R[0, plasmaedge]; -pl CM R'
                        )
    parser.add_argument('-mla', '--modedecomp_factor_A',
                        type=float, nargs=2,
                        required=False,
                        default=[None,None],
                        help='amp_factor and yaxis factor,  for plot in Abel decomp.e.g -mla amp yaxis'
                        )
    parser.add_argument('-ml', '--modedecomp_factor',
                        type=float, nargs=2,
                        required=False,
                        default=[None,None],
                        help='amp_factor and yaxis factor,  for plot in RAW decomp.e.g -ml amp yaxis'
                        )
    parser.add_argument('-af', '--analysis_freq_t_window',
                        type=float, nargs=2,
                        required=False,
                        default=[1, 50],
                        help='time window and FFT digi_f,  for plot in analysis overview plot.e.g -df sliding timewindow; digit_f'
                        )
    parser.add_argument('-df', '--modedecomp_freq_t_window',
                        type=float, nargs=2,
                        required=False,
                        default=[1, 50],
                        help='time window and FFT digi_f,  for plot in RAW and Abel decomp.e.g -df timewindow digit_f'
                        )
    parser.add_argument('-cmrpf', '--cm_r_plot_f',
                        type=float, nargs=2,
                        required=False,
                        default=[10, 1.5],
                        help='freq. for showing plots, max display time;default=10 kHz, 1.5x max time'
                        )
    # CHANGE 1: Add --write-mds flag for MDSPlus saving
    parser.add_argument('--write-mds', type=bool, nargs='?',
                        const=True, default=True,
                        help='Write derived quantities to MDSPlus tree (default: True)'
                        )
    args = parser.parse_args()

# Kai to do: call postproc_axuv.py -s $shotnum
# try: except statement to attempt to read data from the processed try
# so that you don't need a knob. 
#test

    if args.shotnum:

        for shot in args.shotnum:
            shotnum = shot
            print(f"Processing shot {shot}")

            # Prepare diode selection input
            if args.D is not None:
                selected_indices = args.D
            else:
                selected_indices = axuvcc_diode_list  # default explicit choice

            
            plasmaedge = args.plasmaedge
            print(args.recal_in)
            axuv = AXUV_Read(shotnum, tree='wham', selected_indices_input=selected_indices,
                             plasma_edge=plasmaedge, recal = args.recal_in)

            diode_keys = list(axuv.diodes.keys())

            if args.D is not None:
                # map each integer index (1-based) to the corresponding diode key
                selected_keys = [diode_keys[i-1] for i in args.D]
            else:
                # no diode specified → take all
                selected_keys = diode_keys




            for i, key in enumerate(selected_keys):
                print(f"Processing diode: {key}")
                D = axuv.diodes[key]
                #time_window = [0, 20]  # ms

                time_window   = args.timewindow
                max_tw_factor = args.cm_r_plot_f[1]
                tw = [time_window[0],time_window[0]+int((time_window[1]-time_window[0])*max_tw_factor)]
                analy_tw = args.analysis_freq_t_window[0]
                analy_f  = args.analysis_freq_t_window[1]
                D_result = AXUV_analysis(D, time_window, hard_noise=0.002, fft_dig_khz = analy_f, FFT_time_window = analy_tw)
                D_plots = AXUV_plots(D_result, plt_freq=args.cm_r_plot_f[0], walltime_window=tw, cm_R_plot_limit=args.plot_cm_R_limit, diode_key=key, write_mds=args.write_mds)  # CHANGE 2: Pass diode_key and write_mds
                if args.plot:
                    plt.show()
                else:
                    plt.close("all")
                if args.modedecomp:
                    print(f"decomposing Abel Signal")
                    # First decomposition: radial profile
                    signal_abel = D_result.radial_profile.T
                    x_abel = D_result.dnumhr[D_result.dnumhr.shape[0] // 2:] * 100
                    decomp_tw = args.modedecomp_freq_t_window[0]
                    decomp_f  = args.modedecomp_freq_t_window[1]
                    D_plots.plot_mode_decomp(x_abel, signal_abel,
                                             m_name='A', amp_factor=args.modedecomp_factor_A[0],
                                             yaxis_factor=args.modedecomp_factor_A[1],
                                             time_window=decomp_tw, fft_dig_khz=decomp_f)
                    if args.plot:
                        plt.show()
                    else:
                        plt.close("all")

                    # Second decomposition: original padded signal
                    signal_direct = D_result.y_padded
                    x_direct = D_result.x_padded[:, 0] * 100
                    print(f"decomposing Raw Signal")
                    D_plots.plot_mode_decomp(x_direct, signal_direct,
                                             m_name='', amp_factor=args.modedecomp_factor[0],
                                             yaxis_factor=args.modedecomp_factor[1],
                                             time_window=decomp_tw, fft_dig_khz=decomp_f)
                    if args.plot:
                        plt.show()
                    else:
                        plt.close("all")
    else:
        shotnum = int(input("Enter shot number: "))
