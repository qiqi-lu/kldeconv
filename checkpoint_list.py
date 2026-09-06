"""
List of checkpoints.
'n1_r1_iter2' represents the number of sample used for trianing is 1 and the first repeatation of the experiment.
The sample used for training for each repeatation is different.
"""

checkpoints_v1 = {
    "F-actin-nonlinear-9": {
        "forward": {
            "n1_r1": "checkpoints/F-actin-nonlinear-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_gauss_9_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_500.pt",  # v1
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/F-actin-nonlinear-9/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_31_gauss_9_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1/epoch_10000.pt"  # v1
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-nonlinear-9/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "F-actin-nonlinear-2": {
        "forward": {
            "n1_r1": "checkpoints/F-actin-nonlinear-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/F-actin-nonlinear-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "F-actin-nonlinear-1": {
        "forward": {
            "n1_r1": "checkpoints/F-actin-nonlinear-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/F-actin-nonlinear-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-nonlinear-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "Microtubules2-9": {
        "forward": {
            # "n1_r1": "checkpoints/Microtubules2-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_gauss_9_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_500.pt",  # v1
            "n1_r1": "checkpoints/Microtubules2-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_499_499.pt",
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/Microtubules2-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_gauss_9_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1/epoch_10000.pt",  # v1
            "n1_r1_iter1": "checkpoints/Microtubules2-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/Microtubules2-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/Microtubules2-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/Microtubules2-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/Microtubules2-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/Microtubules2-9/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "Microtubules2-6": {
        "forward": {
            "n1_r1": "checkpoints/Microtubules2-6/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter1": "checkpoints/Microtubules2-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/Microtubules2-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/Microtubules2-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/Microtubules2-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/Microtubules2-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/Microtubules2-6/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "Microtubules2-3": {
        "forward": {
            "n1_r1": "checkpoints/Microtubules2-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt",
        },
        "backward": {
            "n1_r1_iter1": "checkpoints/Microtubules2-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/Microtubules2-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/Microtubules2-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/Microtubules2-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/Microtubules2-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/Microtubules2-3/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "Microtubules2-2": {
        "forward": {
            "n1_r1": "checkpoints/Microtubules2-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/Microtubules2-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "Microtubules2-1": {
        "forward": {
            "n1_r1": "checkpoints/Microtubules2-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
            # "n1_r1": "checkpoints/Microtubules2-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_norm_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/Microtubules2-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
            # "n1_r1_iter2": "checkpoints/Microtubules2-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_norm_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/Microtubules2-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "CCPs-9": {
        "forward": {
            # "n1_r1": "checkpoints/CCPs-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
            "n1_r1": "checkpoints/CCPs-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_499_499.pt",
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/CCPs-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
            "n1_r1_iter1": "checkpoints/CCPs-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/CCPs-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/CCPs-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/CCPs-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/CCPs-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/CCPs-9/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "CCPs-6": {
        "forward": {
            "n1_r1": "checkpoints/CCPs-6/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter1": "checkpoints/CCPs-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/CCPs-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/CCPs-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/CCPs-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/CCPs-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/CCPs-6/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "CCPs-3": {
        "forward": {
            "n1_r1": "checkpoints/CCPs-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt",
        },
        "backward": {
            "n1_r1_iter1": "checkpoints/CCPs-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/CCPs-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/CCPs-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/CCPs-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/CCPs-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/CCPs-3/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "CCPs-2": {
        "forward": {
            "n1_r1": "checkpoints/CCPs-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/CCPs-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "CCPs-1": {
        "forward": {
            "n1_r1": "checkpoints/CCPs-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/CCPs-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/CCPs-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "ER-6": {
        "forward": {
            # "n1_r1": "checkpoints/ER-6/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
            "n1_r1": "checkpoints/ER-6/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_499_499.pt",
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/ER-6/kernelnet/backward/kernet_bs_1_lr_0.001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
            "n1_r1_iter1": "checkpoints/ER-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/ER-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/ER-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/ER-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/ER-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/ER-6/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "ER-3": {
        "forward": {
            "n1_r1": "checkpoints/ER-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt",
        },
        "backward": {
            "n1_r1_iter1": "checkpoints/ER-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/ER-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/ER-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/ER-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/ER-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/ER-3/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "ER-2": {
        "forward": {
            "n1_r1": "checkpoints/ER-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/ER-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "ER-1": {
        "forward": {
            "n1_r1": "checkpoints/ER-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/ER-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/ER-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "F-actin-9": {
        "forward": {
            # "n1_r1": "checkpoints/F-actin-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
            "n1_r1": "checkpoints/F-actin-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_499_499.pt",
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/F-actin-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
            "n1_r1_iter1": "checkpoints/F-actin-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/F-actin-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/F-actin-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/F-actin-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/F-actin-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-9/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "F-actin-6": {
        "forward": {
            "n1_r1": "checkpoints/F-actin-6/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter1": "checkpoints/F-actin-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/F-actin-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/F-actin-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/F-actin-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/F-actin-6/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-6/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "F-actin-3": {
        "forward": {
            "n1_r1": "checkpoints/F-actin-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt",
        },
        "backward": {
            "n1_r1_iter1": "checkpoints/F-actin-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/F-actin-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/F-actin-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/F-actin-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/F-actin-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-3/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "F-actin-2": {
        "forward": {
            "n1_r1": "checkpoints/F-actin-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/F-actin-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "F-actin-1": {
        "forward": {
            "n1_r1": "checkpoints/F-actin-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/F-actin-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    # --------------------------------------------------------------------------
    "SimuMix3D-128-31-0-0-1": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_20.pt",
            "n1_r2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_1_2_s100/epoch_20.pt",
            "n1_r3": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_2_3_s100/epoch_20.pt",
            "n1_r4": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_3_4_s100/epoch_20.pt",
            "n1_r5": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_4_5_s100/epoch_20.pt",
            "n2_r1": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_2_s100/epoch_20.pt",
            "n2_r2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_2_4_s100/epoch_20.pt",
            "n2_r3": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_4_6_s100/epoch_20.pt",
            "n2_r4": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_6_8_s100/epoch_20.pt",
            "n2_r5": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_8_10_s100/epoch_20.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1/epoch_10000.pt",
            "n1_r2_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_1_2/epoch_10000.pt",
            "n1_r3_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_2_3/epoch_10000.pt",
            "n2_r1_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_2/epoch_10000.pt",
            "n2_r2_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_2_4/epoch_10000.pt",
            "n2_r3_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_4_6/epoch_10000.pt",
            "n3_r1_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_3/epoch_10000.pt",  # [v1] + knwon fp
            "n3_r2_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_3_6/epoch_10000.pt",
            "n3_r3_iter2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_6_9/epoch_10000.pt",
        },
        "rln": {
            "n1_r1": "",
        },
    },
    "SimuMix3D-128-31-05-1-1": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_20.pt",
            "n2_r1": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_2_s100/epoch_20.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1/epoch_10000.pt",
            "n1_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_1_2/epoch_10000.pt",
            "n1_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_2_3/epoch_10000.pt",
            "n2_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_2/epoch_10000.pt",
            "n2_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_2_4/epoch_10000.pt",
            "n2_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_4_6/epoch_10000.pt",
            "n3_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_3/epoch_10000.pt",
            "n3_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_3_6/epoch_10000.pt",
            "n3_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_6_9/epoch_10000.pt",
        },
    },
    "SimuMix3D-128-31-05-1-03": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_0_1_s100/epoch_20.pt",
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_0_1/epoch_10000.pt",
            "n1_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_1_2/epoch_10000.pt",
            "n1_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_2_3/epoch_10000.pt",
            "n2_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_0_2/epoch_10000.pt",
            "n2_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_2_4/epoch_10000.pt",
            "n2_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_4_6/epoch_10000.pt",
            "n3_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_0_3/epoch_10000.pt",
            "n3_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_3_6/epoch_10000.pt",
            "n3_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_6_9/epoch_10000.pt",
        },
    },
    "SimuMix3D-128-31-05-1-01": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_0_1_s100/epoch_20.pt",
        },
        "backward": {
            "n1_r1_iter1": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_continue_median_in/epoch_9999_9999.pt",
            "n1_r1_iter3": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_continue_median_in/epoch_9999_9999.pt",
            "n1_r1_iter4": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_continue_median_in/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_continue_median_in/epoch_9999_9999.pt",
            # ------------------------------------------------------------------
            # "n1_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_continue_median_in/epoch_9999_9999.pt",
            "n2_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_2_v3_continue_median_in/epoch_5000_10000.pt",
            "n3_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_3_v3_continue_median_in/epoch_3333_10000.pt",
            "n4_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_4_v3_continue_median_in/epoch_2500_10000.pt",
            "n5_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_5_v3_continue_median_in/epoch_2000_10000.pt",
            # ------------------------------------------------------------------
            # "n1_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_0_1/epoch_10000.pt",
            # "n1_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_1_2/epoch_10000.pt",
            # "n1_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_2_3/epoch_10000.pt",
            # "n2_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_0_2/epoch_10000.pt",
            # "n2_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_2_4/epoch_10000.pt",
            # "n2_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_4_6/epoch_10000.pt",
            # "n3_r1_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_0_3/epoch_10000.pt",  # this one + known fp
            # "n3_r2_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_3_6/epoch_10000.pt",
            # "n3_r3_iter2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_6_9/epoch_10000.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-01/rln/rln_mae_bs_1_lr_0.01_id_0_1/epoch_29999_iter_29999.pt",
            "n2_r1": "checkpoints/SimuMix3D-128-31-05-1-01/rln/rln_mae_bs_1_lr_0.01_id_0_2/epoch_14999_iter_29999.pt",
            "n3_r1": "checkpoints/SimuMix3D-128-31-05-1-01/rln/rln_mae_bs_1_lr_0.01_id_0_3/epoch_9999_iter_29999.pt",
            "n4_r1": "checkpoints/SimuMix3D-128-31-05-1-01/rln/rln_mae_bs_1_lr_0.01_id_0_4/epoch_7500_iter_30000.pt",
            "n80_r1": "checkpoints/SimuMix3D-128-31-05-1-01/rln/rln_mae_bs_1_lr_0.01_id_0_80/epoch_375_iter_30000.pt",
        },
    },
    "SimuMix3D-256-31-0-0-1": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_50.pt",
            "n1_r2": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_1_2_s100/epoch_50.pt",
            "n1_r3": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_2_3_s100/epoch_50.pt",
            "n2_r1": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_2_s100/epoch_50.pt",
            "n2_r2": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_2_4_s100/epoch_50.pt",
            "n2_r3": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_4_6_s100/epoch_50.pt",
            "n3_r1": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_3_s100/epoch_50.pt",
            "n3_r2": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_3_6_s100/epoch_50.pt",
            "n3_r3": "checkpoints/SimuMix3D-256-31-0-0-1/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_6_9_s100/epoch_50.pt",
        },
        "backward": {},
    },
    "SimuMix3D-256-31-05-1-1": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_10.pt",
            "n1_r2": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_1_2_s100/epoch_10.pt",
            "n1_r3": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_2_3_s100/epoch_10.pt",
            "n2_r1": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_2_s100/epoch_10.pt",
            "n2_r2": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_2_4_s100/epoch_10.pt",
            "n2_r3": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_4_6_s100/epoch_10.pt",
            "n3_r1": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_3_s100/epoch_10.pt",
            "n3_r2": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_3_6_s100/epoch_10.pt",
            "n3_r3": "checkpoints/SimuMix3D-256-31-05-1-1/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_6_9_s100/epoch_10.pt",
        },
        "backward": {},
    },
    "SimuMix3D-256-31-05-1-03": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_0_1_s100/epoch_10.pt",
            "n1_r2": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_1_2_s100/epoch_10.pt",
            "n1_r3": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_2_3_s100/epoch_10.pt",
            "n2_r1": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_0_2_s100/epoch_10.pt",
            "n2_r2": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_2_4_s100/epoch_10.pt",
            "n2_r3": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_4_6_s100/epoch_10.pt",
            "n3_r1": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_0_3_s100/epoch_10.pt",
            "n3_r2": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_3_6_s100/epoch_10.pt",
            "n3_r3": "checkpoints/SimuMix3D-256-31-05-1-03/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_6_9_s100/epoch_10.pt",
        },
        "backward": {},
    },
    "SimuMix3D-256-31-05-1-01": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_0_1_s100/epoch_10.pt",
            "n1_r2": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_1_2_s100/epoch_10.pt",
            "n1_r3": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_2_3_s100/epoch_10.pt",
            "n2_r1": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_0_2_s100/epoch_10.pt",
            "n2_r2": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_2_4_s100/epoch_10.pt",
            "n2_r3": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_4_6_s100/epoch_10.pt",
            "n3_r1": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_0_3_s100/epoch_10.pt",
            "n3_r2": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_3_6_s100/epoch_10.pt",
            "n3_r3": "checkpoints/SimuMix3D-256-31-05-1-01/forward/kernet_fp_bs_3_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_6_9_s100/epoch_10.pt",
        },
        "backward": {},
    },
    "SimuMix3D-382-101-05-1-1-560": {},
    "SimuMix3D-382-101-05-1-1-642": {},
    # --------------------------------------------------------------------------
    "deepbacs-ecoli-ave2": {
        "forward": {
            "n1_r1": "checkpoints/deepbacs-ecoli-ave2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v2/epoch_499_499.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/deepbacs-ecoli-ave2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt"
        },
        "dfcan": {
            "n1_r1": "checkpoints/deepbacs-ecoli-ave2/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "deepbacs-saureus-ave2": {
        "forward": {
            "n1_r1": "checkpoints/deepbacs-saureus-ave2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v2/epoch_499_499.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/deepbacs-saureus-ave2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt"
        },
        "dfcan": {
            "n1_r1": "checkpoints/deepbacs-saureus-ave2/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    # --------------------------------------------------------------------------
    "biotisr-ccps-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-ccps-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-ccps-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-ccps-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-ccps-1/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-ccps-2": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-ccps-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-ccps-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-ccps-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-ccps-2/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-ccps-3": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-ccps-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-ccps-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-ccps-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-ccps-3/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_3000_iter_30000.pt"
        },
    },
    "biotisr-factin-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-factin-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-factin-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-factin-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-factin-1/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-factin-2": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-factin-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-factin-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-factin-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-factin-2/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-factin-3": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-factin-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-factin-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-factin-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-factin-3/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_3000_iter_30000.pt"
        },
    },
    "biotisr-factin-nonlinear-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-factin-nonlinear-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-factin-nonlinear-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-factin-nonlinear-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-factin-nonlinear-1/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-factin-nonlinear-2": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-factin-nonlinear-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-factin-nonlinear-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-factin-nonlinear-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-factin-nonlinear-2/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-factin-nonlinear-3": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-factin-nonlinear-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-factin-nonlinear-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-factin-nonlinear-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-factin-nonlinear-3/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-lysosomes-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-lysosomes-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-lysosomes-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-lysosomes-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-lysosomes-1/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-lysosomes-2": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-lysosomes-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-lysosomes-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-lysosomes-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-lysosomes-2/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-lysosomes-3": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-lysosomes-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-lysosomes-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-lysosomes-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-lysosomes-3/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-mt-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-mt-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-mt-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-mt-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-mt-1/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-mt-2": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-mt-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-mt-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-mt-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-mt-2/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-mt-3": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-mt-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-mt-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-mt-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-mt-3/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-mito-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-mito-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-mito-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-mito-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-mito-1/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-mito-2": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-mito-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-mito-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-mito-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-mito-2/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-mito-3": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-mito-3/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_(31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-mito-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-mito-3/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_5_ker_(31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/biotisr-mito-3/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    # --------------------------------------------------------------------------
    "w2s-0-sim-ave": {
        "forward": {
            "n1_r1": "checkpoints/w2s-0-sim-ave/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v2/epoch_499_499.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/w2s-0-sim-ave/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt"
        },
        "dfcan": {
            "n1_r1": "checkpoints/w2s-0-sim-ave/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "w2s-1-sim-ave": {
        "forward": {
            "n1_r1": "checkpoints/w2s-1-sim-ave/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v2/epoch_499_499.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/w2s-1-sim-ave/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt"
        },
        "dfcan": {
            "n1_r1": "checkpoints/w2s-1-sim-ave/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "w2s-2-sim-ave": {
        "forward": {
            "n1_r1": "checkpoints/w2s-2-sim-ave/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v2/epoch_499_499.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/w2s-2-sim-ave/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt"
        },
        "dfcan": {
            "n1_r1": "checkpoints/w2s-2-sim-ave/dfcan/dfcan_mae_bs_4_lr_0.001_v3_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    # --------------------------------------------------------------------------
    "Microtubule2-3d-1024": {
        "forward": {
            # "n1_r1": "checkpoints/Microtubule2-3d-1024/kernelnet/forward/kernet_fp_bs_4_lr_0.01_ker_3_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_4_s100/epoch_500.pt",  # v1
            # "n1_r1": "checkpoints/Microtubule2-3d-1024/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v3_median_in/epoch_999_999.pt",
            "n1_r1": "checkpoints/Microtubule2-3d-1024/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(5, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3_median_in/epoch_999_999.pt",
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_4_lr_1e-05_iter_2_ker_3_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_4/epoch_10000.pt",  # v1
            "n1_r1_iter1": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_9999_9999.pt",
            # "n1_r1_iter2": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_14999_14999.pt",
            "n1_r1_iter2": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(5, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_14999_14999.pt",
            "n1_r1_iter3": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_3_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_14999_14999.pt",
            "n1_r1_iter4": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_4_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_14999_14999.pt",
            # "n1_r1_iter5": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_14999_14999.pt",
            "n1_r1_iter5": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(5, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_14999_14999.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/Microtubule2-3d-1024/rln/rln_mae_bs_4_lr_0.01_id_0_1/epoch_699_iter_31499.pt"
        },
    },
    "Nuclear-pore-complex2-1024": {
        "forward": {
            # "n1_r1": "checkpoints/Nuclear-pore-complex2-1024/kernelnet/forward/kernet_fp_bs_4_lr_0.01_ker_3_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_4_s100/epoch_500.pt",  # v1
            "n1_r1": "checkpoints/Nuclear-pore-complex2-1024/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(5, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3_median_in/epoch_999_999.pt",
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/Nuclear-pore-complex2-1024/kernelnet/backward/kernet_bs_4_lr_1e-05_iter_2_ker_3_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_4/epoch_10000.pt",  # v1
            "n1_r1_iter2": "checkpoints/Nuclear-pore-complex2-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(5, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_14999_14999.pt",
            "n1_r1_iter5": "checkpoints/Nuclear-pore-complex2-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(5, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_14999_14999.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/Nuclear-pore-complex2-1024/rln/rln_mae_bs_4_lr_0.01_id_0_1/epoch_699_iter_31499.pt"
        },
    },
    # --------------------------------------------------------------------------
    "SirDNA-1024": {
        "forward": {
            "n1_r1": "checkpoints/SirDNA-1024/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v2/epoch_499_499.pt"
            # "n1_r1": "checkpoints/SirDNA-1024/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_31_mse_over2_inter_norm_fft_ts_0_1_s100_v2/epoch_499_499.pt"
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/SirDNA-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/SirDNA-1024/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt",
            # "n1_r1_iter2": "checkpoints/SirDNA-1024/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_norm_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt",
        },
    },
    # --------------------------------------------------------------------------
    "ZeroShotDeconvNet-simutrain-642": {
        "forward": {},
        "backward": {
            "n1_r1_iter2": "checkpoints/SimuMix3D-382-101-05-1-1-642/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1_642/epoch_4000.pt",
            "n9_r1_iter2": "checkpoints/SimuMix3D-382-101-05-1-1-642/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_9_642/epoch_4000.pt",
        },
    },
    "ZeroShotDeconvNet-simutrain-560": {
        "forward": {},
        "backward": {
            "n1_r1_iter2": "checkpoints/SimuMix3D-382-101-05-1-1-560/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1_560/epoch_4000.pt",
            "n9_r1_iter2": "checkpoints/SimuMix3D-382-101-05-1-1-560/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_9_560/epoch_4000.pt",
        },
    },
    # --------------------------------------------------------------------------
    "ZeroShotDeconvNet-ss-642": {
        "forward": {},
        "backward": {
            "n1_r1_iter2": "checkpoints/ZeroShotDeconvNet-ss-642/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_noise_0.5_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_101x101_642_ss/epoch_3000.pt",
        },
    },
    "ZeroShotDeconvNet-ss-560": {
        "forward": {},
        "backward": {
            "n1_r1_iter2": "checkpoints/ZeroShotDeconvNet-ss-560/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_noise_0.5_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_101x101_560_ss/epoch_3000.pt",
        },
    },
    # --------------------------------------------------------------------------
    "biotisr-3d-factin-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-3d-factin-1/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(5, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-3d-factin-1/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(5, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-3d-factin-1/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(5, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/biotisr-3d-factin-1/rln/rln_mae_bs_4_lr_0.01_v3_even_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-3d-factin-2": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-3d-factin-2/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(5, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-3d-factin-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(5, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-3d-factin-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(5, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/biotisr-3d-factin-2/rln/rln_mae_bs_4_lr_0.01_v3_even_id_0_1/epoch_2999_iter_29999.pt",
        },
    },
    # --------------------------------------------------------------------------
    "biotisr-3d-mt-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-3d-mt-1/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(7, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-3d-mt-1/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(7, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-3d-mt-1/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(7, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/biotisr-3d-mt-1/rln/rln_mae_bs_4_lr_0.01_v3_even_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-3d-mt-2": {
        "forward": {
            # "n1_r1": "checkpoints/biotisr-3d-mt-2/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100_v2/epoch_499_499.pt"
            "n1_r1": "checkpoints/biotisr-3d-mt-2/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(7, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/biotisr-3d-mt-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v2/epoch_9999_9999.pt"
            "n1_r1_iter2": "checkpoints/biotisr-3d-mt-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(7, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-3d-mt-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(7, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/biotisr-3d-mt-2/rln/rln_mae_bs_4_lr_0.01_v3_even_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    # --------------------------------------------------------------------------
    "biotisr-3d-mito-1": {
        "forward": {
            "n1_r1": "checkpoints/biotisr-3d-mito-1/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(9, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            "n1_r1_iter2": "checkpoints/biotisr-3d-mito-1/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(9, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-3d-mito-1/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(9, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/biotisr-3d-mito-1/rln/rln_mae_bs_4_lr_0.01_v3_even_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    "biotisr-3d-mito-2": {
        "forward": {
            # "n1_r1": "checkpoints/biotisr-3d-mito-2/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(9, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
            "n1_r1": "checkpoints/biotisr-3d-mito-2/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(9, 31, 31)_mse_over2_inter_normx_fft_ts_0_1_s100_v3_dark/epoch_999_999.pt"
            # "n1_r1": "checkpoints/biotisr-3d-mito-2/kernelnet/forward/kernet_fp_bs_1_lr_0.01_ker_(9, 101, 101)_mse_over2_inter_normx_fft_ts_0_1_s100_v3/epoch_999_999.pt"
        },
        "backward": {
            # "n1_r1_iter2": "checkpoints/biotisr-3d-mito-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(9, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter2": "checkpoints/biotisr-3d-mito-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_(9, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_dark/epoch_9999_9999.pt",
            # "n1_r1_iter2": "checkpoints/biotisr-3d-mito-2/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_(9, 101, 101)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            # "n1_r1_iter5": "checkpoints/biotisr-3d-mito-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(9, 31, 31)_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
            "n1_r1_iter5": "checkpoints/biotisr-3d-mito-2/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_5_ker_(9, 31, 31)_mae_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_dark/epoch_9999_9999.pt",
        },
        "rln": {
            "n1_r1": "checkpoints/biotisr-3d-mito-2/rln/rln_mae_bs_4_lr_0.01_v3_even_id_0_1/epoch_2999_iter_29999.pt"
        },
    },
    # --------------------------------------------------------------------------
}
