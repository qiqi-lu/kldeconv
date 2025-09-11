"""
List of checkpoints.
"""

checkpoints_v1 = {
    "F-actin-nonlinear-9": {
        "forward": {
            "n1_r1": "checkpoints/F-actin-nonlinear-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_gauss_9_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_500.pt",  # v1
        },
        "backward": {
            "n1_r1": "checkpoints/F-actin-nonlinear-9/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_2_ker_31_gauss_9_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1/epoch_10000.pt"  # v1
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-nonlinear-9/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "F-actin-nonlinear-2": {
        "forward": {
            "n1_r1": "checkpoints/v2/F-actin-nonlinear-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/F-actin-nonlinear-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "F-actin-nonlinear-1": {
        "forward": {
            "n1_r1": "checkpoints/v2/F-actin-nonlinear-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/F-actin-nonlinear-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-nonlinear-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "Microtubules2-9": {
        "forward": {
            "n1_r1": "checkpoints/Microtubules2-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_gauss_9_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_500.pt",  # v1
        },
        "backward": {
            "n1_r1": "checkpoints/Microtubules2-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_gauss_9_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1/epoch_10000.pt"  # v1
        },
        "dfcan": {
            "n1_r1": "checkpoints/Microtubules2-9/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "Microtubules2-2": {
        "forward": {
            "n1_r1": "checkpoints/v2/Microtubules2-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/Microtubules2-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "Microtubules2-1": {
        "forward": {
            "n1_r1": "checkpoints/v2/Microtubules2-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
            # "n1_r1": "checkpoints/v2/Microtubules2-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_norm_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/Microtubules2-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
            # "n1_r1": "checkpoints/v2/Microtubules2-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_norm_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/Microtubules2-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "CCPs-9": {
        "forward": {
            "n1_r1": "checkpoints/v2/CCPs-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/CCPs-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/CCPs-9/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "CCPs-2": {
        "forward": {
            "n1_r1": "checkpoints/v2/CCPs-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/CCPs-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "CCPs-1": {
        "forward": {
            "n1_r1": "checkpoints/v2/CCPs-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/CCPs-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/CCPs-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "ER-6": {
        "forward": {
            "n1_r1": "checkpoints/v2/ER-6/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/ER-6/kernelnet/backward/kernet_bs_1_lr_0.001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/ER-6/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "ER-2": {
        "forward": {
            "n1_r1": "checkpoints/v2/ER-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/ER-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "ER-1": {
        "forward": {
            "n1_r1": "checkpoints/v2/ER-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/ER-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/ER-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "F-actin-9": {
        "forward": {
            "n1_r1": "checkpoints/v2/F-actin-9/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/F-actin-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-9/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "F-actin-2": {
        "forward": {
            "n1_r1": "checkpoints/v2/F-actin-2/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/F-actin-2/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {},
    },
    "F-actin-1": {
        "forward": {
            "n1_r1": "checkpoints/v2/F-actin-1/kernelnet/forward/kernet_fp_bs_1_lr_0.001_ker_31_mse_over2_inter_normx_fft_ts_0_1_s100/epoch_499_499.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/v2/F-actin-1/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_2_ker_31_mse_over2_inter_norm_fft_ts_0_1/epoch_9999_9999.pt",
        },
        "dfcan": {
            "n1_r1": "checkpoints/F-actin-1/dfcan/dfcan_mae_bs_16_lr_0.001_id_0_1/epoch_14999_iter_29999.pt",
        },
    },
    "SimuBeads3D-128-31-0-0-1": {},
    "SimuBeads3D-128-31-05-1-1": {},
    "SimuBeads3D-128-31-05-1-03": {},
    "SimuBeads3D-128-31-05-1-01": {},
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
            "n1_r1": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1/epoch_10000.pt",
            "n1_r2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_1_2/epoch_10000.pt",
            "n1_r3": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_2_3/epoch_10000.pt",
            "n2_r1": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_2/epoch_10000.pt",
            "n2_r2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_2_4/epoch_10000.pt",
            "n2_r3": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_4_6/epoch_10000.pt",
            "n3_r1": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_3/epoch_10000.pt",  # [v1] + knwon fp
            "n3_r2": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_3_6/epoch_10000.pt",
            "n3_r3": "checkpoints/SimuMix3D-128-31-0-0-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_6_9/epoch_10000.pt",
        },
    },
    "SimuMix3D-128-31-05-1-1": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100/epoch_20.pt",
            "n2_r1": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/forward/kernet_fp_bs_2_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_2_s100/epoch_20.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1/epoch_10000.pt",
            "n1_r2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_1_2/epoch_10000.pt",
            "n1_r3": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_2_3/epoch_10000.pt",
            "n2_r1": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_2/epoch_10000.pt",
            "n2_r2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_2_4/epoch_10000.pt",
            "n2_r3": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_4_6/epoch_10000.pt",
            "n3_r1": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_3/epoch_10000.pt",
            "n3_r2": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_3_6/epoch_10000.pt",
            "n3_r3": "checkpoints/SimuMix3D-128-31-05-1-1/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_6_9/epoch_10000.pt",
        },
    },
    "SimuMix3D-128-31-05-1-03": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.3_ts_0_1_s100/epoch_20.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_0_1/epoch_10000.pt",
            "n1_r2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_1_2/epoch_10000.pt",
            "n1_r3": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_2_3/epoch_10000.pt",
            "n2_r1": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_0_2/epoch_10000.pt",
            "n2_r2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_2_4/epoch_10000.pt",
            "n2_r3": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_4_6/epoch_10000.pt",
            "n3_r1": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_0_3/epoch_10000.pt",
            "n3_r2": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_3_6/epoch_10000.pt",
            "n3_r3": "checkpoints/SimuMix3D-128-31-05-1-03/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.3_ts_6_9/epoch_10000.pt",
        },
    },
    "SimuMix3D-128-31-05-1-01": {
        "forward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/forward/kernet_fp_bs_1_lr_1_ker_31_gauss_0.5_poiss_1_sf_1_mse_over2_inter_normx_fft_ratio_0.1_ts_0_1_s100/epoch_20.pt",
        },
        "backward": {
            "n1_r1": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_0_1/epoch_10000.pt",
            "n1_r2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_1_2/epoch_10000.pt",
            "n1_r3": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_2_3/epoch_10000.pt",
            "n2_r1": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_0_2/epoch_10000.pt",
            "n2_r2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_2_4/epoch_10000.pt",
            "n2_r3": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_2_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_4_6/epoch_10000.pt",
            "n3_r1": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_0_3/epoch_10000.pt",
            "n3_r2": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_3_6/epoch_10000.pt",
            "n3_r3": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_3_lr_1e-06_iter_2_ker_31_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_0.1_ts_6_9/epoch_10000.pt",
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
    "Microtubule2-3d-1024": {
        "forward": {
            "n1_r1": "checkpoints/Microtubule2-3d-1024/kernelnet/forward/kernet_fp_bs_4_lr_0.01_ker_3_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_4_s100/epoch_500.pt",  # v1
        },
        "backward": {
            "n1_r1": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_4_lr_1e-05_iter_2_ker_3_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_4/epoch_10000.pt",  # v1
        },
        "rln": {
            "n1_r1": "checkpoints/Microtubule2-3d-1024/rln/rln_mae_bs_4_lr_0.01_id_0_1/epoch_699_iter_31499.pt"
        },
    },
    "Nuclear-pore-complex2-1024": {
        "forward": {
            "n1_r1": "checkpoints/Nuclear_Pore_complex2-1024/kernelnet/forward/kernet_fp_bs_4_lr_0.01_ker_3_gauss_0_poiss_0_sf_1_mse_over2_inter_normx_fft_ratio_1_ts_0_4_s100/epoch_500.pt",  # v1
        },
        "backward": {
            "n1_r1": "checkpoints/Nuclear_Pore_complex2-1024/kernelnet/backward/kernet_bs_4_lr_1e-05_iter_2_ker_3_gauss_0_poiss_0_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_4/epoch_10000.pt",  # v1
        },
        "rln": {
            "n1_r1": "checkpoints/Nuclear-pore-complex2-1024/rln/rln_mae_bs_4_lr_0.01_id_0_1/epoch_699_iter_31499.pt"
        },
    },
    "ZeroShotDeconvNet-simutrain-642": {
        "forward": {},
        "backward": {
            "n1_r1": "checkpoints/SimuMix3D-382-101-05-1-1-642/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1_642/epoch_4000.pt",
            "n9_r1": "checkpoints/SimuMix3D-382-101-05-1-1-642/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_9_642/epoch_4000.pt",
        },
    },
    "ZeroShotDeconvNet-simutrain-560": {
        "forward": {},
        "backward": {
            "n1_r1": "checkpoints/SimuMix3D-382-101-05-1-1-560/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_1_560/epoch_4000.pt",
            "n9_r1": "checkpoints/SimuMix3D-382-101-05-1-1-560/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_gauss_0.5_poiss_1_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_ts_0_9_560/epoch_4000.pt",
        },
    },
    "ZeroShotDeconvNet-ss-642": {
        "forward": {},
        "backward": {
            "n1_r1": "checkpoints/ZeroShotDeconvNet-ss-642/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_noise_0.5_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_101x101_642_ss/epoch_3000.pt",
        },
    },
    "ZeroShotDeconvNet-ss-560": {
        "forward": {},
        "backward": {
            "n1_r1": "checkpoints/ZeroShotDeconvNet-ss-560/kernelnet/backward/kernet_bs_1_lr_1e-07_iter_2_ker_101_noise_0.5_sf_1_lam_0.0_mse_over2_inter_norm_fft_ratio_1_101x101_560_ss/epoch_3000.pt",
        },
    },
}
