"""
Synthetic Storm Generator v4
Generates balanced training data for cGAN
500 storms per category (Cat1-Cat5) = 2,500 total

Key improvement over v1-v3:
Storm position computed backwards from peak intensity
to guarantee the storm passes DIRECTLY over
Lee County center (26.3N, 81.8W) at peak intensity.
"""

import numpy as np
import pandas as pd
import os

LEE_CENTER_LAT = 26.3
LEE_CENTER_LON = -81.8

BINS = [
    {'name': 'Cat1', 'vmax_range': (64,  83),  'count': 500},
    {'name': 'Cat2', 'vmax_range': (83,  96),  'count': 500},
    {'name': 'Cat3', 'vmax_range': (96,  113), 'count': 500},
    {'name': 'Cat4', 'vmax_range': (113, 137), 'count': 500},
    {'name': 'Cat5', 'vmax_range': (137, 165), 'count': 500},
]


def generate_synthetic_storm_v4(vmax_kt, n_records=40):
    """
    Generate synthetic storm track passing directly over
    Lee County center at peak intensity.

    Parameters
    ----------
    vmax_kt : float
        Maximum sustained wind speed in knots
    n_records : int
        Number of track records to generate

    Returns
    -------
    pd.DataFrame
        Storm track with LAT, LON, VMAX_kt, PMIN_mb,
        PENV_mb, RMW_nmile, STORM_SPEED_kt, STORM_DIR_deg
    """
    rmw_nmile   = (np.random.uniform(10, 35) if vmax_kt > 100
                   else np.random.uniform(20, 55))
    penv_mb     = 1013.0
    speed_kt    = np.random.uniform(8, 15)
    peak_record = n_records // 2

    records = []
    for i in range(n_records):
        # Position — storm centered on Lee County at peak
        dist_from_peak = i - peak_record
        lat = LEE_CENTER_LAT - dist_from_peak * 0.25
        lon = LEE_CENTER_LON + np.random.uniform(-0.1, 0.1)

        # Intensity profile peaks AT Lee County
        adist = abs(dist_from_peak)
        if adist == 0:
            v = vmax_kt
        elif adist <= 5:
            v = vmax_kt * (1 - adist * 0.08)
        else:
            v = vmax_kt * max(0.3, 1 - adist * 0.08)

        records.append({
            'LAT':            lat,
            'LON':            lon,
            'VMAX_kt':        max(v, 34),
            'PMIN_mb':        1013 - (max(v, 34) * 0.9),
            'PENV_mb':        penv_mb,
            'RMW_nmile':      rmw_nmile,
            'STORM_SPEED_kt': speed_kt,
            'STORM_DIR_deg':  0.0,
        })

    return pd.DataFrame(records)


def generate_balanced_data(tracks, lat_coarse, lon_coarse,
                            lat_fine, lon_fine,
                            n_per_cat=500, save_dir=None):
    """
    Generate balanced synthetic training data for cGAN.

    Parameters
    ----------
    tracks : pd.DataFrame
        Historical storm tracks (used for reference only)
    lat_coarse, lon_coarse : np.ndarray
        Coarse resolution grid (22×21)
    lat_fine, lon_fine : np.ndarray
        Fine resolution grid (201×201)
    n_per_cat : int
        Number of storms per category (default 500)
    save_dir : str, optional
        Directory to save arrays to disk

    Returns
    -------
    X, Y, C : np.ndarray
        Normalized coarse, fine, condition arrays
    X_max : float
        Normalization factor (m/s)
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from hazard.wind_field import compute_track_peak_gust

    bins = [
        {'name': 'Cat1', 'vmax_range': (64,  83)},
        {'name': 'Cat2', 'vmax_range': (83,  96)},
        {'name': 'Cat3', 'vmax_range': (96,  113)},
        {'name': 'Cat4', 'vmax_range': (113, 137)},
        {'name': 'Cat5', 'vmax_range': (137, 165)},
    ]

    syn_coarse = []
    syn_fine   = []
    syn_params = []

    for bin_info in bins:
        vmin, vmax_v = bin_info['vmax_range']
        generated    = 0
        print(f"Generating {n_per_cat} {bin_info['name']} "
              f"storms ({vmin}-{vmax_v} kt)...")

        while generated < n_per_cat:
            vmax_kt  = np.random.uniform(vmin, vmax_v)
            storm_df = generate_synthetic_storm_v4(vmax_kt)

            coarse = compute_track_peak_gust(
                storm_df, lat_coarse, lon_coarse)
            fine   = compute_track_peak_gust(
                storm_df, lat_fine, lon_fine)

            # Skip if insufficient wind over domain
            if coarse.max() * 1.944 < vmin * 0.5:
                continue

            params = [
                vmax_kt / 150.0,
                storm_df['RMW_nmile'].iloc[0] / 100.0,
                float(storm_df['PMIN_mb'].min()) / 1013.0,
                LEE_CENTER_LAT / 35.0,
            ]

            if any(np.isnan(p) for p in params):
                continue

            syn_coarse.append(coarse)
            syn_fine.append(fine)
            syn_params.append(params)
            generated += 1

        print(f"  Done! {generated} pairs generated")

    X_syn  = np.array(syn_coarse)
    Y_syn  = np.array(syn_fine)
    C_syn  = np.array(syn_params)
    X_max  = X_syn.max()

    X = X_syn / X_max
    Y = Y_syn / X_max
    C = np.nan_to_num(C_syn, nan=0.5)

    print(f"\nTotal pairs : {len(X)}")
    print(f"X_max       : {X_max:.4f} m/s = {X_max*2.237:.1f} mph")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'X_balanced.npy'), X)
        np.save(os.path.join(save_dir, 'Y_balanced.npy'), Y)
        np.save(os.path.join(save_dir, 'C_balanced.npy'), C)
        np.save(os.path.join(save_dir, 'X_max_bal.npy'),
                np.array([X_max]))
        print(f"Saved to {save_dir}")

    return X, Y, C, X_max
